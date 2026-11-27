"""Experts (one per modality), the EMA target, and the uncertainty-aware regression head.

Every expert maps a feature sequence to a diagonal Gaussian belief (mu, log var) over a shared 64-d latent.
Expert variants:
  Expert(n_heads=1)  : one Gaussian predictor head (V-JEPA, reconstruction, alignment, supervised)
  Expert(n_heads=K)  : predictor ensemble, K heads on one backbone; belief = moment-matched mixture of the heads
                       (variance = mean head variance + variance of head means)
  SJEPAExpert        : S-JEPA (Giakoumoglou et al. 2026): stochastic latent z with a context-conditioned prior and
                       a training-only posterior; belief = mean and variance of predictions over prior samples
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import RT

LV_MIN, LV_MAX = -6.0, 4.0          # log-variance bounds: sigma in [0.05, 7.4]


def bound_lv(lv):
    return LV_MIN + (LV_MAX - LV_MIN) * torch.sigmoid(lv)


class SeqEncoder(nn.Module):
    """Small transformer over a frozen feature sequence -> pooled d_lat vector."""
    def __init__(self, d_in, max_len, c):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, c['d_model']))
        self.pos = nn.Parameter(torch.randn(1, max_len, c['d_model']) * 0.02)
        layer = nn.TransformerEncoderLayer(c['d_model'], 4, 2 * c['d_model'], c['dropout'],
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, c['n_layers'], enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(c['d_model']), nn.Linear(c['d_model'], c['d_lat']))

    def forward(self, x, mask):
        h = self.inp(x) + self.pos[:, :x.size(1)]
        h = self.tf(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return self.out((h * m).sum(1) / m.sum(1).clamp(min=1))


class Expert(nn.Module):
    """spherical=True (alignment): mu on the sphere of radius sqrt(d), variance from a separate head."""
    def __init__(self, d_in, max_len, c, spherical=False, n_heads=1):
        super().__init__()
        d, w = c['d_lat'], 2 * c['d_model']
        self.d, self.spherical, self.K = d, spherical, n_heads
        self.backbone = SeqEncoder(d_in, max_len, c)
        out = d if spherical else 2 * d
        self.pred = nn.Sequential(nn.Linear(d, w), nn.GELU(), nn.Linear(w, out * n_heads))
        if spherical:
            self.varhead = nn.Sequential(nn.Linear(d, w), nn.GELU(), nn.Linear(w, d))

    def heads(self, h, detach_var=False):
        """All heads: mu, lv of shape (K, B, d)."""
        if self.spherical:
            mu = math.sqrt(self.d) * F.normalize(self.pred(h), dim=-1)
            lv = self.varhead(h.detach() if detach_var else h)
            return mu.unsqueeze(0), bound_lv(lv).unsqueeze(0)
        o = self.pred(h).view(h.shape[0], self.K, 2 * self.d).transpose(0, 1)
        mu, lv = o.chunk(2, -1)
        return mu, bound_lv(lv)

    def head(self, h, detach_var=False):
        mu, lv = self.heads(h, detach_var)
        if self.K == 1:
            return mu[0], lv[0]
        var = lv.exp().mean(0) + mu.var(0, unbiased=False)       # law of total variance over the heads
        return mu.mean(0), torch.log(var)

    def forward(self, x, mask):
        return self.head(self.backbone(x, mask))


class SJEPAExpert(nn.Module):
    """S-JEPA expert. prior p(z | h), posterior q(z | h, sg(target)) (training only), predictor g(h, z) -> latent.
    The belief is the mean and variance of g(h, z) over `n_samples` prior draws, so the PoE and head can use it."""
    def __init__(self, d_in, max_len, c):
        super().__init__()
        d, w, dz = c['d_lat'], 2 * c['d_model'], c['sjepa_dz']
        self.d, self.dz, self.n_samples, self.K = d, dz, c['sjepa_samples'], 1
        self.backbone = SeqEncoder(d_in, max_len, c)
        self.prior = nn.Sequential(nn.Linear(d, w), nn.GELU(), nn.Linear(w, 2 * dz))
        self.post = nn.Sequential(nn.Linear(2 * d, w), nn.GELU(), nn.Linear(w, 2 * dz))
        self.zproj = nn.Linear(dz, d)
        self.g = nn.Sequential(nn.Linear(d, w), nn.GELU(), nn.Linear(w, d))

    def predict(self, h, z):
        return self.g(h + self.zproj(z))

    def prior_params(self, h):
        mu, lv = self.prior(h).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def post_params(self, h, target):
        mu, lv = self.post(torch.cat([h, target.detach()], -1)).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def head(self, h, detach_var=False, n=None):
        n = n or self.n_samples
        pm, plv = self.prior_params(h)
        z = pm.unsqueeze(0) + (0.5 * plv).exp().unsqueeze(0) * torch.randn(n, *pm.shape, device=h.device)
        preds = self.predict(h.unsqueeze(0).expand(n, -1, -1), z)              # (n, B, d)
        var = preds.var(0, unbiased=True).clamp(min=math.exp(LV_MIN), max=math.exp(LV_MAX))
        return preds.mean(0), torch.log(var)

    def forward(self, x, mask):
        return self.head(self.backbone(x, mask))


def make_experts(mods, c, objective='vjepa'):
    if objective == 'sjepa':
        return nn.ModuleDict({m: SJEPAExpert(RT.DIMS[m], RT.LENS[m], c) for m in mods})
    K = c['pe_heads'] if objective == 'vjepa_pe' else 1
    return nn.ModuleDict({m: Expert(RT.DIMS[m], RT.LENS[m], c, spherical=(objective == 'align'), n_heads=K)
                          for m in mods})


def expert_beliefs(experts, b, masks=None, xs=None):
    """Stack all experts' beliefs -> mus, lvs: (B, M, d). masks / xs override a modality's mask / input."""
    mus, lvs = [], []
    for m in experts:
        x = xs[m] if xs and m in xs else b[m]
        mk = masks[m] if masks and m in masks else b[m + 'm']
        mu, lv = experts[m](x, mk)
        mus.append(mu)
        lvs.append(lv)
    return torch.stack(mus, 1), torch.stack(lvs, 1)


class Target(nn.Module):
    """EMA target built from the text expert. Always runs without dropout.
    kind='det': EMA text encoder -> LayerNorm -> a point z (variance 0)
    kind='vt' : EMA of the whole text expert -> its Gaussian belief N(mu, var) (variational target)"""
    def __init__(self, module, kind):
        super().__init__()
        self.m, self.kind = module, kind
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x, mask):
        if self.kind == 'det':
            h = self.m(x, mask)
            z = F.layer_norm(h, (h.shape[-1],))
            return z, torch.zeros_like(z)
        mu, lv = self.m(x, mask)
        return mu, lv.exp()

    @torch.no_grad()
    def ema_from(self, online, mom):
        for pt, po in zip(self.m.parameters(), online.parameters()):
            pt.mul_(mom).add_(po.detach(), alpha=1 - mom)


class Head(nn.Module):
    """y ~ N(w.mu + b, alpha * sum_i w_i^2 var_i + s^2): latent uncertainty propagated analytically."""
    def __init__(self, d_lat):
        super().__init__()
        self.f = nn.Linear(d_lat, 1)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.log_s = nn.Parameter(torch.tensor(-0.5))

    def forward(self, mu, var):
        w = self.f.weight.squeeze(0)
        m = self.f(mu).squeeze(-1)
        v_lat = (var * w ** 2).sum(-1)
        return m, self.log_alpha.exp() * v_lat + torch.exp(2 * self.log_s), v_lat


def gauss_nll(y, m, v):
    return 0.5 * (torch.log(2 * math.pi * v) + (y - m) ** 2 / v)


def expected_nll(z, vt, mu, var):
    """E_{y ~ N(z, vt)}[-log N(y; mu, var)]: the NLL at z plus vt / (2 var). vt = 0 gives the plain NLL."""
    return 0.5 * (torch.log(2 * math.pi * var) + ((z - mu) ** 2 + vt) / var)
