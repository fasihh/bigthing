"""Fusion methods. Input: expert beliefs mus, lvs (B, M, d) and availability av (B, M).
Latent-level fusions return a fused Gaussian (mu, var, share); the head then maps it to the label.
MoNIG is output-level: it returns the label distribution directly.

Baselines (from the literature)
  concat   : early fusion, MLP on [mu_m, log var_m, available_m] -> Gaussian latent (ignores the belief semantics)
  late_avg : average of the available experts' means and variances
  moe      : uniform mixture of available experts, moment-matched (Shi et al. 2019 MMVAE-style)
  attn     : learned attention weights over experts, moment-matched mixture
  poe      : product of experts with an N(0, I) prior expert (Wu & Goodman 2018 MVAE; Huang 2026 BJEPA)
  poe_pc   : PoE with the duplicated N(0, I) priors divided out
  monig    : Mixture of Normal-Inverse-Gamma fusion (Ma et al., NeurIPS 2021), evidential, output-level
Novel (this project)
  dapoe    : disagreement-aware PoE: var += softplus(g) * precision-weighted spread of expert means
  rpoe     : robust PoE: precision_m *= 1 / (1 + softplus(k) * d2_m), d2_m = distance to leave-one-out consensus
  rdapoe   : rpoe + dapoe
  mpoe     : marginal-corrected PoE: prior = the target's marginal distribution, duplicated priors divided out
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = ['concat', 'late_avg', 'moe', 'attn', 'poe', 'poe_pc', 'dapoe', 'rpoe', 'rdapoe', 'mpoe']
OUTPUT = ['monig']
BASELINES = ['concat', 'late_avg', 'moe', 'attn', 'poe', 'poe_pc', 'monig']
NOVEL = ['dapoe', 'rpoe', 'rdapoe', 'mpoe']


def poe_core(mus, lvs, av, prior_mu=None, prior_prec=None, corrected=False, floor=0.1):
    """Gaussian PoE. corrected=True: every expert already contains the prior once, so n - 1 copies are removed."""
    prec = torch.exp(-lvs) * av.unsqueeze(-1)
    if prior_prec is None:
        prior_prec, prior_mu = torch.ones_like(mus[:, 0]), torch.zeros_like(mus[:, 0])
    n = av.sum(1, keepdim=True)
    f = (1 - n) if corrected else torch.ones_like(n)
    P = prec.sum(1) + prior_prec * f
    if corrected:
        P = torch.maximum(P, floor * prior_prec)
    mu = ((mus * prec).sum(1) + prior_mu * prior_prec * f) / P
    share = (prec / P.unsqueeze(1)).mean(-1)
    return mu, 1.0 / P, share, prec, P


def poe(mus, lvs, av):
    return poe_core(mus, lvs, av)[:3]


def mixture(mus, lvs, w):
    """Moment-matched Gaussian of a mixture with weights w (B, M)."""
    wu = w.unsqueeze(-1)
    mu = (wu * mus).sum(1)
    second = (wu * (lvs.exp() + mus ** 2)).sum(1)
    return mu, (second - mu ** 2).clamp(min=1e-6), w


class Fusion(nn.Module):
    output_level = False

    def __init__(self, kind, d, M, marginal=None):
        super().__init__()
        self.kind = kind
        if kind == 'attn':
            self.gate = nn.Sequential(nn.Linear(2 * d, 64), nn.GELU(), nn.Linear(64, 1))
        if kind == 'concat':
            self.net = nn.Sequential(nn.Linear(M * (2 * d + 1), 256), nn.GELU(), nn.Linear(256, 2 * d))
        if kind in ('dapoe', 'rdapoe'):
            self.g = nn.Parameter(torch.tensor(-2.0))
        if kind in ('rpoe', 'rdapoe'):
            self.k = nn.Parameter(torch.tensor(0.0))
        if kind == 'mpoe':
            if marginal is None:
                raise ValueError('mpoe needs the target marginal (objectives with an EMA target only)')
            self.register_buffer('m_mu', marginal[0].clone())
            self.register_buffer('m_prec', 1.0 / marginal[1].clamp(min=1e-4))

    def forward(self, mus, lvs, av):
        k, B = self.kind, mus.shape[0]
        if k == 'poe':
            return poe(mus, lvs, av)
        if k == 'poe_pc':
            return poe_core(mus, lvs, av, corrected=True, floor=0.1)[:3]
        if k == 'mpoe':
            return poe_core(mus, lvs, av, self.m_mu.expand(B, -1), self.m_prec.expand(B, -1),
                            corrected=True, floor=0.25)[:3]
        if k == 'moe':
            return mixture(mus, lvs, av / av.sum(1, keepdim=True).clamp(min=1))
        if k == 'attn':
            logits = self.gate(torch.cat([mus, lvs], -1)).squeeze(-1)
            return mixture(mus, lvs, torch.softmax(logits.masked_fill(av == 0, -1e9), 1))
        if k == 'late_avg':
            w = av / av.sum(1, keepdim=True).clamp(min=1)
            return (w.unsqueeze(-1) * mus).sum(1), (w.unsqueeze(-1) * lvs.exp()).sum(1), w
        if k == 'concat':
            a = av.unsqueeze(-1)
            mu, lv = self.net(torch.cat([mus * a, lvs * a, a], -1).flatten(1)).chunk(2, -1)
            return mu, torch.exp(-6 + 10 * torch.sigmoid(lv)), av / av.sum(1, keepdim=True).clamp(min=1)
        # robust / disagreement-aware PoE
        lvs_use = lvs
        if k in ('rpoe', 'rdapoe'):
            prec = torch.exp(-lvs) * av.unsqueeze(-1)
            P0 = prec.sum(1, keepdim=True)
            Pm = (P0 - prec).clamp(min=1e-6)
            mu_loo = ((mus * prec).sum(1, keepdim=True) - prec * mus) / Pm
            d2 = ((mus - mu_loo) ** 2 / (lvs.exp() + 1.0 / Pm)).mean(-1)
            others = av.sum(1, keepdim=True) - av
            w = torch.where(others > 0, 1.0 / (1.0 + F.softplus(self.k) * d2), torch.ones_like(d2))
            lvs_use = lvs - torch.log(w.clamp(min=1e-6)).unsqueeze(-1)
        mu, var, share, prec, P = poe_core(mus, lvs_use, av)
        if k in ('dapoe', 'rdapoe'):
            var = var + F.softplus(self.g) * ((prec / P.unsqueeze(1)) * (mus - mu.unsqueeze(1)) ** 2).sum(1)
        return mu, var, share

    def info(self):
        return dict(gamma=F.softplus(self.g).item() if hasattr(self, 'g') else float('nan'),
                    kappa=F.softplus(self.k).item() if hasattr(self, 'k') else float('nan'))


# ----------------------------------------------------------------------------- MoNIG (output level)
def nig_nll(y, g, nu, a, b):
    """Negative log-likelihood of y under the Student-t marginal of NIG(g, nu, a, b) (Amini et al. 2020)."""
    om = 2 * b * (1 + nu)
    return (0.5 * torch.log(math.pi / nu) - a * torch.log(om) + (a + 0.5) * torch.log(nu * (y - g) ** 2 + om)
            + torch.lgamma(a) - torch.lgamma(a + 0.5))


def nig_reg(y, g, nu, a):
    return (y - g).abs() * (2 * nu + a)


def nig_sum(p1, p2):
    """NIG summation (Ma et al. 2021, Def. 2): combines two modality NIGs into one."""
    g1, n1, a1, b1 = p1
    g2, n2, a2, b2 = p2
    nu = n1 + n2
    g = (n1 * g1 + n2 * g2) / nu
    return g, nu, a1 + a2 + 0.5, b1 + b2 + 0.5 * n1 * (g1 - g) ** 2 + 0.5 * n2 * (g2 - g) ** 2


class MoNIG(nn.Module):
    """Each modality's belief -> NIG(gamma, nu, alpha, beta) over the label; available NIGs are summed.
    Predictive mean = gamma; predictive variance (Student-t) = beta (1 + nu) / (nu (alpha - 1))."""
    output_level = True

    def __init__(self, d, M):
        super().__init__()
        self.kind = 'monig'
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Linear(128, 4))
                                    for _ in range(M)])

    def per_modality(self, mus, lvs):
        out = []
        for i, h in enumerate(self.heads):
            o = h(torch.cat([mus[:, i], lvs[:, i]], -1))
            g, nu, a, b = o.unbind(-1)
            out.append((g, F.softplus(nu) + 1e-3, F.softplus(a) + 1.0 + 1e-3, F.softplus(b) + 1e-3))
        return out

    def combine(self, params, av):
        """Sequential NIG summation over available modalities (unavailable ones are skipped)."""
        acc = None
        for i, p in enumerate(params):
            on = av[:, i] > 0
            if acc is None:
                acc = tuple(torch.where(on, x, torch.full_like(x, float('nan'))) for x in p)
                continue
            first = torch.isnan(acc[0])
            s = nig_sum(tuple(torch.nan_to_num(x, nan=1.0) for x in acc), p)
            acc = tuple(torch.where(on & ~first, s[j], torch.where(on & first, p[j], acc[j])) for j in range(4))
        return acc

    def forward(self, mus, lvs, av):
        params = self.per_modality(mus, lvs)
        g, nu, a, b = self.combine(params, av)
        var = b * (1 + nu) / (nu * (a - 1))
        share = (torch.stack([p[1] for p in params], 1) * av)
        share = share / share.sum(1, keepdim=True).clamp(min=1e-6)
        return g, var, share, (g, nu, a, b), params

    def loss(self, y, mus, lvs, av, reg):
        g, var, share, fused, params = self(mus, lvs, av)
        tot = nig_nll(y, *fused) + reg * nig_reg(y, fused[0], fused[1], fused[2])
        for i, p in enumerate(params):                      # MoNIG also trains every modality's own NIG
            li = nig_nll(y, *p) + reg * nig_reg(y, p[0], p[1], p[2])
            tot = tot + li * av[:, i]
        return tot

    def info(self):
        return dict(gamma=float('nan'), kappa=float('nan'))


def make_fusion(kind, d, M, marginal=None):
    if kind == 'monig':
        return MoNIG(d, M)
    return Fusion(kind, d, M, marginal)
