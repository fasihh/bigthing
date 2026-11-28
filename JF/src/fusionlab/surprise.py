"""Surprise-tempered product of experts (pilot for Part II).

In V-JEPA every expert predicts the same target latent, so the product of the OTHER experts is a cross-modal
prediction of what expert m should believe. The surprise of expert m is how far its belief lies from that
prediction (mean per-dimension Mahalanobis distance under the combined variance). Surprise is standardised with
clean training clips; an expert whose surprise is unusually high has its precision scaled down
(log-variance raised), so the product relies on the others and the fused uncertainty rises. No labels, no retraining.

Variants
  sigma     : the model's own fused sd (reference: the blindness found in Part I)
  stp-k     : tempering with fixed strength k above the clean 99th percentile, w = min(1, (c / z)^k)
  stp-tuned : k chosen on validation clips with self-generated corruptions (noise lambda=1, random swap)
"""
import math
import zlib

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from .batching import get_batch
from .models import expert_beliefs
from .runtime import BASE, RT
from .stage2 import fused_output

Z90 = norm.ppf(0.95)


def conditions():
    out = [dict(name='clean', kind='clean')]
    for m in BASE:
        out += [dict(name=f'noise[{m}]', kind='noise', m=m, lam=1.0),
                dict(name=f'noise[{m}]@0.5', kind='noise', m=m, lam=0.5),
                dict(name=f'swap[{m}]', kind='swap', m=m),
                dict(name=f'dead[{m}]', kind='dead', m=m)]
    for p in (('t', 'a'), ('t', 'v'), ('a', 'v')):
        out.append(dict(name=f'noise2[{"".join(p)}]', kind='noise2', ms=p))
    return out


def _corrupt(b, cond, split, idx, g, perm, feat_mean):
    xs, masks = {}, {}
    k = cond['kind']
    for m in cond.get('ms', (cond.get('m'),) if 'm' in cond else ()):
        x, mk = b[m], b[m + 'm']
        mkf = mk.unsqueeze(-1).float()
        if k in ('noise', 'noise2'):
            lam = cond.get('lam', 1.0)
            eps = torch.randn(x.shape, device=RT.DEV, generator=g) * RT.FEAT_STD[m]
            xs[m] = (math.sqrt(1 - lam) * x + math.sqrt(lam) * eps) * mkf
        elif k == 'dead':
            xs[m] = feat_mean[m].expand_as(x) * mkf
        elif k == 'swap':
            j = perm[idx]
            xs[m] = RT.DATA[split][m][j].to(RT.DEV)
            masks[m] = RT.DATA[split][m + 'm'][j].to(RT.DEV)
    return xs, masks


@torch.no_grad()
def beliefs(model, split, cond, perm, feat_mean, seed=123):
    """Expert beliefs (mus, lvs) under a condition, plus labels."""
    experts = model['experts']
    g = torch.Generator(device=RT.DEV).manual_seed(seed + zlib.crc32(cond['name'].encode()) % 10_000)
    n = RT.N_SPLIT[split]
    MU, LV = [], []
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(n, i + 1024))
        b = get_batch(split, idx, BASE, g)
        xs, masks = _corrupt(b, cond, split, idx, g, perm, feat_mean)
        mus, lvs = expert_beliefs(experts, b, masks, xs)
        MU.append(mus.cpu()); LV.append(lvs.cpu())
    return torch.cat(MU), torch.cat(LV), RT.DATA[split]['y'].numpy().astype(np.float64)


def surprise(mus, lvs):
    """(B, M): mean per-dimension squared distance of each expert's mean from the leave-one-out product of the others."""
    prec = torch.exp(-lvs)
    out = []
    for m in range(mus.shape[1]):
        keep = [k for k in range(mus.shape[1]) if k != m]
        P = 1.0 + prec[:, keep].sum(1)
        mu_o = (prec[:, keep] * mus[:, keep]).sum(1) / P
        v = 1.0 / P + lvs[:, m].exp()
        out.append((((mus[:, m] - mu_o) ** 2) / v).mean(-1))
    return torch.stack(out, 1)


@torch.no_grad()
def fused(model, mus, lvs, w=None):
    """Label mean / variance from the model's own fusion + head, with expert precisions scaled by w (B, M)."""
    if w is not None:
        lvs = lvs - torch.log(w.clamp(min=1e-6)).unsqueeze(-1)
    av = torch.ones(mus.shape[0], mus.shape[1])
    M_, V_ = [], []
    for i in range(0, len(mus), 2048):
        m, v, *_ = fused_output(model['fusion'], model['head'], mus[i:i + 2048].to(RT.DEV), lvs[i:i + 2048].to(RT.DEV),
                                av[i:i + 2048].to(RT.DEV))
        M_.append(m.cpu()); V_.append(v.cpu())
    return torch.cat(M_).numpy().astype(np.float64), torch.cat(V_).numpy().astype(np.float64)


class Surprise:
    def fit(self, mus, lvs, q=0.99):
        s = surprise(mus, lvs)
        self.mean, self.sd = s.mean(0), s.std(0) + 1e-6
        z = (s - self.mean) / self.sd
        self.c = torch.quantile(z, q, dim=0)
        return self

    def z(self, mus, lvs):
        return (surprise(mus, lvs) - self.mean) / self.sd

    def weights(self, mus, lvs, k):
        z = self.z(mus, lvs)
        r = (z / self.c).clamp(min=1e-6)
        return torch.where(z > self.c, r ** (-k), torch.ones_like(r))


def metrics(y, mu, v, tau):
    v = v * tau
    return dict(MAE=np.mean(np.abs(y - mu)), NLL=np.mean(0.5 * np.log(2 * np.pi * v) + 0.5 * (y - mu) ** 2 / v),
                Cov90=np.mean(np.abs(y - mu) <= Z90 * np.sqrt(v)), SD=np.mean(np.sqrt(v)))


def auroc(a, b):
    return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
