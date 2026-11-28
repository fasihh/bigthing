"""Cross-Examined Fusion (CXF): a modality's weight is earned by being corroborated by the other senses.

FYP-2 method (Phase A: no new training).
  * Auditor = frozen stage-1 experts (never sees labels). Worker = any PoE stage-2 model (frozen or fine-tuned).
  * Witness: PCA summaries o_m (k dims) of each auditor expert's belief mean; a clean joint Gaussian N(m, C) is fitted
    on clean, unlabelled data (validation half A). No corruption is ever seen.
  * Switch posterior: c_m = 1 means "modality m is lying". Under c_m = 1 its block is decoupled and broadened
    (kappa C_mm). log p(c | o) = (1/T) log p(o | c) + log prior(c). T is set on validation half B so that the
    clean false-alarm rate (mean over m of [r_m > 0.5]) is about 1%.
  * Prediction = Bayesian model averaging over the worker's subset predictions (PoE_R -> head), weighted by
    p(R | o); the worker was trained on random modality subsets, so every subset prediction is in-distribution.
  * CXF-C: per-subset (Mondrian) conformal quantiles calibrated on CLEAN validation with declared-missing subsets,
    mixed by p(R | o): a corrupted modality is calibrated as if it were missing.
"""
import itertools
import math

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from .batching import all_subsets, get_batch
from .evaluate import (apply_condition, conf_q, interval_metrics, point_metrics, prob_metrics, z_lev, levels)
from .models import expert_beliefs
from .runtime import NAMES, RT, set_seed
from .stage2 import fused_output


@torch.no_grad()
def _beliefs(experts, split, cond=None, seed=123):
    set_seed(seed)
    g = torch.Generator(device=RT.DEV).manual_seed(seed)
    n = RT.N_SPLIT[split]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 7))
    MU, LV = [], []
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(n, i + 1024))
        b = get_batch(split, idx, list(experts.keys()), g)
        masks, xs = apply_condition(b, cond or {}, split, idx, g, perm)
        mu, lv = expert_beliefs(experts, b, masks, xs)
        MU.append(mu); LV.append(lv)
    return torch.cat(MU), torch.cat(LV)


class Witness:
    def __init__(self, k=8, pi=0.05, kappa=10.0, shrink=0.05, far=0.01):
        self.k, self.pi, self.kappa, self.shrink, self.far = k, pi, kappa, shrink, far
        self.T = 1.0

    # ---------------------------------------------------------------- fitting (clean, unlabelled)
    def fit(self, auditor, mods):
        self.mods, self.M = list(mods), len(mods)
        tr = _beliefs(auditor, 'train')[0].cpu().numpy().astype(np.float64)
        self.P, self.c = [], []
        for m in range(self.M):
            X = tr[:, m] - tr[:, m].mean(0)
            _, _, Vt = np.linalg.svd(X[:20000], full_matrices=False)
            self.P.append(Vt[:self.k].T)
            self.c.append(tr[:, m].mean(0))
        va = self.summarise(_beliefs(auditor, 'valid')[0])
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(va))
        A, B = va[idx[:len(va) // 2]], va[idx[len(va) // 2:]]
        self.mean = A.mean(0)
        C = np.cov(A.T)
        self.C = (1 - self.shrink) * C + self.shrink * np.diag(np.diag(C))
        self._cache = {}
        full = tuple(range(self.M))
        best = None
        for T in np.exp(np.linspace(np.log(0.05), np.log(50), 60)):      # clean false-alarm rate -> ~far
            self.T = T
            r = self.posterior(B, full)[1]
            fa = np.mean(r > 0.5)
            if best is None or abs(fa - self.far) < best[0]:
                best = (abs(fa - self.far), T)
        self.T = best[1]
        # Lee/CARAT-style baseline: per-modality marginal Mahalanobis, threshold = 99th percentile on clean data
        self.marg_thr = np.quantile(self.marginal_scores(B), 0.99, axis=0)
        return self

    def marginal_scores(self, O):
        """Per-modality Mahalanobis^2 of each block under its own clean marginal (unimodal evidence only)."""
        O = O - self.mean
        out = []
        for m in range(self.M):
            Ci, _ = self._gauss(self._blk(m))
            x = O[:, self._blk(m)]
            out.append((x @ Ci * x).sum(1))
        return np.stack(out, 1)

    def summarise(self, mus):
        mus = mus.cpu().numpy().astype(np.float64) if torch.is_tensor(mus) else mus
        return np.concatenate([(mus[:, m] - self.c[m]) @ self.P[m] for m in range(self.M)], 1)

    def _blk(self, m):
        return np.arange(m * self.k, (m + 1) * self.k)

    def _gauss(self, idx):
        key = tuple(idx)
        if key not in self._cache:
            C = self.C[np.ix_(idx, idx)]
            L = np.linalg.cholesky(C)
            self._cache[key] = (np.linalg.inv(C), 2 * np.log(np.diag(L)).sum())
        return self._cache[key]

    def _loglik(self, O, idx, scale=1.0):
        Ci, ld = self._gauss(idx)
        x = O[:, idx]
        return -0.5 * ((x @ Ci * x).sum(1) / scale + ld + len(idx) * np.log(scale) + len(idx) * np.log(2 * np.pi))

    # ---------------------------------------------------------------- inference
    def posterior(self, O, present):
        """O: centred summaries (N, M*k); present: indices of available modalities.
        Returns (configs as tuples of reliable modality indices, p(R | o) (N, n_configs), r (N, M))."""
        O = O - self.mean
        present = list(present)
        configs, logp = [], []
        for bits in itertools.product([0, 1], repeat=len(present)):     # 1 = lying
            R = [m for m, c in zip(present, bits) if c == 0]
            lp = np.zeros(len(O))
            if R:
                lp += self._loglik(O, np.concatenate([self._blk(m) for m in R]))
            for m, c in zip(present, bits):
                if c:
                    lp += self._loglik(O, self._blk(m), self.kappa)
            lp = lp / self.T + sum(np.log(self.pi) if c else np.log(1 - self.pi) for c in bits)
            configs.append(tuple(R)); logp.append(lp)
        logp = np.stack(logp, 1)
        p = np.exp(logp - logp.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        r = np.zeros((len(O), self.M))
        for j, R in enumerate(configs):
            for m in present:
                if m not in R:
                    r[:, m] += p[:, j]
        return configs, p, r


# -------------------------------------------------------------------- worker predictions
@torch.no_grad()
def _subset_outputs(model, mus, lvs, configs):
    """Worker prediction for every reliable subset R: means (N, n_cfg), vars (N, n_cfg), shares (N, n_cfg, M)."""
    M = mus.shape[1]
    Ms, Vs, Ss = [], [], []
    for R in configs:
        av = torch.zeros(mus.shape[0], M, device=RT.DEV)
        if R:
            av[:, list(R)] = 1.0
        m, v, _, _, _, share = fused_output(model['fusion'], model['head'], mus, lvs, av)
        Ms.append(m); Vs.append(v); Ss.append(share)
    return (torch.stack(Ms, 1).cpu().numpy().astype(np.float64), torch.stack(Vs, 1).cpu().numpy().astype(np.float64),
            torch.stack(Ss, 1).cpu().numpy().astype(np.float64))


def _mixture(p, Ms, Vs):
    mean = (p * Ms).sum(1)
    var = (p * (Vs + Ms ** 2)).sum(1) - mean ** 2
    return mean, np.maximum(var, 1e-8)


def _mixture_halfwidth(p, Ms, Ss, mean, level, iters=40):
    """Half-width h of the central interval around `mean` holding `level` of the Gaussian mixture."""
    lo, hi = np.zeros(len(mean)), np.full(len(mean), 20.0)
    for _ in range(iters):
        h = (lo + hi) / 2
        cov = (p * (stats.norm.cdf((mean[:, None] + h[:, None] - Ms) / Ss)
                    - stats.norm.cdf((mean[:, None] - h[:, None] - Ms) / Ss))).sum(1)
        lo, hi = np.where(cov < level, h, lo), np.where(cov < level, hi, h)
    return (lo + hi) / 2


def _cond_modalities(cd, mods):
    if cd.get('noise'):
        return [mods.index(m) for m in cd['noise']]
    if cd.get('swap'):
        return [mods.index(m) for m in cd['swap']]
    return []


VARIANTS = {'cxf': 'soft posterior (BMA)', 'cxf_map': 'hard most-likely subset',
            'hard_drop': 'Lee/CARAT-style hard drop (marginal detector)'}


def _variant_weights(variant, witness, O, present, configs, p):
    """Turn evidence into subset weights for each variant; returns (weights over configs, per-modality score)."""
    M = witness.M
    if variant == 'cxf':
        r = np.zeros((len(O), M))
        for j, R in enumerate(configs):
            for m in present:
                if m not in R:
                    r[:, m] += p[:, j]
        return p, r
    if variant == 'cxf_map':
        w = np.zeros_like(p)
        w[np.arange(len(p)), p.argmax(1)] = 1.0
        r = np.zeros((len(O), M))
        for j, R in enumerate(configs):
            for m in present:
                if m not in R:
                    r[:, m] += p[:, j]
        return w, r
    z = witness.marginal_scores(O) / witness.marg_thr                 # > 1 means flagged
    w = np.zeros_like(p)
    for i in range(len(O)):
        flagged = [m for m in present if z[i, m] > 1]
        drop = [max(flagged, key=lambda m: z[i, m])] if flagged else []  # Lee: drop the most surprising one
        R = tuple(m for m in present if m not in drop)
        w[i, configs.index(R)] = 1.0
    return w, z


def evaluate_cxf(model, auditor, witness, conds, tag, variants=('cxf', 'cxf_map', 'hard_drop')):
    rows = []
    for v in variants:
        rows += _evaluate_variant(model, auditor, witness, conds, dict(tag), v)
    return rows


def _evaluate_variant(model, auditor, witness, conds, tag, variant):
    """Result rows (same columns as evaluate.evaluate_model) for fusion = 'cxf'.
    Decoders: D2 (raw mixture), D3b (pooled split conformal on CXF output), D3x (CXF-C: per-subset conformal
    calibrated on clean declared-missing subsets, mixed by the switch posterior)."""
    mods, L = model['mods'], levels()
    M = len(mods)

    def run(split, cd):
        mus, lvs = _beliefs(model['experts'], split, cd)
        amus = mus if auditor is model['experts'] else _beliefs(auditor, split, cd)[0]
        present = [i for i in range(M) if cd['avail'][i]]
        O = witness.summarise(amus)
        configs, p, _ = witness.posterior(O, present)
        p, r = _variant_weights(variant, witness, O, present, configs, p)
        Ms, Vs, Sh = _subset_outputs(model, mus, lvs, configs)
        mean, var = _mixture(p, Ms, Vs)
        share = (p[:, :, None] * Sh).sum(1)
        return dict(y=RT.DATA[split]['y'].numpy().astype(np.float64), m=mean, v=var, r=r, p=p, Ms=Ms, Vs=Vs,
                    configs=configs, share=share)

    # per-subset conformal quantiles from CLEAN validation with declared-missing subsets (worker only, no CXF)
    qR = {}
    mus_v, lvs_v = _beliefs(model['experts'], 'valid')
    yv = RT.DATA['valid']['y'].numpy().astype(np.float64)
    for code in all_subsets(M):
        R = tuple(i for i in range(M) if code[i])
        Ms, Vs, _ = _subset_outputs(model, mus_v, lvs_v, [R])
        z = np.abs(yv - Ms[:, 0]) / np.sqrt(Vs[:, 0])
        qR[R] = {pl: conf_q(z, pl) for pl in L}
    qR[()] = {pl: z_lev(pl) for pl in L}

    vals = {cd['name']: run('valid', cd) for cd in conds}
    yy = np.concatenate([v['y'] for v in vals.values()])
    zz = np.concatenate([np.abs(v['y'] - v['m']) / np.sqrt(v['v']) for v in vals.values()])
    qn = {pl: conf_q(zz, pl) for pl in L}
    tests = {cd['name']: run('test', cd) for cd in conds}
    clean = tests.get('TAV')
    rows = []
    for cd in conds:
        te = tests[cd['name']]
        y, m, sd = te['y'], te['m'], np.sqrt(te['v'])
        hx = {}
        for pl in L:
            Ss = np.stack([np.sqrt(te['Vs'][:, j]) * qR[R][pl] / z_lev(pl) for j, R in enumerate(te['configs'])], 1)
            hx[pl] = _mixture_halfwidth(te['p'], te['Ms'], Ss, m, pl)
        base = dict(tag, fusion=variant, unc='base', cond=cd['name'], kind=cd['kind'], lam=cd.get('lam', 0.0),
                    corrupted=cd.get('corrupted', ''), text_drop=cd.get('text_drop', 0.0), cxf_T=witness.T)
        for k, mm in enumerate(mods):
            base[f'share_{mm}'] = te['share'][:, k].mean()
            base[f'r_{mm}'] = te['r'][:, k].mean()
        base['false_alarm'] = np.mean(te['r'].max(1) > (1.0 if variant == 'hard_drop' else 0.5))
        decs = {'D2': (sd, {pl: z_lev(pl) * sd for pl in L}), 'D3b': (None, {pl: qn[pl] * sd for pl in L}),
                'D3x': (None, hx)}
        per = {}
        for dn, (sdd, hw) in decs.items():
            row = dict(base, decoder=dn, **point_metrics(y, m), **interval_metrics(y, m, hw))
            if sdd is not None:
                row.update(prob_metrics(y, m, sdd))
            per[dn] = row
        cm = _cond_modalities(cd, mods)
        if cm and clean is not None:
            lab = np.r_[np.zeros(len(y)), np.ones(len(y))]
            per['D2']['det_AUROC_cxf'] = np.mean([roc_auc_score(lab, np.r_[clean['r'][:, j], te['r'][:, j]])
                                                  for j in cm])
            per['D2']['det_AUROC_cxf_any'] = roc_auc_score(lab, np.r_[clean['r'].max(1), te['r'].max(1)])
            if len(cm) == 1:
                per['D2']['localise_acc'] = np.mean(np.argmax(te['r'], 1) == cm[0])
        rows += list(per.values())
    return rows
