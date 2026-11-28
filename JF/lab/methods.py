"""Uncertainty-tracking methods. Each method: fit(C) on train/valid, then predict(C, split, cond) -> (mu, var).

Rule for fairness: every method may use the self-generated corruptions (white noise lambda=1, random swap) present in
the train/valid splits; nothing else. Held-out test corruptions are never seen.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from .cache import MODS, PATS

Z90 = norm.ppf(0.95)
SELF = ('clean',) + tuple(f'noise[{m}]' for m in MODS) + tuple(f'swap[{m}]' for m in MODS)
LAB = {'clean': 0, **{f'noise[{m}]': j + 1 for j, m in enumerate(MODS)}, **{f'swap[{m}]': j + 1 for j, m in enumerate(MODS)}}
torch.set_num_threads(2)


def stack(C, split, conds, what):
    return np.concatenate([C[what][split][c] for c in conds])


def ys(C, split, conds):
    return np.concatenate([C['y'][split] for _ in conds])


def nll(y, mu, v):
    return np.mean(0.5 * np.log(2 * np.pi * v) + 0.5 * (y - mu) ** 2 / v)


def fit_temp(y, mu, v):
    return float(np.mean((y - mu) ** 2 / v))


def fit_q(y, mu, v, p=0.90):
    """Conformal scale: multiply sd so that the 90% Gaussian interval covers p of calibration clips."""
    s = np.abs(y - mu) / np.sqrt(v)
    n = len(s)
    q = np.sort(s)[min(n - 1, int(np.ceil((n + 1) * p)) - 1)]
    return float((q / Z90) ** 2)


class MLP(nn.Module):
    def __init__(self, d, out, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, out))

    def forward(self, x):
        return self.f(x)


# ----------------------------------------------------------------------------- detectors (shared, fitted once)
class Detectors:
    """ELECTRA-style replaced-modality classifier on pooled stats (classes: none, t, a, v) and an unsupervised
    per-modality Mahalanobis density score. Both fitted on train only."""

    def fit(self, C, seed=0, epochs=60):
        F = stack(C, 'train', SELF, 'F')
        L = np.concatenate([np.full(len(C['y']['train']), LAB[c]) for c in SELF])
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        torch.manual_seed(seed)
        self.net = MLP(F.shape[1], 4)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        X = torch.from_numpy(((F - self.mu) / self.sd).astype(np.float32)); Y = torch.from_numpy(L).long()
        g = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            for idx in torch.randperm(len(X), generator=g).split(512):
                loss = nn.functional.cross_entropy(self.net(X[idx]), Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        # temperature on valid
        Fv = stack(C, 'valid', SELF, 'F'); Lv = np.concatenate([np.full(len(C['y']['valid']), LAB[c]) for c in SELF])
        with torch.no_grad():
            lg = self.net(torch.from_numpy(((Fv - self.mu) / self.sd).astype(np.float32)))
        best = (1e9, 1.0)
        for T in np.linspace(0.5, 5, 46):
            l = nn.functional.cross_entropy(lg / T, torch.from_numpy(Lv).long()).item()
            best = min(best, (l, T))
        self.T = best[1]
        # density on clean train, per modality slice
        Fc = C['F']['train']['clean']
        self.slices, o = [], 0
        for d in C['fdims']:
            self.slices.append(slice(o, o + d)); o += d
        self.g = []
        for sl in self.slices:
            Z = Fc[:, sl]; mu = Z.mean(0); Cv = np.cov(Z, rowvar=False)
            Cv = 0.9 * Cv + 0.1 * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            self.g.append((mu, np.linalg.inv(Cv), Z.shape[1]))
        return self

    def probs(self, F):
        with torch.no_grad():
            lg = self.net(torch.from_numpy(((F - self.mu) / self.sd).astype(np.float32)))
        return torch.softmax(lg / self.T, 1).numpy()

    def density(self, F):
        out = []
        for sl, (mu, P, d) in zip(self.slices, self.g):
            D = F[:, sl] - mu
            out.append(np.einsum('ij,jk,ik->i', D, P, D) / d)
        return np.stack(out, 1)


# ----------------------------------------------------------------------------- methods
class Method:
    name = 'base'
    uses_corruptions = False

    def __init__(self, det=None, **kw):
        self.det = det
        self.kw = kw

    def raw(self, C, split, cond):
        return C['MU'][split][cond][:, 0], C['VAR'][split][cond][:, 0]

    def fit(self, C):
        conds = SELF if self.uses_corruptions else ('clean',)
        mu = np.concatenate([self.raw(C, 'valid', c)[0] for c in conds])
        v = np.concatenate([self.raw(C, 'valid', c)[1] for c in conds])
        self.s = self.scale(ys(C, 'valid', conds), mu, v)
        return self

    def scale(self, y, mu, v):
        return fit_temp(y, mu, v)

    def predict(self, C, split, cond):
        mu, v = self.raw(C, split, cond)
        return mu, v * self.s


class PoE(Method):
    name = 'poe (temp, clean cal)'


class ConfClean(Method):
    name = 'conformal (clean cal)'

    def scale(self, y, mu, v):
        return fit_q(y, mu, v)


class ConfMix(ConfClean):
    name = 'conformal (self-gen cal)'
    uses_corruptions = True


class GateDrop(Method):
    """Drop the modality the detector blames (if P(clean) < 0.5)."""
    name = 'gate-drop'
    uses_corruptions = True

    def raw(self, C, split, cond):
        p = self.det.probs(C['F'][split][cond])
        j = p.argmax(1)
        MU, V = C['MU'][split][cond], C['VAR'][split][cond]
        idx = np.where(j == 0, 0, j)            # pattern 1..3 = drop t, a, v
        r = np.arange(len(j))
        return MU[r, idx], V[r, idx]


class PatternBMA(Method):
    """Bayesian model averaging over 'which modality is broken': mixture of the full and drop-one patterns,
    weighted by the detector's calibrated posterior. Spread between patterns enters the variance."""
    name = 'pattern-BMA (detector posterior)'
    uses_corruptions = True

    def raw(self, C, split, cond):
        p = self.det.probs(C['F'][split][cond])
        MU, V = C['MU'][split][cond][:, :4], C['VAR'][split][cond][:, :4]
        mu = (p * MU).sum(1)
        return mu, np.maximum((p * (V + MU ** 2)).sum(1) - mu ** 2, 1e-9)


class Jackknife(Method):
    """Leave-one-modality-out jackknife: var += alpha * sum_m (mu_full - mu_drop_m)^2 (no detector, no training)."""
    name = 'modality jackknife (clean cal)'

    def fit(self, C):
        conds = SELF if self.uses_corruptions else ('clean',)
        y = ys(C, 'valid', conds)
        MU = stack(C, 'valid', conds, 'MU'); V = stack(C, 'valid', conds, 'VAR')
        J = ((MU[:, :1] - MU[:, 1:4]) ** 2).sum(1)
        best = (1e9, 0, 1)
        for a in [0, .1, .25, .5, 1, 2, 4, 8]:
            v = V[:, 0] + a * J
            t = fit_temp(y, MU[:, 0], v)
            best = min(best, (nll(y, MU[:, 0], v * t), a, t))
        _, self.a, self.s = best
        return self

    def predict(self, C, split, cond):
        MU, V = C['MU'][split][cond], C['VAR'][split][cond]
        J = ((MU[:, :1] - MU[:, 1:4]) ** 2).sum(1)
        return MU[:, 0], (V[:, 0] + self.a * J) * self.s


class JackknifeMix(Jackknife):
    name = 'modality jackknife (self-gen cal)'
    uses_corruptions = True


def features(C, split, cond, det, use_det=True, feat='all'):
    MU, V = C['MU'][split][cond], C['VAR'][split][cond]
    if feat == 'det':
        F = C['F'][split][cond]
        return np.concatenate([np.log(V[:, :1]), MU[:, :1], np.log(det.probs(F) + 1e-6),
                               np.log1p(det.density(F))], 1).astype(np.float32)
    cols = [np.log(V), MU[:, :1] - MU[:, 1:], MU[:, :1]]
    if use_det:
        F = C['F'][split][cond]
        cols += [np.log(det.probs(F) + 1e-6), np.log1p(det.density(F))]
    return np.concatenate(cols, 1).astype(np.float32)


class MetaSigma(Method):
    """Second-opinion variance head: Gaussian NLL on the residual of a base mean, from features that can see
    corruption (all pattern variances, leave-one-out shifts, detector log-probs, densities). Trained on train
    with self-generated corruptions; recalibrated on valid."""
    name = 'meta-sigma'
    uses_corruptions = True

    def __init__(self, det=None, base='full', use_det=True, predict_mean=False, epochs=40, train_conds=SELF,
                 feat='all', n_ens=1, **kw):
        super().__init__(det, **kw)
        self.base, self.use_det, self.pm, self.epochs = base, use_det, predict_mean, epochs
        self.tc, self.feat, self.n_ens = train_conds, feat, n_ens

    def base_mu(self, C, split, cond):
        if self.base == 'bma':
            return PatternBMA(self.det).raw(C, split, cond)[0]
        if self.base == 'gate':
            return GateDrop(self.det).raw(C, split, cond)[0]
        return C['MU'][split][cond][:, 0]

    def fit(self, C):
        Xs = np.concatenate([features(C, 'train', c, self.det, self.use_det, self.feat) for c in self.tc])
        mu0 = np.concatenate([self.base_mu(C, 'train', c) for c in self.tc])
        y = ys(C, 'train', self.tc)
        self.m, self.sd = Xs.mean(0), Xs.std(0) + 1e-6
        X = torch.from_numpy((Xs - self.m) / self.sd); R = torch.from_numpy((y - mu0).astype(np.float32))
        self.nets = []
        for k in range(self.n_ens):
            torch.manual_seed(k)
            net = MLP(Xs.shape[1], 2, h=64)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            g = torch.Generator().manual_seed(k)
            for ep in range(self.epochs):
                for idx in torch.randperm(len(X), generator=g).split(512):
                    o = net(X[idx])
                    d = o[:, 0] if self.pm else 0
                    lv = o[:, 1].clamp(-6, 4)
                    loss = 0.5 * (lv + (R[idx] - d) ** 2 / lv.exp()).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            self.nets.append(net.eval())
        mu = np.concatenate([self._raw(C, 'valid', c)[0] for c in SELF])
        v = np.concatenate([self._raw(C, 'valid', c)[1] for c in SELF])
        self.s = fit_temp(ys(C, 'valid', SELF), mu, v)
        return self

    def _raw(self, C, split, cond):
        X = torch.from_numpy((features(C, split, cond, self.det, self.use_det, self.feat) - self.m) / self.sd)
        with torch.no_grad():
            O = np.stack([n(X).numpy() for n in self.nets])
        base = self.base_mu(C, split, cond)
        mus = base[None] + (O[:, :, 0] if self.pm else 0)
        vs = np.exp(np.clip(O[:, :, 1], -6, 4))
        mu = mus.mean(0)
        return mu, vs.mean(0) + mus.var(0)

    def predict(self, C, split, cond):
        mu, v = self._raw(C, split, cond)
        return mu, v * self.s


class MondrianDet(Method):
    """Conformal scale per detector bin (P(corrupted) < .2, .2-.8, > .8), calibrated on valid with self-gen data,
    on top of a base method."""
    name = 'mondrian-by-detector'
    uses_corruptions = True

    def __init__(self, det=None, inner=None, **kw):
        super().__init__(det, **kw)
        self.inner = inner

    def _bins(self, C, split, cond):
        p = 1 - self.det.probs(C['F'][split][cond])[:, 0]
        return np.digitize(p, [0.2, 0.8])

    def fit(self, C):
        self.inner.fit(C)
        y = ys(C, 'valid', SELF)
        mu = np.concatenate([self.inner.predict(C, 'valid', c)[0] for c in SELF])
        v = np.concatenate([self.inner.predict(C, 'valid', c)[1] for c in SELF])
        b = np.concatenate([self._bins(C, 'valid', c) for c in SELF])
        self.q = {k: fit_q(y[b == k], mu[b == k], v[b == k]) if (b == k).sum() > 30 else fit_q(y, mu, v) for k in range(3)}
        return self

    def predict(self, C, split, cond):
        mu, v = self.inner.predict(C, split, cond)
        b = self._bins(C, split, cond)
        return mu, v * np.vectorize(self.q.get)(b)


class OracleDrop(Method):
    """Reference: knows which modality is corrupted and drops it (cannot fix two bad modalities)."""
    name = 'oracle drop (reference)'

    def raw(self, C, split, cond):
        MU, V = C['MU'][split][cond], C['VAR'][split][cond]
        m = C['meta'].get(cond, {}).get('m', '') if split == 'test' else (cond.split('[')[1][0] if '[' in cond else '')
        if len(m) == 1:
            j = 1 + MODS.index(m)
            return MU[:, j], V[:, j]
        return MU[:, 0], V[:, 0]


class CQR(Method):
    """Conformalized quantile regression (Romano et al. 2019) on the meta features, trained with self-generated
    corruptions: two quantile heads (5%, 95%) plus a median, conformalised on valid. Gaussian sd = half-width / z."""
    name = 'CQR'
    uses_corruptions = True

    def fit(self, C, epochs=40):
        Xs = np.concatenate([features(C, 'train', c, self.det) for c in SELF])
        y = ys(C, 'train', SELF)
        self.m, self.sd = Xs.mean(0), Xs.std(0) + 1e-6
        torch.manual_seed(0)
        self.net = MLP(Xs.shape[1], 3, h=64)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        X = torch.from_numpy((Xs - self.m) / self.sd); Y = torch.from_numpy(y.astype(np.float32))
        qs = torch.tensor([0.05, 0.5, 0.95])
        g = torch.Generator().manual_seed(0)
        for ep in range(epochs):
            for idx in torch.randperm(len(X), generator=g).split(512):
                e = Y[idx, None] - self.net(X[idx])
                loss = torch.maximum(qs * e, (qs - 1) * e).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        lo, md, hi = self._q(C, 'valid')
        yv = ys(C, 'valid', SELF)
        sc = np.maximum(lo - yv, yv - hi)
        n = len(sc)
        self.c = np.sort(sc)[min(n - 1, int(np.ceil((n + 1) * 0.9)) - 1)]
        return self

    def _q(self, C, split, cond=None):
        conds = SELF if cond is None else (cond,)
        X = np.concatenate([features(C, split, c, self.det) for c in conds])
        with torch.no_grad():
            o = self.net(torch.from_numpy((X - self.m) / self.sd)).numpy()
        o = np.sort(o, 1)
        return o[:, 0], o[:, 1], o[:, 2]

    def predict(self, C, split, cond):
        lo, md, hi = self._q(C, split, cond)
        hw = np.maximum((hi - lo) / 2 + self.c, 1e-3)
        mu = (lo + hi) / 2
        return mu, (hw / Z90) ** 2
