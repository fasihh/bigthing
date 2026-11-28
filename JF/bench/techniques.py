"""Fusion and detection techniques for the noisy-TV screen.

Uniform interface. A technique gets, per clip set:
    P    : {m: (mu_m, var_m)}  expert beliefs for the AVAILABLE modalities (missing ones absent)
    X    : {m: raw clips}      raw inputs of the available modalities (detectors may use them)
    ctx  : Context             fitted helpers (all fitted on CLEAN train / calibration data)
and returns (mu, var, susp) where susp is an (n, M) array of per-modality suspicion
(np.nan for missing modalities) or None.

Detectors are turned into fusion rules in one uniform way (gate()): a modality is dropped when
its suspicion exceeds the 99th percentile of that detector on clean calibration clips, at most
one modality per clip, and never the last one. Soft variants inflate the variance instead.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from .experts import pool

EPS = 1e-9


# ----------------------------------------------------------------------------- fusion primitives
def poe(P, prior=None):
    """Product of experts. With prior=(m0, v0) the M-1 extra copies of the prior are divided out."""
    ms = list(P)
    lam = sum(1.0 / P[m][1] for m in ms)
    num = sum(P[m][0] / P[m][1] for m in ms)
    if prior is not None and len(ms) > 1:
        m0, v0 = prior
        k = len(ms) - 1
        lam = lam - k / v0
        num = num - k * m0 / v0
        lam = np.maximum(lam, 0.05 / v0)
    return num / lam, 1.0 / lam


def poe_w(P, w, prior=None):
    """PoE with per-expert precision weights w[m] (arrays in [0, 1])."""
    ms = list(P)
    lam = sum(w[m] / P[m][1] for m in ms)
    num = sum(w[m] * P[m][0] / P[m][1] for m in ms)
    if prior is not None:
        m0, v0 = prior
        k = sum(w[m] for m in ms) - 1
        k = np.maximum(k, 0)
        lam = lam - k / v0
        num = num - k * m0 / v0
        lam = np.maximum(lam, 0.05 / v0)
    return num / lam, 1.0 / lam


def mixture(P, w=None):
    ms = list(P)
    if w is None:
        w = {m: np.full_like(P[m][0], 1.0 / len(ms)) for m in ms}
    mu = sum(w[m] * P[m][0] for m in ms)
    second = sum(w[m] * (P[m][1] + P[m][0] ** 2) for m in ms)
    return mu, np.maximum(second - mu ** 2, EPS)


def stack_susp(P, ms_all, d):
    n = len(next(iter(P.values()))[0])
    out = np.full((n, len(ms_all)), np.nan)
    for j, m in enumerate(ms_all):
        if m in d:
            out[:, j] = d[m]
    return out


# ----------------------------------------------------------------------------- baselines
def t_poe(P, X, ctx):
    mu, v = poe(P)
    return mu, v, None


def t_poe_pc(P, X, ctx):
    mu, v = poe(P, ctx.prior)
    return mu, v, None


def t_mean(P, X, ctx):
    mu, v = mixture(P)
    return mu, v, None


class LearnedGate:
    """Attention-style baseline: MLP on (mu, log var, present) of every expert -> mixture weights.
    Trained on clean calibration clips with random modality dropout (so it can handle subsets)."""

    def fit(self, ctx):
        ms = ctx.mods
        P, y = ctx.cal_P, ctx.y_cal
        rng = np.random.default_rng(ctx.seed)
        feats, masks = self._feats(P, ms), None
        torch.manual_seed(ctx.seed)
        net = nn.Sequential(nn.Linear(feats.shape[1], 32), nn.GELU(), nn.Linear(32, len(ms) + 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        F = torch.from_numpy(feats).float(); Y = torch.from_numpy(y).float()
        MU = torch.from_numpy(np.stack([P[m][0] for m in ms], 1)).float()
        V = torch.from_numpy(np.stack([P[m][1] for m in ms], 1)).float()
        for ep in range(400):
            keep = torch.from_numpy(rng.random((len(y), len(ms))) > 0.25)
            keep[keep.sum(1) == 0, 0] = True
            Fm = F.clone()
            for j in range(len(ms)):
                Fm[~keep[:, j], 3 * j:3 * j + 3] = 0
            out = net(Fm)
            logits = out[:, :-1].masked_fill(~keep, -1e9)
            w = torch.softmax(logits, 1)
            mu = (w * MU).sum(1)
            var = (w * (V + MU ** 2)).sum(1) - mu ** 2 + 1e-6
            var = var * torch.exp(out[:, -1])
            loss = 0.5 * (var.log() + (Y - mu) ** 2 / var).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        self.net = net.eval()
        return self

    @staticmethod
    def _feats(P, ms):
        n = len(next(iter(P.values()))[0])
        cols = []
        for m in ms:
            if m in P:
                cols += [P[m][0], np.log(P[m][1]), np.ones(n)]
            else:
                cols += [np.zeros(n)] * 3
        return np.stack(cols, 1).astype(np.float32)

    def __call__(self, P, X, ctx):
        ms = ctx.mods
        with torch.no_grad():
            out = self.net(torch.from_numpy(self._feats(P, ms)))
            keep = torch.tensor([[m in P for m in ms]]).expand(out.shape[0], -1)
            w = torch.softmax(out[:, :-1].masked_fill(~keep, -1e9), 1).numpy()
            scale = torch.exp(out[:, -1]).numpy()
        W = {m: w[:, j] for j, m in enumerate(ms) if m in P}
        mu, v = mixture(P, W)
        susp = stack_susp(P, ms, {m: 1 - W[m] * len(P) for m in P})
        return mu, v * scale, susp


# ----------------------------------------------------------------------------- robust statistics
def t_median(P, X, ctx):
    ms = list(P)
    if len(ms) < 3:
        return (*poe(P, ctx.prior), None)
    M = np.stack([P[m][0] for m in ms], 1)
    V = np.stack([P[m][1] for m in ms], 1)
    idx = np.argsort(M, 1)[:, len(ms) // 2]
    r = np.arange(len(M))
    med = M[r, idx]
    dev = np.abs(M - med[:, None]) / np.sqrt(V)
    return med, V[r, idx], stack_susp(P, ctx.mods, {m: dev[:, j] for j, m in enumerate(ms)})


def _irls(P, ctx, weight_fn, iters=15):
    ms = list(P)
    mu, v = poe(P, ctx.prior)
    w = {m: np.ones_like(mu) for m in ms}
    for _ in range(iters):
        for m in ms:
            r = (P[m][0] - mu) / np.sqrt(P[m][1] + v)
            w[m] = weight_fn(r)
        mu, v = poe_w(P, w, ctx.prior)
    return mu, v, stack_susp(P, ctx.mods, {m: 1 - w[m] for m in ms})


def t_huber(P, X, ctx, c=1.345):
    if len(P) < 2:
        return (*poe(P, ctx.prior), None)
    return _irls(P, ctx, lambda r: np.minimum(1.0, c / np.maximum(np.abs(r), EPS)))


def t_student(P, X, ctx, nu=3.0):
    if len(P) < 2:
        return (*poe(P, ctx.prior), None)
    return _irls(P, ctx, lambda r: (nu + 1) / (nu + r ** 2))


def t_trimmed(P, X, ctx):
    ms = list(P)
    if len(ms) < 3:
        return (*poe(P, ctx.prior), None)
    M = np.stack([P[m][0] for m in ms], 1); V = np.stack([P[m][1] for m in ms], 1)
    med = np.median(M, 1, keepdims=True)
    dev = np.abs(M - med) / np.sqrt(V)
    worst = dev.argmax(1)
    mu = np.zeros(len(M)); v = np.zeros(len(M))
    for j, m in enumerate(ms):
        sel = worst == j
        if sel.any():
            Q = {k: (P[k][0][sel], P[k][1][sel]) for k in ms if k != m}
            mu[sel], v[sel] = poe(Q, ctx.prior)
    return mu, v, stack_susp(P, ctx.mods, {m: dev[:, j] for j, m in enumerate(ms)})


# ----------------------------------------------------------------------------- sensor FDI / Bayesian
def nis(P, ctx):
    """Leave-one-out normalised innovation squared per modality."""
    ms = list(P)
    out = {}
    for m in ms:
        Q = {k: P[k] for k in ms if k != m}
        mo, vo = poe(Q, ctx.prior)
        out[m] = (P[m][0] - mo) ** 2 / (P[m][1] + vo)
    return out


def raim_stat(P, ctx):
    mu, _ = poe(P)
    return sum((P[m][0] - mu) ** 2 / P[m][1] for m in P)


def t_ci(P, X, ctx):
    """Covariance intersection in 1-D: the optimal omega puts all weight on the most precise expert."""
    ms = list(P)
    V = np.stack([P[m][1] for m in ms], 1); M = np.stack([P[m][0] for m in ms], 1)
    j = V.argmin(1); r = np.arange(len(V))
    return M[r, j], V[r, j], None


def t_contam(P, X, ctx, beta=0.05, infl=None):
    """Good-and-bad (Box & Tiao) contamination model: exact posterior over which experts are bad.
    Good experts: mu_m ~ N(y, var_m); bad experts: mu_m ~ N(m0, v0 + var_m) (uninformative about y).
    Predictive = mixture over states of PoE(good), moment matched."""
    ms = list(P)
    m0, v0 = ctx.prior
    n = len(P[ms[0]][0])
    logw, mus, vs, states = [], [], [], []
    for b in itertools.product((0, 1), repeat=len(ms)):
        if all(b) and len(ms) > 0:
            pass
        good = [m for m, bb in zip(ms, b) if bb == 0]
        bad = [m for m, bb in zip(ms, b) if bb == 1]
        lp = np.full(n, np.log(beta) * len(bad) + np.log(1 - beta) * len(good))
        for m in bad:
            vb = v0 + P[m][1] if infl is None else infl * P[m][1]
            lp += norm.logpdf(P[m][0], m0, np.sqrt(vb))
        if good:
            Mg = np.stack([P[m][0] for m in good], 1)
            Vg = np.stack([P[m][1] for m in good], 1)
            k = len(good)
            # joint of good means: N(m0 1, v0 11^T + diag(Vg)) -- closed form via matrix determinant lemma
            d = Mg - m0
            s1 = (d / Vg).sum(1); s0 = (1 / Vg).sum(1)
            quad = (d ** 2 / Vg).sum(1) - v0 * s1 ** 2 / (1 + v0 * s0)
            logdet = np.log(Vg).sum(1) + np.log1p(v0 * s0)
            lp += -0.5 * (quad + logdet + k * np.log(2 * np.pi))
            mu_s, v_s = poe({m: P[m] for m in good}, ctx.prior)
        else:
            mu_s, v_s = np.full(n, m0), np.full(n, v0)
        logw.append(lp); mus.append(mu_s); vs.append(v_s); states.append(b)
    L = np.stack(logw, 1); L -= L.max(1, keepdims=True)
    W = np.exp(L); W /= W.sum(1, keepdims=True)
    MU = np.stack(mus, 1); VV = np.stack(vs, 1)
    mu = (W * MU).sum(1)
    var = (W * (VV + MU ** 2)).sum(1) - mu ** 2
    S = np.array(states)
    pbad = W @ S                                   # marginal P(bad) per modality
    return mu, np.maximum(var, EPS), stack_susp(P, ctx.mods, {m: pbad[:, j] for j, m in enumerate(ms)})


# ----------------------------------------------------------------------------- detectors
class Detector:
    """Base: per-modality suspicion s_m(X, P) fitted on clean data; thresholds = clean-cal 99th pct."""
    name = 'det'

    def fit(self, ctx):
        return self

    def score(self, P, X, ctx):
        raise NotImplementedError

    def calibrate(self, ctx, q=0.99):
        s = self.score(ctx.cal_P, ctx.X_cal, ctx)
        self.thr = {m: np.quantile(s[m], q) for m in s}
        self.sd = {m: np.std(s[m]) + EPS for m in s}
        return self


class NISDet(Detector):
    def score(self, P, X, ctx):
        if len(P) < 3:
            return {m: np.zeros(len(P[m][0])) for m in P}
        return nis(P, ctx)


class RAIMDet(Detector):
    """Global residual test; per-modality score = drop in residual when that modality is excluded."""

    def score(self, P, X, ctx):
        if len(P) < 3:
            return {m: np.zeros(len(P[m][0])) for m in P}
        J = raim_stat(P, ctx)
        out = {}
        for m in P:
            Q = {k: P[k] for k in P if k != m}
            out[m] = J - raim_stat(Q, ctx)
        return out


class GaussDensity:
    def __init__(self, Z, shrink=0.1):
        self.mu = Z.mean(0)
        C = np.cov(Z, rowvar=False)
        C = (1 - shrink) * C + shrink * np.eye(len(C)) * np.trace(C) / len(C)
        self.P = np.linalg.inv(C)

    def maha(self, Z):
        D = Z - self.mu
        return np.einsum('ij,jk,ik->i', D, self.P, D)


class DDUDet(Detector):
    """Feature-space density of each expert's hidden layer (Mukhoti et al.)."""

    def __init__(self, expert_kind='plain'):
        self.kind = expert_kind

    def fit(self, ctx):
        self.g = {m: GaussDensity(ctx.experts[self.kind][m].predict(ctx.X_train[m])[2]) for m in ctx.mods}
        return self

    def score(self, P, X, ctx):
        return {m: self.g[m].maha(ctx.experts[self.kind][m].predict(X[m])[2]) for m in P}


class InputMahaDet(Detector):
    def fit(self, ctx):
        self.g = {m: GaussDensity(pool(ctx.X_train[m])) for m in ctx.mods}
        return self

    def score(self, P, X, ctx):
        return {m: self.g[m].maha(pool(X[m])) for m in P}


class KNNDet(Detector):
    def fit(self, ctx, k=10):
        self.k = k
        self.ref = {}
        for m in ctx.mods:
            F = pool(ctx.X_train[m]); mu, sd = F.mean(0), F.std(0) + 1e-6
            self.ref[m] = (torch.from_numpy((F - mu) / sd), mu, sd)
        return self

    def score(self, P, X, ctx):
        out = {}
        for m in P:
            R, mu, sd = self.ref[m]
            Q = torch.from_numpy((pool(X[m]) - mu) / sd)
            d = torch.cdist(Q, R)
            out[m] = d.topk(self.k, largest=False).values[:, -1].numpy()
        return out


def lag1(x):
    a, b = x[:, :-1], x[:, 1:]
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return ((a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-9)).mean(1)


class WhitenessDet(Detector):
    """Speech-VAD idea: real feature streams are autocorrelated, white noise is spectrally flat."""

    def fit(self, ctx):
        self.ref = {m: (lag1(ctx.X_train[m]).mean(), lag1(ctx.X_train[m]).std() + 1e-6) for m in ctx.mods}
        return self

    def score(self, P, X, ctx):
        return {m: np.abs(lag1(X[m]) - self.ref[m][0]) / self.ref[m][1] for m in P}


class RNDDet(Detector):
    """Random network distillation (Burda et al. 2019) per modality."""

    def fit(self, ctx):
        self.nets = {}
        for m in ctx.mods:
            F = pool(ctx.X_train[m]); mu, sd = F.mean(0), F.std(0) + 1e-6
            torch.manual_seed(ctx.seed + 7)
            tgt = nn.Sequential(nn.Linear(F.shape[1], 64), nn.ReLU(), nn.Linear(64, 32)).eval()
            prd = nn.Sequential(nn.Linear(F.shape[1], 64), nn.ReLU(), nn.Linear(64, 32))
            opt = torch.optim.Adam(prd.parameters(), lr=3e-3)
            X = torch.from_numpy((F - mu) / sd)
            with torch.no_grad():
                Tg = tgt(X)
            for _ in range(500):
                loss = ((prd(X) - Tg) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.nets[m] = (tgt, prd.eval(), mu, sd)
        return self

    def score(self, P, X, ctx):
        out = {}
        for m in P:
            tgt, prd, mu, sd = self.nets[m]
            Z = torch.from_numpy((pool(X[m]) - mu) / sd)
            with torch.no_grad():
                out[m] = ((prd(Z) - tgt(Z)) ** 2).mean(1).numpy()
        return out


class CrossModalDet(Detector):
    """AV-sync / CXF-lite: ridge-predict each modality's pooled stats from the other available ones."""

    def fit(self, ctx, alpha=10.0):
        self.alpha = alpha
        F = {m: pool(ctx.X_train[m]) for m in ctx.mods}
        self.norm = {m: (F[m].mean(0), F[m].std(0) + 1e-6) for m in ctx.mods}
        Fz = {m: (F[m] - self.norm[m][0]) / self.norm[m][1] for m in ctx.mods}
        self.models = {}
        for m in ctx.mods:
            others = [k for k in ctx.mods if k != m]
            for r in range(1, len(others) + 1):
                for sub in itertools.combinations(others, r):
                    A = np.concatenate([Fz[k] for k in sub], 1)
                    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Fz[m])
                    res = Fz[m] - A @ W
                    self.models[(m, sub)] = (W, res.var(0) + 1e-3)
        return self

    def score(self, P, X, ctx):
        Fz = {m: (pool(X[m]) - self.norm[m][0]) / self.norm[m][1] for m in P}
        out = {}
        for m in P:
            sub = tuple(k for k in ctx.mods if k in P and k != m)
            if not sub:
                out[m] = np.zeros(len(Fz[m])); continue
            W, rv = self.models[(m, sub)]
            A = np.concatenate([Fz[k] for k in sub], 1)
            out[m] = (((Fz[m] - A @ W) ** 2) / rv).mean(1)
        # a corrupted input also spoils the others' predictions; subtract the mean of the others so
        # the culprit stands out (residual of m minus average residual of the rest)
        if len(out) >= 3:
            tot = sum(out.values())
            out = {m: out[m] - (tot - out[m]) / (len(out) - 1) for m in out}
        return out


class PredConsistDet(Detector):
    """Joint density of the experts' predictions (Lee 2020 / CARAT style, on outputs not inputs):
    z-score of mu_m given the other experts' mus, from a Gaussian fitted on clean cal clips."""

    def fit(self, ctx):
        ms = ctx.mods
        M = np.stack([ctx.cal_P[m][0] for m in ms], 1)
        self.mu, self.C = M.mean(0), np.cov(M, rowvar=False)
        return self

    def score(self, P, X, ctx):
        ms = ctx.mods
        avail = [m for m in ms if m in P]
        out = {}
        for m in avail:
            others = [k for k in avail if k != m]
            if not others:
                out[m] = np.zeros(len(P[m][0])); continue
            i = ms.index(m); J = [ms.index(k) for k in others]
            Sjj = self.C[np.ix_(J, J)]; Sij = self.C[i, J]
            beta = np.linalg.solve(Sjj, Sij)
            cv = self.C[i, i] - Sij @ beta
            O = np.stack([P[k][0] for k in others], 1) - self.mu[J]
            pred = self.mu[i] + O @ beta
            out[m] = (P[m][0] - pred) ** 2 / cv
        return out


class ElectraDet(Detector):
    """ELECTRA-style replaced-modality detection [self-sup]: a classifier over all pooled inputs
    predicts which modality (if any) was replaced. Training corruptions: white noise and swaps only."""

    def __init__(self, kinds=('white', 'swap'), hard=False, k_nn=10):
        self.kinds, self.hard, self.k_nn = kinds, hard, k_nn

    def _neighbours(self, ctx, F):
        """For each modality m: the k nearest training clips in the OTHER modalities' pooled stats
        (hard negatives: plausible partners from a similar-looking clip; no speaker labels used)."""
        ms = ctx.mods
        nb = {}
        for m in ms:
            A = np.concatenate([(F[k] - F[k].mean(0)) / (F[k].std(0) + 1e-6) for k in ms if k != m], 1)
            A = torch.from_numpy(A.astype(np.float32))
            idx = []
            for i in range(0, len(A), 1024):
                d = torch.cdist(A[i:i + 1024], A)
                d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float('inf')
                idx.append(d.topk(self.k_nn, largest=False).indices)
            nb[m] = torch.cat(idx).numpy()
        return nb

    def fit(self, ctx, epochs=300):
        ms = ctx.mods
        F = {m: pool(ctx.X_train[m]) for m in ms}
        self.norm = {m: (F[m].mean(0), F[m].std(0) + 1e-6) for m in ms}
        self.xst = {m: (ctx.X_train[m].mean((0, 1)), ctx.X_train[m].std((0, 1)) + 1e-6) for m in ms}
        torch.manual_seed(ctx.seed + 3)
        d = sum(F[m].shape[1] + 1 for m in ms)
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, len(ms) + 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.default_rng(ctx.seed + 5)
        Xtr = ctx.X_train
        n = len(ctx.y_train)
        nb = self._neighbours(ctx, F) if self.hard else None
        for ep in range(epochs):
            lab = rng.integers(0, len(ms) + 1, size=n)        # 0 = clean, j+1 = modality j replaced
            kind = rng.random(n) < 0.5 if len(self.kinds) == 2 else np.full(n, self.kinds[0] == 'white')
            X = {m: Xtr[m].copy() for m in ms}
            for j, m in enumerate(ms):
                sel = np.where(lab == j + 1)[0]
                w = sel[kind[sel]]; s = sel[~kind[sel]]
                mu, sd = self.xst[m]
                X[m][w] = (mu + sd * rng.normal(size=X[m][w].shape)).astype(np.float32)
                if nb is not None:                      # half hard (look-alike clip), half random
                    hard = rng.random(len(s)) < 0.5
                    partner = rng.permutation(n)[:len(s)]
                    partner[hard] = nb[m][s[hard], rng.integers(0, self.k_nn, size=hard.sum())]
                    X[m][s] = Xtr[m][partner]
                else:
                    X[m][s] = Xtr[m][rng.permutation(n)[:len(s)]]
            # random missing modalities so it works on subsets
            pres = rng.random((n, len(ms))) > 0.2
            pres[np.arange(n), np.maximum(lab - 1, 0)] |= lab > 0
            feats = torch.from_numpy(self._feats(X, ms, pres))
            logits = self.net(feats)
            logits = logits.masked_fill(torch.from_numpy(np.concatenate([np.ones((n, 1), bool), pres], 1)) == 0, -1e9)
            loss = nn.functional.cross_entropy(logits, torch.from_numpy(lab))
            opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    def _feats(self, X, ms, pres):
        cols = []
        for j, m in enumerate(ms):
            f = (pool(X[m]) - self.norm[m][0]) / self.norm[m][1] if m in X else None
            n = len(pres)
            if f is None:
                f = np.zeros((n, len(self.norm[m][0])), np.float32)
            f = f * pres[:, j:j + 1]
            cols += [f, pres[:, j:j + 1].astype(np.float32)]
        return np.concatenate(cols, 1).astype(np.float32)

    def score(self, P, X, ctx):
        ms = ctx.mods
        n = len(next(iter(P.values()))[0])
        pres = np.stack([np.full(n, m in P) for m in ms], 1)
        with torch.no_grad():
            lg = self.net(torch.from_numpy(self._feats(X, ms, pres)))
            lg = lg.masked_fill(torch.from_numpy(np.concatenate([np.ones((n, 1), bool), pres], 1)) == 0, -1e9)
            p = torch.softmax(lg, 1).numpy()
        return {m: p[:, j + 1] for j, m in enumerate(ms) if m in P}


class OneClassNoiseDet(Detector):
    """Real-vs-white-noise classifier per modality [self-sup] (noise-contrastive)."""

    def fit(self, ctx, epochs=200):
        self.nets = {}
        rng = np.random.default_rng(ctx.seed + 11)
        for m in ctx.mods:
            X = ctx.X_train[m]
            F = pool(X); mu, sd = F.mean(0), F.std(0) + 1e-6
            xmu, xsd = X.mean((0, 1)), X.std((0, 1)) + 1e-6
            torch.manual_seed(ctx.seed + 13)
            net = nn.Sequential(nn.Linear(F.shape[1], 64), nn.GELU(), nn.Linear(64, 1))
            opt = torch.optim.Adam(net.parameters(), lr=3e-3)
            for _ in range(epochs):
                Nz = (xmu + xsd * rng.normal(size=X.shape)).astype(np.float32)
                A = torch.from_numpy(np.concatenate([(F - mu) / sd, (pool(Nz) - mu) / sd]).astype(np.float32))
                L = torch.cat([torch.zeros(len(F)), torch.ones(len(F))])
                loss = nn.functional.binary_cross_entropy_with_logits(net(A)[:, 0], L)
                opt.zero_grad(); loss.backward(); opt.step()
            self.nets[m] = (net.eval(), mu, sd)
        return self

    def score(self, P, X, ctx):
        out = {}
        for m in P:
            net, mu, sd = self.nets[m]
            with torch.no_grad():
                out[m] = net(torch.from_numpy(((pool(X[m]) - mu) / sd).astype(np.float32)))[:, 0].numpy()
        return out


class AgreeDet(Detector):
    """CLIP / SyncNet-style cross-modal agreement: per-modality encoders trained with pairwise InfoNCE on
    clean clips (in-batch negatives); suspicion of m = low agreement of m with the others, relative to theirs."""

    def fit(self, ctx, epochs=300, d=32, temp=0.1):
        ms = ctx.mods
        F = {m: pool(ctx.X_train[m]) for m in ms}
        self.norm = {m: (F[m].mean(0), F[m].std(0) + 1e-6) for m in ms}
        Z = {m: torch.from_numpy(((F[m] - self.norm[m][0]) / self.norm[m][1]).astype(np.float32)) for m in ms}
        torch.manual_seed(ctx.seed + 17)
        self.enc = nn.ModuleDict({m: nn.Sequential(nn.Linear(Z[m].shape[1], 64), nn.GELU(), nn.Linear(64, d))
                                  for m in ms})
        opt = torch.optim.Adam(self.enc.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Z[ms[0]])
        g = torch.Generator().manual_seed(ctx.seed)
        for ep in range(epochs):
            idx = torch.randperm(n, generator=g)[:512]
            E = {m: nn.functional.normalize(self.enc[m](Z[m][idx]), dim=1) for m in ms}
            loss = 0
            for a, b in itertools.combinations(ms, 2):
                L = E[a] @ E[b].T / temp
                lab = torch.arange(len(idx))
                loss = loss + nn.functional.cross_entropy(L, lab) + nn.functional.cross_entropy(L.T, lab)
            opt.zero_grad(); loss.backward(); opt.step()
        self.enc.eval()
        return self

    def score(self, P, X, ctx):
        avail = [m for m in ctx.mods if m in P]
        with torch.no_grad():
            E = {m: nn.functional.normalize(self.enc[m](torch.from_numpy(
                ((pool(X[m]) - self.norm[m][0]) / self.norm[m][1]).astype(np.float32))), dim=1) for m in avail}
        out = {}
        for m in avail:
            others = [k for k in avail if k != m]
            out[m] = -sum((E[m] * E[k]).sum(1) for k in others).numpy() / max(len(others), 1) if others \
                else np.zeros(len(E[m]))
        if len(out) >= 3:
            tot = sum(out.values())
            out = {m: out[m] - (tot - out[m]) / (len(out) - 1) for m in out}
        return out


class ContamSuspDet(Detector):
    """Posterior P(bad) from the contamination model, used as a detector."""

    def score(self, P, X, ctx):
        if len(P) < 3:
            return {m: np.zeros(len(P[m][0])) for m in P}
        _, _, s = t_contam(P, X, ctx)
        return {m: s[:, ctx.mods.index(m)] for m in P}


class SumDet(Detector):
    """Combination: sum of calibrated (threshold-normalised) suspicions of several detectors."""

    def __init__(self, dets):
        self.dets = dets

    def score(self, P, X, ctx):
        outs = [d.score(P, X, ctx) for d in self.dets]
        return {m: sum(o[m] / d.thr[m] for o, d in zip(outs, self.dets)) for m in P}


# ----------------------------------------------------------------------------- detector -> fusion
class Gate:
    """Turn a detector into a fusion rule.
    mode='drop'  : drop the most suspicious modality if it exceeds its clean 99th percentile
    mode='soft'  : inflate var_m by (s_m / thr_m)^k for every modality above threshold
    mode='widen' : keep PoE mean, but inflate the fused variance when anything is flagged"""

    def __init__(self, det, mode='drop', k=2.0, base='poe_pc'):
        self.det, self.mode, self.k, self.base = det, mode, k, base

    def __call__(self, P, X, ctx):
        prior = ctx.prior if self.base == 'poe_pc' else None
        ms = list(P)
        s = self.det.score(P, X, ctx)
        r = {m: s[m] / (self.det.thr[m] if self.det.thr[m] > 0 else 1.0) for m in ms}
        susp = stack_susp(P, ctx.mods, r)
        if len(ms) < 2:
            return (*poe(P, prior), susp)
        if self.mode == 'drop':
            R = np.stack([r[m] for m in ms], 1)
            j = R.argmax(1); flag = R.max(1) > 1
            mu, v = poe(P, prior)
            for jj, m in enumerate(ms):
                sel = flag & (j == jj)
                if sel.any():
                    Q = {k: (P[k][0][sel], P[k][1][sel]) for k in ms if k != m}
                    mu[sel], v[sel] = poe(Q, prior)
            return mu, v, susp
        if self.mode == 'multi':
            # drop EVERY modality above threshold, but always keep the least suspicious one
            R = np.stack([r[m] for m in ms], 1)
            keep = R <= 1
            keep[np.arange(len(R)), R.argmin(1)] = True
            mu, v = np.zeros(len(R)), np.zeros(len(R))
            for pat in {tuple(k) for k in keep}:
                sel = (keep == np.array(pat)).all(1)
                Q = {m: (P[m][0][sel], P[m][1][sel]) for m, kk in zip(ms, pat) if kk}
                mu[sel], v[sel] = poe(Q, prior)
            return mu, v, susp
        if self.mode == 'soft':
            # fractional expert: precision weight 1 / r^k above threshold (prior correction follows the weight)
            w = {m: np.minimum(1.0, np.maximum(r[m], 1e-9) ** -self.k) for m in ms}
            return (*poe_w(P, w, prior), susp)
        if self.mode == 'widen':
            mu, v = poe(P, prior)
            R = np.stack([r[m] for m in ms], 1).max(1)
            return mu, v * np.clip(R, 1.0, 5.0) ** self.k, susp
        raise ValueError(self.mode)
