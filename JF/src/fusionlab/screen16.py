"""Idea screen for Part II (notebook 16): detect a corrupted expert, then act on it, all on FROZEN models.

Why: notebook 15 showed that surprise from the leave-one-out product works only when the experts' latents are
aligned (contrastive experts). V-JEPA experts do not share such a space, so their raw disagreement means little.
The ideas below learn the clean relations between experts from data instead of assuming them.

Detectors (a score per expert, standardised on clean training clips; flagged above the clean 99th percentile)
  loo     : distance from the leave-one-out product of the others (notebook 15 baseline)
  parity  : data-driven parity relation from fault diagnosis. A ridge map predicts expert m's mean from the other
            experts' means; the residual is whitened with its clean covariance. Learned redundancy, any latent space.
  spe     : PCA fault detection. PCA of all experts' means on clean clips; the squared prediction error outside the
            main subspace, split into each expert's share (contribution analysis).
  self    : each expert's own belief (mean and log-variance) against a Gaussian of clean beliefs (no redundancy).
  max     : the larger of parity and self (standardised).

Actions once an expert is flagged
  temper  : scale its precision down by (z / c)^-k inside the product (k tuned)
  widen   : keep the prediction, inflate the fused variance by 1 + a * sum_m relu(z_m / c_m - 1) (a tuned)
  impute  : replace the flagged expert's belief by the parity prediction from the others, with the clean residual
            variance as its uncertainty (accommodation step of detect-isolate-accommodate)
  temper+widen : both
"""
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from .surprise import surprise as loo_surprise


def _whiten_inv(R, eps=1e-3):
    C = np.cov(R, rowvar=False) + eps * np.eye(R.shape[1])
    return np.linalg.inv(C), np.diag(C)


class Parity:
    """Expert m's mean predicted (ridge) from the other experts' means, fitted on clean clips."""

    def fit(self, mus, alpha=1.0):
        mus = mus.numpy().astype(np.float64)
        B, M, D = mus.shape
        self.W, self.b, self.Pinv, self.var = [], [], [], []
        for m in range(M):
            X = mus[:, [k for k in range(M) if k != m]].reshape(B, -1)
            Y = mus[:, m]
            xm, ym = X.mean(0), Y.mean(0)
            Xc, Yc = X - xm, Y - ym
            W = np.linalg.solve(Xc.T @ Xc + alpha * len(X) * np.eye(X.shape[1]) * 1e-3, Xc.T @ Yc)
            R = Yc - Xc @ W
            Pinv, var = _whiten_inv(R)
            self.W.append(W); self.b.append((xm, ym)); self.Pinv.append(Pinv); self.var.append(var)
        return self

    def predict(self, mus, m):
        mus = mus.numpy().astype(np.float64)
        B, M, _ = mus.shape
        X = mus[:, [k for k in range(M) if k != m]].reshape(B, -1)
        xm, ym = self.b[m]
        return (X - xm) @ self.W[m] + ym

    def scores(self, mus):
        M, D = mus.shape[1], mus.shape[2]
        out = []
        for m in range(M):
            r = mus[:, m].numpy().astype(np.float64) - self.predict(mus, m)
            out.append(np.einsum('bi,ij,bj->b', r, self.Pinv[m], r) / D)
        return torch.tensor(np.stack(out, 1), dtype=torch.float32)


class SPE:
    """PCA of concatenated expert means; per-expert share of the residual outside the main subspace."""

    def fit(self, mus, keep=0.9):
        X = mus.numpy().astype(np.float64).reshape(len(mus), -1)
        self.M, self.D = mus.shape[1], mus.shape[2]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        r = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), keep)) + 1
        self.P = Vt[:r].T
        return self

    def scores(self, mus):
        Z = (mus.numpy().astype(np.float64).reshape(len(mus), -1) - self.mu) / self.sd
        E = Z - (Z @ self.P) @ self.P.T
        return torch.tensor((E ** 2).reshape(len(mus), self.M, self.D).mean(-1), dtype=torch.float32)


class SelfDensity:
    """Each expert's own (mean, log-variance) against a Ledoit-Wolf Gaussian of clean beliefs."""

    def fit(self, mus, lvs):
        self.g = []
        for m in range(mus.shape[1]):
            F = torch.cat([mus[:, m], lvs[:, m]], -1).numpy().astype(np.float64)
            lw = LedoitWolf().fit(F)
            self.g.append((lw.location_, lw.precision_))
        return self

    def scores(self, mus, lvs):
        out = []
        for m, (loc, P) in enumerate(self.g):
            F = torch.cat([mus[:, m], lvs[:, m]], -1).numpy().astype(np.float64) - loc
            out.append(np.einsum('bi,ij,bj->b', F, P, F) / F.shape[1])
        return torch.tensor(np.stack(out, 1), dtype=torch.float32)


class Standard:
    """z-score each expert's score on clean clips; c = clean q-quantile of z."""

    def fit(self, s, q=0.99):
        self.mean, self.sd = s.mean(0), s.std(0) + 1e-6
        self.c = torch.quantile((s - self.mean) / self.sd, q, dim=0).clamp(min=0.5)
        return self

    def __call__(self, s):
        return (s - self.mean) / self.sd


class Detectors:
    NAMES = ['loo', 'parity', 'spe', 'self', 'max']

    def fit(self, mus, lvs):
        self.parity = Parity().fit(mus)
        self.spe = SPE().fit(mus)
        self.self_ = SelfDensity().fit(mus, lvs)
        raw = self._raw(mus, lvs)
        self.std = {n: Standard().fit(s) for n, s in raw.items()}
        zs = {n: self.std[n](s) for n, s in raw.items()}
        r = torch.maximum(zs['parity'] / self.std['parity'].c, zs['self'] / self.std['self'].c)
        self.std['max'] = Standard().fit(r)
        self.std['max'].mean, self.std['max'].sd = torch.zeros_like(r[0]), torch.ones_like(r[0])
        self.std['max'].c = torch.quantile(r, 0.99, dim=0).clamp(min=0.5)
        return self

    def _raw(self, mus, lvs):
        return {'loo': loo_surprise(mus, lvs), 'parity': self.parity.scores(mus), 'spe': self.spe.scores(mus),
                'self': self.self_.scores(mus, lvs)}

    def z(self, mus, lvs):
        """Returns {name: (z (B,M), c (M,))}."""
        raw = self._raw(mus, lvs)
        out = {n: (self.std[n](s), self.std[n].c) for n, s in raw.items()}
        r = torch.maximum(out['parity'][0] / out['parity'][1], out['self'][0] / out['self'][1])
        out['max'] = (r, self.std['max'].c)
        return out


def temper_w(z, c, k):
    r = (z / c).clamp(min=1e-6)
    return torch.where(z > c, r ** (-k), torch.ones_like(r))


def widen_factor(z, c, a):
    return (1 + a * torch.relu(z / c - 1).sum(1)).numpy().astype(np.float64)


def impute(mus, lvs, z, c, parity):
    """Replace flagged experts' beliefs by the parity prediction (variance = clean residual variance)."""
    mus0, mus, lvs = mus, mus.clone(), lvs.clone()
    flag = z > c
    for m in range(mus.shape[1]):
        idx = flag[:, m]
        if idx.any():
            pred = torch.tensor(parity.predict(mus0, m), dtype=mus.dtype)
            mus[idx, m] = pred[idx]
            lvs[idx, m] = torch.log(torch.tensor(parity.var[m], dtype=lvs.dtype)).expand(int(idx.sum()), -1)
    return mus, lvs
