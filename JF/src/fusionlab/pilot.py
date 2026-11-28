"""FYP-2 go/no-go pilot: can a label-free cross-modal consistency score tell WHICH modality is corrupted,
where each expert's own sigma cannot?

Uses frozen stage-1 experts only (no labels, no retraining). For every test clip it computes, per modality m:
  sigma      : the expert's own predicted sigma (FYP-1 baseline)
  disagree   : mean symmetric KL to the other experts (FYP-1 baseline)
  cond_maha  : conditional Mahalanobis distance of expert m's (PCA-reduced) belief given the other experts' beliefs,
               under a joint Gaussian fitted on CLEAN training clips (the core of "Liar's Posterior Fusion")
  xpred      : standardised residual of ridge predictions of expert m's mean from the other experts' means,
               fitted on clean training clips (a linear, training-free stand-in for the cross-modal residual graph)
and the posterior over the 2^M corruption configurations (LPF) with a broad outlier model.
Reported: detection AUROC (corrupted clip vs clean clip, score of the corrupted modality) and localisation accuracy
(argmax over modalities = the corrupted one) for noise and swap corruptions.
"""
import itertools
import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .batching import get_batch
from .models import expert_beliefs
from .runtime import BASE, RT, set_seed


@torch.no_grad()
def beliefs(experts, split, corrupt=None, seed=123):
    """corrupt: None | ('noise', mods, lam) | ('swap', mods). Returns mus, lvs as numpy (N, M, d)."""
    set_seed(seed)
    g = torch.Generator(device=RT.DEV).manual_seed(seed)
    n = RT.N_SPLIT[split]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 7))
    MU, LV = [], []
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(n, i + 1024))
        b = get_batch(split, idx, list(experts.keys()), g)
        xs, masks = {}, {}
        if corrupt and corrupt[0] == 'noise':
            for m in corrupt[1]:
                eps = torch.randn(b[m].shape, device=RT.DEV, generator=g) * RT.FEAT_STD[m]
                lam = corrupt[2]
                xs[m] = (math.sqrt(1 - lam) * b[m] + math.sqrt(lam) * eps) * b[m + 'm'].unsqueeze(-1)
        if corrupt and corrupt[0] == 'swap':
            for m in corrupt[1]:
                j = perm[idx]
                xs[m] = RT.DATA[split][m][j].to(RT.DEV)
                masks[m] = RT.DATA[split][m + 'm'][j].to(RT.DEV)
        mu, lv = expert_beliefs(experts, b, masks, xs)
        MU.append(mu.cpu()); LV.append(lv.cpu())
    return torch.cat(MU).numpy().astype(np.float64), torch.cat(LV).numpy().astype(np.float64)


class ConsistencyModel:
    """Label-free joint model of the experts' beliefs, fitted on clean training clips."""

    def __init__(self, mus, k=8, shrink=0.05, kappa=10.0, prior=0.05):
        N, M, d = mus.shape
        self.M, self.k, self.kappa, self.prior = M, k, kappa, prior
        self.P, self.c = [], []
        U = []
        for m in range(M):
            X = mus[:, m] - mus[:, m].mean(0)
            _, _, Vt = np.linalg.svd(X[:20000], full_matrices=False)
            self.P.append(Vt[:k].T)
            self.c.append(mus[:, m].mean(0))
            U.append(X @ Vt[:k].T)
        U = np.concatenate(U, 1)
        self.mean = U.mean(0)
        C = np.cov(U.T)
        self.C = (1 - shrink) * C + shrink * np.diag(np.diag(C))       # simple shrinkage for stability
        # ridge cross-prediction of each expert's mean from the others' means
        self.W = []
        for m in range(M):
            others = [j for j in range(M) if j != m]
            Xo = np.concatenate([mus[:, j] for j in others], 1)
            Xo1 = np.c_[Xo, np.ones(len(Xo))]
            W = np.linalg.solve(Xo1.T @ Xo1 + 1.0 * np.eye(Xo1.shape[1]), Xo1.T @ mus[:, m])
            R = mus[:, m] - Xo1 @ W
            self.W.append((W, R.var(0) + 1e-6))

    def u(self, mus):
        return np.concatenate([(mus[:, m] - self.c[m]) @ self.P[m] for m in range(self.M)], 1)

    def blocks(self, m):
        return slice(m * self.k, (m + 1) * self.k)

    def cond_maha(self, mus):
        """Per modality: Mahalanobis^2 / k of u_m given u_{-m} under the clean joint Gaussian."""
        U = self.u(mus) - self.mean
        out = []
        for m in range(self.M):
            a = np.arange(self.M * self.k)
            i = a[self.blocks(m)]
            o = np.setdiff1d(a, i)
            Coo_inv = np.linalg.inv(self.C[np.ix_(o, o)])
            A = self.C[np.ix_(i, o)] @ Coo_inv
            S = self.C[np.ix_(i, i)] - A @ self.C[np.ix_(o, i)]
            r = U[:, i] - U[:, o] @ A.T
            out.append(np.einsum('ni,ij,nj->n', r, np.linalg.inv(S), r) / self.k)
        return np.stack(out, 1)

    def xpred(self, mus):
        out = []
        for m in range(self.M):
            others = [j for j in range(self.M) if j != m]
            Xo1 = np.c_[np.concatenate([mus[:, j] for j in others], 1), np.ones(len(mus))]
            W, v = self.W[m]
            out.append((((mus[:, m] - Xo1 @ W) ** 2) / v).mean(1))
        return np.stack(out, 1)

    def liar_posterior(self, mus):
        """P(modality m corrupted | all beliefs) under 2^M configurations, broad independent outlier model."""
        U = self.u(mus) - self.mean
        configs = list(itertools.product([0, 1], repeat=self.M))           # 1 = corrupted
        logp = []
        for cfg in configs:
            R = [m for m in range(self.M) if cfg[m] == 0]
            lp = np.zeros(len(U))
            if R:
                idx = np.concatenate([np.arange(self.M * self.k)[self.blocks(m)] for m in R])
                C = self.C[np.ix_(idx, idx)]
                Ci = np.linalg.inv(C)
                _, ld = np.linalg.slogdet(C)
                lp += -0.5 * (np.einsum('ni,ij,nj->n', U[:, idx], Ci, U[:, idx]) + ld + len(idx) * np.log(2 * np.pi))
            for m in range(self.M):
                if cfg[m] == 1:
                    C = self.kappa * self.C[self.blocks(m), self.blocks(m)]
                    x = U[:, self.blocks(m)]
                    _, ld = np.linalg.slogdet(C)
                    lp += -0.5 * (np.einsum('ni,ij,nj->n', x, np.linalg.inv(C), x) + ld + self.k * np.log(2 * np.pi))
            lp += sum(np.log(self.prior) if c else np.log(1 - self.prior) for c in cfg)
            logp.append(lp)
        logp = np.stack(logp, 1)
        post = np.exp(logp - logp.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
        return np.stack([post[:, [i for i, c in enumerate(configs) if c[m] == 1]].sum(1) for m in range(self.M)], 1)


def sym_kl_to_others(mus, lvs):
    M = mus.shape[1]
    out = np.zeros(mus.shape[:2])
    for m in range(M):
        acc = []
        for j in range(M):
            if j != m:
                vi, vj = np.exp(lvs[:, m]), np.exp(lvs[:, j])
                acc.append(0.25 * (vi / vj + vj / vi - 2 + (mus[:, m] - mus[:, j]) ** 2 * (1 / vi + 1 / vj)).sum(1))
        out[:, m] = np.mean(acc, 0)
    return out


def scores(cm, mus, lvs):
    return {'sigma (own)': np.exp(0.5 * lvs).mean(2), 'disagreement': sym_kl_to_others(mus, lvs),
            'cond. Mahalanobis': cm.cond_maha(mus), 'cross-prediction': cm.xpred(mus),
            'liar posterior': cm.liar_posterior(mus)}


def evaluate(cm, clean, corrupted, cases):
    """cases: {name: (corrupted modality indices)} -> rows of detection AUROC and localisation accuracy."""
    rows = []
    s_clean = scores(cm, *clean)
    for name, (mods_idx, data) in cases.items():
        s_cor = scores(cm, *data)
        for sname in s_clean:
            sc, sx = s_clean[sname], s_cor[sname]
            det = np.mean([roc_auc_score(np.r_[np.zeros(len(sc)), np.ones(len(sx))], np.r_[sc[:, m], sx[:, m]])
                           for m in mods_idx])
            loc = np.mean(np.isin(np.argmax(sx, 1), mods_idx)) if len(mods_idx) == 1 else np.nan
            rows.append(dict(case=name, score=sname, detect_AUROC=det, localise_acc=loc))
    return rows
