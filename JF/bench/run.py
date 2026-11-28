"""Run the noisy-TV screen: every technique x every condition x every seed -> one CSV row each.

    python -m bench.run --seeds 0 1 2 3 4 --out bench/results/screen.csv [--only name ...] [--noise-channel]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from . import data as D
from . import techniques as TQ
from .experts import Expert

N_TRAIN, N_CAL, N_TEST = 3000, 1500, 2000
Z90 = norm.ppf(0.95)


class Context:
    def __init__(self, seed, noise_channel=False, suite='main'):
        self.seed, self.suite = seed, suite
        self.world = D.World(seed, noise_channel)
        rng = np.random.default_rng(seed)
        self.X_train, self.y_train, _ = self.world.sample(N_TRAIN, rng)
        Xc, yc, _ = self.world.sample(N_CAL, rng)
        self.X_test, self.y_test, _ = self.world.sample(N_TEST, rng)
        self.spk_test = self.world.last_spk
        half = N_CAL // 2
        self.X_cal = {m: v[:half] for m, v in Xc.items()}; self.y_cal = yc[:half]
        self.X_cal2 = {m: v[half:] for m, v in Xc.items()}; self.y_cal2 = yc[half:]
        self.mods = list(self.X_train)                     # t, a, v (+ n)
        self.prior = (float(self.y_train.mean()), float(self.y_train.var()))
        self.st = D.stats(self.X_train)
        self.experts = {}
        self._cache = {}

    def expert_set(self, kind):
        if kind not in self.experts:
            self.experts[kind] = {m: Expert(kind, seed=self.seed).fit(self.X_train[m], self.y_train,
                                                                       self.X_cal[m], self.y_cal)
                                  for m in self.mods}
        return self.experts[kind]

    def beliefs(self, kind, X):
        E = self.expert_set(kind)
        return {m: E[m].predict(X[m])[:2] for m in X}

    def test_sets(self):
        """Corrupted test sets, identical for every technique (fixed rng per condition)."""
        if 'tests' not in self._cache:
            out = []
            conds = D.conditions() if self.suite == 'main' else D.heldout_conditions()
            for i, c in enumerate(conds):
                rng = np.random.default_rng(10_000 * (self.seed + 1) + i + (500 if self.suite != 'main' else 0))
                X, mask = D.corrupt(self.X_test, c, self.st, rng, spk=self.spk_test)
                X = {m: X[m] for m in X if mask[m]}
                out.append((c, X))
            self._cache['tests'] = out
        return self._cache['tests']


def build(name, ctx):
    """Registry: returns (expert_kind, fn). fn(P, X, ctx) -> mu, var, susp."""
    q = 0.99
    if '#' in name:
        name, qs = name.split('#'); q = float('0.' + qs)

    def gate(det, mode='drop', kind='plain', **kw):
        ctx.cal_P = ctx.beliefs(kind, ctx.X_cal)
        det.fit(ctx).calibrate(ctx, q)
        return kind, TQ.Gate(det, mode, **kw)

    simple = {
        'poe': TQ.t_poe, 'poe_pc': TQ.t_poe_pc, 'mean': TQ.t_mean, 'median': TQ.t_median,
        'huber': TQ.t_huber, 'student_t': TQ.t_student, 'trimmed': TQ.t_trimmed, 'cov_intersect': TQ.t_ci,
        'contam_bma': TQ.t_contam,
    }
    if name == 'oracle_drop':
        return 'plain', TQ.t_poe_pc
    if name in simple:
        return 'plain', simple[name]
    if name == 'contam_bma_b20':
        return 'plain', lambda P, X, c: TQ.t_contam(P, X, c, beta=0.2)
    if name == 'contam_bma_b01':
        return 'plain', lambda P, X, c: TQ.t_contam(P, X, c, beta=0.01)
    if name == 'learned_gate':
        ctx.cal_P = ctx.beliefs('plain', ctx.X_cal)
        return 'plain', TQ.LearnedGate().fit(ctx)
    if name == 'conformal_poe':
        return 'plain', TQ.t_poe_pc
    if name in ('ens_poe', 'mcd_poe', 'aug_poe', 'sn_poe'):
        return name.split('_')[0], TQ.t_poe_pc
    if name == 'aug_contam':
        return 'aug', TQ.t_contam
    dets = {
        'nis': TQ.NISDet, 'raim': TQ.RAIMDet, 'ddu': TQ.DDUDet, 'input_maha': TQ.InputMahaDet,
        'knn': TQ.KNNDet, 'whiteness': TQ.WhitenessDet, 'rnd': TQ.RNDDet, 'crossmodal': TQ.CrossModalDet,
        'predconsist': TQ.PredConsistDet, 'electra': TQ.ElectraDet, 'oneclass_noise': TQ.OneClassNoiseDet,
        'contam_det': TQ.ContamSuspDet, 'agree': TQ.AgreeDet,
        'electra_hard': lambda: TQ.ElectraDet(hard=True),
        'isoforest': TQ.IsoForestDet, 'gmm': TQ.GMMDet, 'tta': TQ.TTADet,
    }
    base, _, mode = name.partition('@')
    mode = mode or 'drop'
    kind = 'plain'
    if base.startswith('aug+'):
        kind, base = 'aug', base[4:]
    if base == 'electra_white':
        return gate(TQ.ElectraDet(('white',)), mode, kind=kind)
    if base == 'electra_swap':
        return gate(TQ.ElectraDet(('swap',)), mode, kind=kind)
    if base == 'ddu_sn':
        ctx.expert_set('sn')
        return gate(TQ.DDUDet('sn'), mode, kind='sn')
    if base in dets:
        return gate(dets[base](), mode, kind=kind)
    if base.startswith('fisher:'):
        parts = base[7:].split('+')
        ctx.cal_P = ctx.beliefs(kind, ctx.X_cal)
        ds = [dets[p]().fit(ctx).calibrate(ctx, q) for p in parts]
        return gate(TQ.FisherDet(ds), mode, kind=kind)
    if base.startswith('contam='):
        ctx.cal_P = ctx.beliefs(kind, ctx.X_cal)
        d = dets[base[7:]]().fit(ctx).calibrate(ctx, q)
        return kind, TQ.DetContam(d)
    if base.startswith('combo:'):
        parts = base[6:].split('+')
        ctx.cal_P = ctx.beliefs(kind, ctx.X_cal)
        ds = [dets[p]().fit(ctx).calibrate(ctx, q) for p in parts]
        return gate(TQ.SumDet(ds), mode, kind=kind)
    raise KeyError(name)


def temperature(fn, kind, ctx):
    """One variance scale per technique, fitted on clean calibration split 2 (uniform for all)."""
    P = ctx.beliefs(kind, ctx.X_cal2)
    mu, v, _ = fn(P, ctx.X_cal2, ctx)
    return float(np.mean((ctx.y_cal2 - mu) ** 2 / v))


def evaluate(name, ctx):
    t0 = time.time()
    kind, fn = build(name, ctx)
    tau = temperature(fn, kind, ctx)
    rows, clean_score = [], None
    for c, X in ctx.test_sets():
        P = ctx.beliefs(kind, X)
        if name == 'oracle_drop' and c[0] not in ('clean', 'missing'):
            P = {m: P[m] for m in P if m not in c[1]}
        mu, v, susp = fn(P, X, ctx)
        v = v * tau
        y = ctx.y_test
        sd = np.sqrt(v)
        r = dict(technique=name, seed=ctx.seed, cond=D.cond_name(c), kind=c[0],
                 corrupted=c[1] if c[0] not in ('clean', 'missing') else '', lam=c[2] if len(c) > 2 else np.nan,
                 MAE=np.mean(np.abs(y - mu)), NLL=np.mean(0.5 * np.log(2 * np.pi * v) + 0.5 * (y - mu) ** 2 / v),
                 Cov90=np.mean(np.abs(y - mu) <= Z90 * sd), SD=np.mean(sd), tau=tau)
        score = np.nanmax(susp, 1) if susp is not None else sd       # own-sigma when no detector
        if c[0] == 'clean':
            clean_score = score
        elif c[0] not in ('clean', 'missing') and clean_score is not None:
            lab = np.r_[np.zeros(len(clean_score)), np.ones(len(score))]
            r['AUROC'] = roc_auc_score(lab, np.r_[clean_score, score])
            if susp is not None and c[2] == 1.0 and len(c[1]) == 1:
                j = ctx.mods.index(c[1])
                r['loc_acc'] = np.mean(np.nanargmax(np.where(np.isnan(susp), -np.inf, susp), 1) == j)
        rows.append(r)
    for r in rows:
        r['secs'] = time.time() - t0
    return rows


COLS = ['technique', 'seed', 'cond', 'kind', 'corrupted', 'lam', 'MAE', 'NLL', 'Cov90', 'SD', 'tau', 'AUROC',
        'loc_acc', 'secs']

ALL = ['poe', 'poe_pc', 'mean', 'learned_gate', 'median', 'huber', 'student_t', 'trimmed', 'cov_intersect',
       'contam_bma', 'nis', 'raim', 'ddu', 'ddu_sn', 'input_maha', 'knn', 'whiteness', 'rnd', 'crossmodal',
       'predconsist', 'electra', 'oneclass_noise', 'ens_poe', 'mcd_poe', 'aug_poe', 'sn_poe']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default='bench/results/screen.csv')
    ap.add_argument('--noise-channel', action='store_true')
    ap.add_argument('--suite', default='main', choices=['main', 'heldout'])
    a = ap.parse_args()
    names = a.only or ALL
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out)
        done = set(zip(old.technique, old.seed))
    for seed in a.seeds:
        ctx = Context(seed, a.noise_channel, a.suite)
        for name in names:
            if (name, seed) in done:
                continue
            rows = evaluate(name, ctx)
            df = pd.DataFrame(rows).reindex(columns=COLS)
            df.to_csv(a.out, mode='a', header=not os.path.exists(a.out), index=False)
            g = df[~df.kind.isin(['clean', 'missing'])]
            print(f'seed {seed} {name:18s} clean NLL {df.NLL.iloc[0]:.3f} MAE {df.MAE.iloc[0]:.3f} | '
                  f'corrupt MAE {g.MAE.mean():.3f} NLL {g.NLL.mean():.3f} cov {g.Cov90.mean():.3f} '
                  f'AUROC {g.AUROC.mean():.3f} | {rows[0]["secs"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
