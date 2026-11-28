"""Run every method on a cache and print an uncertainty-tracking leaderboard.

    python -m lab.run toy --seeds 0 1 2          # synthetic, MOSEI-like
    python -m lab.run real --cache uq_cache.npz --tags vjepa_frozen vjepa_vt_frozen vjepa_finetuned
"""
import argparse
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import methods as M
from .cache import load_real

Z90 = M.Z90


def registry(det):
    P = M.PoE
    return {
        'poe': lambda: M.PoE(det),
        'conformal-clean': lambda: M.ConfClean(det),
        'conformal-selfgen': lambda: M.ConfMix(det),
        'jackknife-clean': lambda: M.Jackknife(det),
        'jackknife-selfgen': lambda: M.JackknifeMix(det),
        'gate-drop': lambda: M.GateDrop(det),
        'pattern-BMA': lambda: M.PatternBMA(det),
        'meta-sigma (no detector)': lambda: M.MetaSigma(det, use_det=False),
        'meta-sigma': lambda: M.MetaSigma(det),
        'meta-sigma + mean': lambda: M.MetaSigma(det, predict_mean=True),
        'meta-sigma on BMA': lambda: M.MetaSigma(det, base='bma'),
        'meta-sigma on gate': lambda: M.MetaSigma(det, base='gate'),
        'mondrian(poe)': lambda: M.MondrianDet(det, inner=M.PoE(det)),
        'mondrian(BMA)': lambda: M.MondrianDet(det, inner=M.PatternBMA(det)),
        'mondrian(meta-sigma on BMA)': lambda: M.MondrianDet(det, inner=M.MetaSigma(det, base='bma')),
        'oracle drop': lambda: M.OracleDrop(det),
    }


def resplit(C, seed=0):
    """Methods that learn from residuals must not use the clips the model was trained on: use valid half A as
    their training split and valid half B for calibration."""
    n = len(C['y']['valid'])
    idx = np.random.default_rng(seed).permutation(n)
    A, B = idx[:n // 2], idx[n // 2:]
    out = dict(C, y=dict(C['y']), F=dict(C['F']), MU=dict(C['MU']), VAR=dict(C['VAR']))
    for k in ('F', 'MU', 'VAR'):
        out[k]['train'] = {c: v[A] for c, v in C[k]['valid'].items()}
        out[k]['valid'] = {c: v[B] for c, v in C[k]['valid'].items()}
    out['y']['train'], out['y']['valid'] = C['y']['valid'][A], C['y']['valid'][B]
    return out


def evaluate(C, method, tag, seed):
    rows = []
    y = C['y']['test']
    clean = None
    for cond, meta in C['meta'].items():
        mu, v = method.predict(C, 'test', cond)
        sd = np.sqrt(v)
        r = dict(tag=tag, seed=seed, cond=cond, kind=meta['kind'], heldout=meta['heldout'],
                 MAE=np.mean(np.abs(y - mu)), NLL=M.nll(y, mu, v), Cov90=np.mean(np.abs(y - mu) <= Z90 * sd),
                 SD=sd.mean())
        if cond == 'clean':
            clean = (r['MAE'], r['SD'], sd)
        rows.append(r)
    for r, (cond, meta) in zip(rows, C['meta'].items()):
        if cond != 'clean':
            mu, v = method.predict(C, 'test', cond)
            r['TR'] = (r['SD'] / clean[1]) / (r['MAE'] / clean[0])
            r['AUROC_sd'] = roc_auc_score(np.r_[np.zeros(len(clean[2])), np.ones(len(v))], np.r_[clean[2], np.sqrt(v)])
    return rows


def summary(df):
    c = df[df.cond != 'clean']
    g = lambda d, col: d.groupby(['method', 'tag', 'seed'])[col].mean().groupby('method').mean()
    out = pd.DataFrame({
        'clean_NLL': g(df[df.cond == 'clean'], 'NLL'),
        'clean_Cov': g(df[df.cond == 'clean'], 'Cov90'),
        'seen_NLL': g(c[~c.heldout], 'NLL'),
        'held_NLL': g(c[c.heldout], 'NLL'),
        'held_MAE': g(c[c.heldout], 'MAE'),
        'seen_covgap': g(c[~c.heldout].assign(gap=(c.Cov90 - 0.9).abs()), 'gap'),
        'held_covgap': g(c[c.heldout].assign(gap=(c.Cov90 - 0.9).abs()), 'gap'),
        'held_worstcov': c[c.heldout].groupby(['method', 'tag', 'seed']).Cov90.min().groupby('method').mean(),
        'held_TR': g(c[c.heldout], 'TR'),
        'held_logTRerr': g(c[c.heldout].assign(e=np.abs(np.log(c.TR))), 'e'),
        'held_AUROC_sd': g(c[c.heldout], 'AUROC_sd'),
    })
    return out.sort_values('held_NLL')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world', choices=['toy', 'real'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--cache', default='uq_cache.npz')
    ap.add_argument('--tags', nargs='+', default=['vjepa_frozen'])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default='lab/results/toy.csv')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rows = []
    runs = [(t, 0) for t in a.tags] if a.world == 'real' else [('toy', s) for s in a.seeds]
    for tag, seed in runs:
        if a.world == 'toy':
            pk = f'/tmp/toycache_{seed}.pkl'
            if os.path.exists(pk):
                C = pickle.load(open(pk, 'rb'))
            else:
                from .toy import build
                C = build(seed); pickle.dump(C, open(pk, 'wb'))
        else:
            C = load_real(a.cache, tag)
        det = M.Detectors().fit(C, seed=seed)          # input-level: trained on the train split
        if a.world == 'real':
            C = resplit(C)                             # the model was fitted on train: learn/calibrate on valid halves
        for name, make in registry(det).items():
            if a.only and name not in a.only:
                continue
            meth = make().fit(C)
            for r in evaluate(C, meth, tag, seed):
                r['method'] = name
                rows.append(r)
            print(tag, seed, name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option('display.width', 250)
    print(summary(df).round(3).to_string())


if __name__ == '__main__':
    main()
