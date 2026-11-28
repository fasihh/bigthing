"""Leaderboard from a screen CSV.  python -m bench.leaderboard bench/results/screen.csv [ref]"""
import sys

import numpy as np
import pandas as pd


def board(path, ref='poe_pc'):
    d = pd.read_csv(path)
    full = d[(d.kind.isin(['white', 'ar', 'swap'])) & (d.lam == 1.0)]
    half = d[(d.kind == 'white') & (d.lam == 0.5)]
    clean = d[d.kind == 'clean']
    miss = d[d.kind == 'missing']
    g = lambda df, col: df.groupby(['technique', 'seed'])[col].mean().groupby('technique')
    out = pd.DataFrame({
        'clean_NLL': g(clean, 'NLL').mean(),
        'miss_NLL': g(miss, 'NLL').mean(),
        'corr_MAE': g(full, 'MAE').mean(),
        'corr_NLL': g(full, 'NLL').mean(),
        'corr_NLL_sd': g(full, 'NLL').std(),
        'corr_Cov90': g(full, 'Cov90').mean(),
        'half_NLL': g(half, 'NLL').mean(),
        'AUC_white': g(full[full.kind == 'white'], 'AUROC').mean(),
        'AUC_ar(held-out)': g(full[full.kind == 'ar'], 'AUROC').mean(),
        'AUC_swap': g(full[full.kind == 'swap'], 'AUROC').mean(),
        'loc_acc': g(full, 'loc_acc').mean(),
    })
    # overall deployment score: average NLL over every condition (clean, missing, corrupted)
    out['all_NLL'] = g(d, 'NLL').mean()
    # paired wins vs reference on corrupted NLL (per seed)
    piv = full.groupby(['technique', 'seed']).NLL.mean().unstack()
    if ref in piv.index:
        out['wins_vs_' + ref] = (piv.lt(piv.loc[ref], axis=1)).sum(1).astype(int).astype(str) + '/' + str(piv.shape[1])
    return out.sort_values('all_NLL')


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    b = board(sys.argv[1], *(sys.argv[2:3]))
    print(b.round(3).to_string())
