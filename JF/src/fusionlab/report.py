"""Numbers-first reporting: one scoreboard table and one bar chart per question.

Every notebook ends with `scoreboard(...)`: rows = methods, columns = a few headline metrics, each shown as
mean +/- std over seeds, best value marked. `bar(...)` draws the matching chart with an optional target line.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEADLINE = {            # metric: (label, better, target)
    'MAE': ('MAE', 'lower', None),
    'Corr': ('Corr', 'higher', None),
    'Acc2': ('Acc-2', 'higher', None),
    'NLL': ('NLL', 'lower', None),
    'Cov90': ('Coverage of 90% range', 'target', 0.90),
    'Width90': ('Width of 90% range', 'lower', None),
    'CalErr': ('Calibration error', 'lower', 0.0),
    'Spearman_unc_err': ('Uncertainty-error rank corr.', 'higher', None),
    'AUROC_bigerr': ('Error-flag AUROC', 'higher', 0.5),
    'det_AUROC_sd': ('Corruption-flag AUROC (sigma)', 'higher', 0.5),
    'det_AUROC_dis': ('Corruption-flag AUROC (disagreement)', 'higher', 0.5),
    'det_AUROC_pe': ('Corruption-flag AUROC (ensemble)', 'higher', 0.5),
    'det_AUROC_epi': ('Corruption-flag AUROC (masking ens.)', 'higher', 0.5),
    'probe_signal_R2': ('Signal R2', 'higher', None),
    'probe_noise_R2': ('Distractor R2', 'lower', 0.0),
    'share_n': ('Weight on noise channel', 'lower', 0.0),
    'E_AURC': ('Excess AURC', 'lower', 0.0),
}

PALETTE = ['#2a6f97', '#e07a1f', '#3a9d5d', '#c03a3a', '#7b5ea7', '#8c6d46', '#d16ba5', '#6b7280', '#b5a32b',
           '#17a2b8', '#5a3e2b', '#9aa0a6']


def method_label(row, by):
    return ' / '.join(str(row[k]) for k in by)


def per_seed(df, metrics, by=('objective', 'regime', 'fusion')):
    """Average each metric over the selected conditions within a model, keeping one value per seed."""
    by = list(by)
    return df.groupby(by + ['seed'])[[m for m in metrics if m in df]].mean().reset_index()


def scoreboard(df, metrics, by=('objective', 'regime', 'fusion'), title='', save=None, sort=None):
    """mean +/- std over seeds; the best mean per column is marked with *."""
    by = list(by)
    ps = per_seed(df, metrics, by)
    metrics = [m for m in metrics if m in ps]
    g = ps.groupby(by)[metrics]
    mean, std, n = g.mean(), g.std().fillna(0), g.size()
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        lab, better, tgt = HEADLINE.get(m, (m, 'lower', None))
        col = mean[m]
        if col.isna().all():
            best = None
        elif better == 'target':
            best = (col - tgt).abs().idxmin()
        elif better == 'higher':
            best = col.idxmax()
        else:
            best = col.idxmin()
        out[lab] = [f'{mu:.3f} ± {sd:.3f}' + (' *' if idx == best else '')
                    for idx, mu, sd in zip(mean.index, col, std[m])]
    out['seeds'] = n
    if sort:
        lab = HEADLINE.get(sort, (sort,))[0]
        asc = HEADLINE.get(sort, (None, 'lower'))[1] != 'higher'
        out = out.iloc[np.argsort((1 if asc else -1) * mean[sort].values)]
    if title:
        print(f'\n### {title}')
    if save:
        os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
        out.to_csv(save)
        mean.add_suffix('_mean').join(std.add_suffix('_std')).to_csv(save.replace('.csv', '_raw.csv'))
    return out


def wins(df, metric, a, b, pair_on=('objective', 'regime', 'seed'), col='fusion'):
    """In how many seed-paired comparisons does method a beat method b on `metric`?"""
    better = HEADLINE.get(metric, (metric, 'lower', None))[1]
    if metric not in df or df[metric].isna().all():
        return 0, 0
    ps = df.groupby(list(pair_on) + [col])[metric].mean().unstack(col)
    ps = ps[[a, b]].dropna()
    if better == 'target':
        tgt = HEADLINE[metric][2]
        w = ((ps[a] - tgt).abs() < (ps[b] - tgt).abs())
    elif better == 'higher':
        w = ps[a] > ps[b]
    else:
        w = ps[a] < ps[b]
    return int(w.sum()), int(len(w))


def bar(df, metric, x='fusion', hue=None, title='', ylabel=None, save=None, target='auto', ax=None,
        order=None):
    """Bar chart of mean +/- std over seeds. target='auto' draws the metric's ideal value when it has one."""
    lab, better, tgt = HEADLINE.get(metric, (metric, 'lower', None))
    keys = [x] + ([hue] if hue else [])
    ps = df.groupby(keys + ['seed'])[metric].mean().reset_index()
    st = ps.groupby(keys)[metric].agg(['mean', 'std']).reset_index()
    xs = order or list(dict.fromkeys(st[x]))
    hs = list(dict.fromkeys(st[hue])) if hue else [None]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(xs) * max(1, len(hs)) + 1.5), 3.6))
    w = 0.8 / len(hs)
    for j, h in enumerate(hs):
        sub = st if h is None else st[st[hue] == h]
        sub = sub.set_index(x).reindex(xs)
        pos = np.arange(len(xs)) + (j - (len(hs) - 1) / 2) * w
        ax.bar(pos, sub['mean'], w * 0.92, yerr=sub['std'].fillna(0), capsize=3,
               color=PALETTE[j % len(PALETTE)], label=None if h is None else str(h))
        for p_, v in zip(pos, sub['mean']):
            if np.isfinite(v):
                ax.text(p_, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    if target == 'auto':
        target = tgt
    if target is not None:
        ax.axhline(target, color='#444', ls='--', lw=1)
        ax.text(len(xs) - 0.5, target, f' target {target:g}', va='bottom', ha='right', fontsize=8, color='#444')
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=20, ha='right')
    ax.set_ylabel(ylabel or lab + (' (lower is better)' if better == 'lower' else
                                   ' (higher is better)' if better == 'higher' else ''))
    ax.set_title(title, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    if hue:
        ax.legend(fontsize=8, frameon=False)
    if own:
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
            plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
    return ax


def headline(results_csv, out_csv, rows):
    """Collect one-line findings: rows = list of dicts(question, number, source). Saved for notebook 08."""
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    for r in rows:
        print(f"- {r['question']}: {r['number']}")


TXT, NOTXT = ['T', 'TA', 'TV', 'TAV'], ['A', 'V', 'AV']
SUBSETS = ['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV']
HEADLINE['tracking'] = ('Tracking ratio (sigma rise / error rise)', 'target', 1.0)
HEADLINE['sd_ratio'] = ('Sigma ratio corrupted / clean', 'higher', 1.0)
HEADLINE['mae_ratio'] = ('MAE ratio corrupted / clean', 'lower', 1.0)
HEADLINE['cov_gap'] = ('Worst per-subset coverage gap', 'lower', 0.0)


def tracking(df, by=('objective', 'regime', 'fusion')):
    """Does predicted sigma rise as much as the error when text is lost?
    (mean sigma without text / with text) / (MAE without text / with text); 1 = sigma tracks the error."""
    by = list(by)
    d = df[df.cond.isin(SUBSETS)].assign(txt=lambda x: x.cond.isin(TXT))
    g = d.groupby(by + ['seed', 'txt'])[['MeanSD', 'MAE']].mean().unstack('txt')
    out = (g[('MeanSD', False)] / g[('MeanSD', True)]) / (g[('MAE', False)] / g[('MAE', True)])
    return out.rename('tracking').reset_index()


def ratio_to_clean(df, clean_df, metric, name, by=('objective', 'regime', 'fusion')):
    """metric under each condition divided by the same model's clean (TAV) value."""
    by = list(by) + ['seed']
    c = clean_df.groupby(by)[metric].mean().rename('clean')
    d = df.groupby(by + ['cond'])[metric].mean().reset_index().join(c, on=by)
    d[name] = d[metric] / d['clean']
    return d
