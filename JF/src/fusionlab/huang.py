"""Reproduction of Huang (2026), VJEPA/BJEPA noisy TV experiment, with more seeds.

The code is NOT copied into this repository: `fetch()` clones https://github.com/YongchaoHuang/VJEPA at run time
and `load_exp1()` executes the author's own multi-seed script (exp1_multiseed_variance) unchanged, except that the
`__main__` block is not run, so we can choose the seeds.
"""
import json
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import stats

REPO = 'https://github.com/YongchaoHuang/VJEPA'
EXP1 = 'exp1_multiseed_variance/VJEPA_[exp1_multiseed_variance].ipynb'
MODELS = ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']
PAPER_SEEDS = [42, 111, 123, 456, 789]


def fetch(dest='huang_vjepa'):
    if not os.path.isdir(dest):
        subprocess.run(['git', 'clone', '--depth', '1', REPO, dest], check=True)
    sha = subprocess.run(['git', '-C', dest, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    print('Huang repo at commit', sha)
    return dest, sha


def _code_cells(nb_path):
    nb = json.load(open(nb_path))
    return [c for c in nb['cells'] if c['cell_type'] == 'code']


def load_exp1(repo_dir):
    """Execute the author's experiment code (everything above `if __name__ == '__main__':`)."""
    cells = _code_cells(os.path.join(repo_dir, EXP1))
    src = next(''.join(c['source']) for c in cells if 'def run_one_seed' in ''.join(c['source']))
    src = src.split("if __name__ == '__main__':")[0]
    ns = {'__name__': 'huang_exp1'}
    exec(compile(src, 'huang_exp1', 'exec'), ns)
    return ns


def published_table(repo_dir):
    """Parse the author's own printed 5-seed table from the saved notebook output."""
    rows = []
    for c in _code_cells(os.path.join(repo_dir, EXP1)):
        for o in c.get('outputs', []):
            for line in ''.join(o.get('text', [])).splitlines():
                m = re.match(r'\s*([\d.]+) \(.*?\)\s*\|\s*(\w+)\s*\|\s*([-\d.]+) ± ([\d.]+)\s*\|\s*([-\d.]+) ± ([\d.]+)'
                             r'\s*\|\s*([-\d.]+) ± ([\d.]+)', line)
                if m:
                    rows.append(dict(scale=float(m[1]), model=m[2], sig_tr=float(m[3]), sig_tr_sd=float(m[4]),
                                     sig_te=float(m[5]), sig_te_sd=float(m[6]), noi_te=float(m[7]),
                                     noi_te_sd=float(m[8])))
    return pd.DataFrame(rows)


def run_seeds(ns, seeds, log=print):
    """-> long table: one row per seed x scale x model."""
    rows = []
    for s in seeds:
        t0 = time.time()
        r = ns['run_one_seed'](s)
        for name in MODELS:
            for i, sc in enumerate(ns['SCALES']):
                rows.append(dict(seed=s, scale=sc, model=name, sig_tr=r[name]['sig_tr'][i],
                                 sig_te=r[name]['sig_te'][i], noi_tr=r[name]['noi_tr'][i],
                                 noi_te=r[name]['noi_te'][i]))
        log(f'  seed {s} done in {time.time() - t0:.0f}s')
    return pd.DataFrame(rows)


def summary(df):
    """mean, std (population, as in the paper), and a 95% t-interval per scale x model."""
    def agg(x):
        n = len(x)
        h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        return pd.Series(dict(mean=x.mean(), median=x.median(), std=x.std(ddof=0), worst=x.min() if x.name == 'sig_te' else x.max(), ci_lo=x.mean() - h, ci_hi=x.mean() + h, n=n))
    out = []
    for (sc, mdl), g in df.groupby(['scale', 'model']):
        for met in ('sig_te', 'noi_te'):
            out.append(dict(scale=sc, model=mdl, metric=met, **agg(g[met])))
    return pd.DataFrame(out)


def win_table(df, a='VJEPA', others=('VAE', 'AR', 'JEPA', 'BJEPA')):
    """Per scale: in how many seeds does model `a` beat each other model?
    signal R2 (test): higher is better; distractor R2 (test): lower is better."""
    rows = []
    for sc, g in df.groupby('scale'):
        p = g.pivot(index='seed', columns='model', values=['sig_te', 'noi_te'])
        for o in others:
            rows.append(dict(scale=sc, vs=o, n=len(p),
                             signal_wins=int((p['sig_te'][a] > p['sig_te'][o]).sum()),
                             distractor_wins=int((p['noi_te'][a] < p['noi_te'][o]).sum()),
                             signal_p=_wilcoxon(p['sig_te'][a], p['sig_te'][o])))
    return pd.DataFrame(rows)


def _wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank p-value of the paired difference (NaN when undefined)."""
    try:
        return stats.wilcoxon(x, y).pvalue if len(x) > 5 and np.any(x != y) else np.nan
    except ValueError:
        return np.nan
