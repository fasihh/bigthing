"""Cache interface shared by the synthetic toy world and the real-MOSEI cache (notebook 13).

A cache is a dict:
  y[split]                         labels
  F[split][cond]                   pooled input statistics (n, D) - concatenated over modalities
  MU[split][cond], VAR[split][cond] (n, 7) label beliefs for availability patterns PATS
  meta[cond] = dict(kind, m, heldout)
Splits: train / valid carry clean + self-generated corruptions (noise[m], swap[m]); test carries everything.
"""
import numpy as np

MODS = ('t', 'a', 'v')
PATS = ('tav', 'av', 'tv', 'ta', 't', 'a', 'v')        # order used by notebook 13 (full, drop-one, singles)


def cond_meta(name):
    if name == 'clean':
        return dict(kind='clean', m='', heldout=False)
    kind, rest = name.split('[', 1)
    m = rest.split(']')[0]
    lam = rest.split('@')[1] if '@' in rest else None
    held = not ((kind == 'noise' and lam is None) or kind == 'swap')
    return dict(kind=kind + ('@' + lam if lam else ''), m=m, heldout=held)


def load_real(path, tag):
    z = np.load(path)
    out = dict(y={}, F={}, MU={}, VAR={}, meta={})
    pats = list(z['patterns'])
    order = [pats.index(p) for p in PATS]
    for k in z.files:
        parts = k.split('/')
        if len(parts) == 2 and parts[1] == 'y':
            out['y'][parts[0]] = z[k].astype(np.float64)
        elif len(parts) == 3 and parts[2] == 'F':
            out['F'].setdefault(parts[0], {})[parts[1]] = z[k].astype(np.float32)
        elif len(parts) == 4 and parts[2] == tag:
            d = out['MU'] if parts[3] == 'mu' else out['VAR']
            d.setdefault(parts[0], {})[parts[1]] = z[k].astype(np.float64)[:, order]
    out['fdims'] = list(z['fdims'])
    for c in out['MU']['test']:
        out['meta'][c] = cond_meta(c)
    return out
