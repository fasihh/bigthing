"""Synthetic toy cache with the same layout as the real-MOSEI cache, made MOSEI-like:
text strong, audio and video weak (so corrupting them barely hurts, as on MOSEI), per-clip quality,
temporal structure, a shared speaker factor. Experts are per-modality Gaussian label predictors; availability
patterns are prior-corrected products of the available experts (what a PoE model does with missing inputs)."""
import numpy as np

from bench import data as D
from bench.experts import Expert, pool
from bench.techniques import poe
from .cache import MODS, PATS, cond_meta

D.STRENGTH.update({'t': 1.0, 'a': 0.35, 'v': 0.25})


def _corrupt(X, name, st, rng, spk):
    meta = cond_meta(name)
    if meta['kind'] == 'clean':
        return X
    kind, m = meta['kind'], meta['m']
    if kind.startswith('noise@'):
        return D.corrupt(X, ('white', m, float(kind.split('@')[1])), st, rng)[0]
    if kind == 'noise2':
        return D.corrupt(X, ('double', m, 1.0), st, rng)[0]
    mapping = dict(noise='white', swap='swap', halfnoise='halfnoise', gain='gain', dead='dead', swapnn='swap_spk',
                   shuffle='shuffle')
    return D.corrupt(X, (mapping[kind], m, 1.0), st, rng, spk=spk)[0]


SELF_GEN = ['clean'] + [f'noise[{m}]' for m in MODS] + [f'swap[{m}]' for m in MODS]
TEST = SELF_GEN + [f'noise[{m}]@0.5' for m in MODS] + [f'{k}[{m}]' for k in ('halfnoise', 'gain', 'dead', 'swapnn')
                                                         for m in MODS] + [f'noise2[{p}]' for p in ('ta', 'tv', 'av')]


def build(seed=0, n=(6000, 1500, 3000)):
    world = D.World(seed)
    rng = np.random.default_rng(seed)
    Xs, ys, spk = {}, {}, {}
    for split, k in zip(('train', 'valid', 'test'), n):
        Xs[split], ys[split], _ = world.sample(k, rng)
        spk[split] = world.last_spk
    st = D.stats(Xs['train'])
    half = n[0] // 2                                     # experts on the first half, self-gen data on the second
    E = {m: Expert('plain', seed=seed).fit(Xs['train'][m][:half], ys['train'][:half],
                                           Xs['valid'][m], ys['valid']) for m in MODS}
    prior = (float(ys['train'].mean()), float(ys['train'].var()))
    out = dict(y={}, F={}, MU={}, VAR={}, meta={}, fdims=[2 * D.DIMS[m] for m in MODS])
    for split in ('train', 'valid', 'test'):
        X0 = {m: v[half:] for m, v in Xs['train'].items()} if split == 'train' else Xs[split]
        y = ys['train'][half:] if split == 'train' else ys[split]
        sp = spk['train'][half:] if split == 'train' else spk[split]
        out['y'][split] = y.astype(np.float64)
        for k in ('F', 'MU', 'VAR'):
            out[k][split] = {}
        for i, c in enumerate(SELF_GEN if split != 'test' else TEST):
            r = np.random.default_rng(seed * 1000 + i + {'train': 1, 'valid': 2, 'test': 3}[split] * 100)
            X = _corrupt(X0, c, st, r, sp)
            B = {m: E[m].predict(X[m])[:2] for m in MODS}
            mus, vs = [], []
            for p in PATS:
                mu, v = poe({m: B[m] for m in p}, prior)
                mus.append(mu); vs.append(v)
            out['F'][split][c] = np.concatenate([pool(X[m]) for m in MODS], 1)
            out['MU'][split][c] = np.stack(mus, 1)
            out['VAR'][split][c] = np.stack(vs, 1)
            if split == 'test':
                out['meta'][c] = cond_meta(c)
    return out
