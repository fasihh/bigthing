"""Batches, masking, modality subsets, and the Noisy TV corruptions (A: test noise, B: noise channel,
C: swapped modality, D: Huang-style sticky distractor)."""
import itertools
import math

import torch

from .runtime import BASE, NAMES, RT


def ar1_distractor(B, L, gen=None):
    """Sticky distractor d_t = rho d_{t-1} + sqrt(1 - rho^2) v_t with unit stationary variance."""
    rho, k = RT.CFG['dist_rho'], RT.CFG['dist_k']
    v = torch.randn(B, L, k, device=RT.DEV, generator=gen)
    d = torch.empty_like(v)
    d[:, 0] = v[:, 0]
    for t in range(1, L):
        d[:, t] = rho * d[:, t - 1] + math.sqrt(1 - rho ** 2) * v[:, t]
    return d


def get_batch(split, idx, mods, gen=None):
    d = RT.DATA[split]
    b = {}
    for m in BASE:
        b[m] = d[m][idx].to(RT.DEV, non_blocking=True)
        b[m + 'm'] = d[m + 'm'][idx].to(RT.DEV, non_blocking=True)
    b['y'] = d['y'][idx].to(RT.DEV)
    env = RT.ENV
    if env is not None:                      # Noisy TV D: x <- x + sigma * (d @ D_m) * feature std, on real frames
        L = max(RT.LENS[m] for m in env['mods'])
        if env['shared']:
            dd = ar1_distractor(len(idx), L, gen)
            ds = {m: dd for m in env['mods']}
            b['dpool'] = dd.mean(1)
        else:
            ds = {m: ar1_distractor(len(idx), L, gen) for m in env['mods']}
            b['dpool'] = torch.cat([ds[m].mean(1) for m in env['mods']], -1)
        for m, dd in ds.items():
            b[m] = b[m] + env['sigma'] * (dd[:, :RT.LENS[m]] @ RT.DIST_MIX[m]) * RT.FEAT_STD[m] * \
                b[m + 'm'].unsqueeze(-1)
    if 'n' in mods:                          # Noisy TV B: a pure-noise modality, fresh every call
        b['n'] = torch.randn(len(idx), RT.LENS['n'], RT.DIMS['n'], device=RT.DEV, generator=gen)
        b['nm'] = torch.ones(len(idx), RT.LENS['n'], dtype=torch.bool, device=RT.DEV)
    return b


def batches(split, bs, shuffle, mods, gen=None):
    n = RT.N_SPLIT[split]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield j, get_batch(split, j, mods, gen)


def random_token_mask(mask, ratio, gen=None):
    keep = (torch.rand(mask.shape, device=mask.device, generator=gen) > ratio) & mask
    empty = keep.sum(1) == 0
    keep[empty] = mask[empty]
    return keep


def random_subsets(B, M):
    """One random non-empty subset of M modalities per example (uniform over the 2^M - 1 subsets)."""
    codes = torch.randint(1, 2 ** M, (B,), device=RT.DEV)
    return torch.stack([(codes >> i) & 1 for i in range(M)], 1).float()


def all_subsets(M):
    return [tuple((c >> i) & 1 for i in range(M)) for c in range(1, 2 ** M)]


def pooled(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


def cosine_lr(opt, base, step, total, warm=0.05):
    w = int(total * warm)
    lr = base * (step / max(1, w) if step < w else 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w))))
    for g in opt.param_groups:
        g['lr'] = lr


# ----------------------------------------------------------------------------- evaluation conditions
def _code(mods, sub):
    return tuple(1 if m in sub else 0 for m in mods)


def subset_conditions(mods):
    conds = []
    for r in (1, 2, 3):
        for sub in itertools.combinations(BASE, r):
            nm = ''.join(NAMES[m] for m in sub)
            conds.append(dict(kind='subset', name=nm, avail=_code(mods, sub)))
            if 'n' in mods:
                conds.append(dict(kind='subset+N', name=nm + '+N', avail=_code(mods, sub + ('n',))))
    return conds


def textdrop_conditions(mods, drops):
    return [dict(kind='textdrop', name=f'{base}~drop{int(dr * 100)}', text_drop=dr,
                 avail=_code(mods, ('t',)) if base == 'T' else _code(mods, BASE))
            for base in ('T', 'TAV') for dr in drops]


def noise_conditions(mods, lambdas):
    """Noisy TV A: x' = sqrt(1 - lam) x + sqrt(lam) eps on every combination of modalities, all still available."""
    full = _code(mods, BASE)
    out = []
    for r in (1, 2, 3):
        for sub in itertools.combinations(BASE, r):
            cor = ''.join(NAMES[m] for m in sub)
            out += [dict(kind='noise', name=f'noise[{cor}]@{lam}', avail=full, noise={m: lam for m in sub},
                         lam=lam, corrupted=cor) for lam in lambdas]
    return out


def swap_conditions(mods, subsets=(('a',), ('v',), ('a', 'v'))):
    """Noisy TV C: the listed modalities taken from another random clip (one fixed shuffle)."""
    full = _code(mods, BASE)
    return [dict(kind='swap', name=f'swap[{"".join(NAMES[m] for m in sub)}]', avail=full, swap=sub,
                 corrupted=''.join(NAMES[m] for m in sub)) for sub in subsets]


def conditions(mods, which, cfg):
    """which: any of 'subset', 'textdrop', 'noise', 'swap'. The clean TAV subset is always included
    (it is the reference for corruption-detection AUROC)."""
    conds = subset_conditions(mods) if 'subset' in which else \
        [c for c in subset_conditions(mods) if c['name'] == 'TAV']
    if 'textdrop' in which:
        conds += textdrop_conditions(mods, cfg['text_drops'])
    if 'noise' in which:
        conds += noise_conditions(mods, cfg['noise_lambdas'])
    if 'swap' in which:
        conds += swap_conditions(mods)
    return conds
