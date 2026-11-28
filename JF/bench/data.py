"""Synthetic multimodal clips for the short noisy-TV screen.

Design goals (so a win here means something):
  * three modalities with different strength (text > audio > video), like CMU-MOSEI;
  * per-clip, per-modality quality q_m, so a good expert *should* vary its sigma on clean data;
  * temporal structure (AR(1) latents), so pooled statistics behave like real features;
  * a shared nuisance factor (speaker) that moves all modalities but not the label;
  * corruptions: white noise, structured AR noise with matched spectrum (held out from every
    technique), swap with another clip (real data, wrong clip), partial noise (lambda = 0.5),
    missing subsets, and an extra pure-noise channel present at train and test (classic noisy TV).
"""
import numpy as np

MODS = ('t', 'a', 'v')
DIMS = {'t': 32, 'a': 24, 'v': 16}
STRENGTH = {'t': 1.0, 'a': 0.6, 'v': 0.45}
T = 20
K = 6           # latent dims per modality
RHO = 0.85


def _ar(rng, n, k, rho=RHO, T=T):
    h = np.zeros((n, T, k))
    h[:, 0] = rng.normal(size=(n, k))
    s = np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        h[:, t] = rho * h[:, t - 1] + s * rng.normal(size=(n, k))
    return h


class World:
    """Fixed generative parameters (one per seed)."""

    def __init__(self, seed, noise_channel=False):
        rng = np.random.default_rng(1000 + seed)
        self.W = {m: rng.normal(size=(DIMS[m], K)) / np.sqrt(K) for m in MODS}
        self.B = {m: rng.normal(size=(DIMS[m], 2)) / np.sqrt(2) for m in MODS}     # speaker loading
        self.offset = {m: rng.normal(size=DIMS[m]) for m in MODS}
        self.scale = {m: np.exp(rng.normal(scale=0.5, size=DIMS[m])) for m in MODS}
        self.noise_channel = noise_channel

    def sample(self, n, rng):
        s = rng.normal(size=n)
        y = (s + 0.5 * rng.normal(size=n)).astype(np.float32)
        spk = rng.normal(size=(n, 2))
        X, Q = {}, {}
        for m in MODS:
            q = rng.uniform(0.15, 1.0, size=n)            # clip quality for this modality
            h = _ar(rng, n, K)
            # signal drives latent 0 and 1 over the whole clip, scaled by quality and strength
            sig = (STRENGTH[m] * q * s)[:, None]
            h[:, :, 0] += 1.5 * sig
            h[:, :, 1] += 0.8 * sig * np.sin(np.linspace(0, np.pi, T))[None]
            x = h @ self.W[m].T + (spk @ self.B[m].T)[:, None, :] + 0.3 * rng.normal(size=(n, T, DIMS[m]))
            X[m] = ((x + self.offset[m]) * self.scale[m]).astype(np.float32)
            Q[m] = q
        if self.noise_channel:
            X['n'] = rng.normal(size=(n, T, 16)).astype(np.float32) * 2.0
        self.last_spk = spk
        return X, y, Q


def stats(X):
    """Per-feature mean/std over (clips, time) for each modality: used to match noise moments."""
    return {m: (X[m].mean(axis=(0, 1)), X[m].std(axis=(0, 1)) + 1e-6) for m in X}


def white_noise(x, st, rng):
    mu, sd = st
    return (mu + sd * rng.normal(size=x.shape)).astype(np.float32)


def ar_noise(x, st, rng, rho=RHO):
    """Held-out corruption: Gaussian noise with the same per-feature moments AND temporal correlation."""
    mu, sd = st
    n, T_, d = x.shape
    e = _ar(rng, n, d, rho=rho, T=T_)
    return (mu + sd * e).astype(np.float32)


def swap(x, rng):
    perm = rng.permutation(len(x))
    # make sure nobody keeps their own clip
    same = perm == np.arange(len(x))
    perm[same] = (perm[same] + 1) % len(x)
    return x[perm]


def same_speaker_perm(spk, rng):
    """Hardest swap: each clip gets the clip of its nearest-speaker neighbour (same 'person')."""
    d = ((spk[:, None, :] - spk[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return d.argmin(1)


def corrupt(X, cond, st, rng, spk=None):
    """cond: ('clean',) | ('missing', frozenset) | (kind, m, lam) with kind in white/ar/swap."""
    X = {k: v.copy() for k, v in X.items()}
    mask = {m: True for m in X}
    if cond[0] == 'missing':
        for m in X:
            if m in MODS and m not in cond[1]:
                mask[m] = False
        return X, mask
    if cond[0] == 'clean':
        return X, mask
    kind, m, lam = cond
    mu_m, sd_m = st[m] if m in st else (0, 1)
    if kind == 'double':                       # white noise on two modalities at once (m = 'ta' etc.)
        for mm in m:
            X[mm] = white_noise(X[mm], st[mm], rng)
        return X, mask
    if kind == 'gain':                         # sensor gain fault: deviations x3
        X[m] = (mu_m + 3.0 * (X[m] - mu_m)).astype(np.float32); return X, mask
    if kind == 'dead':                         # dead sensor: flat signal at the mean
        X[m] = np.broadcast_to(mu_m, X[m].shape).astype(np.float32).copy(); return X, mask
    if kind == 'shuffle':                      # time order scrambled within each clip
        idx = np.argsort(rng.random(X[m].shape[:2]), 1)
        X[m] = np.take_along_axis(X[m], idx[:, :, None], 1); return X, mask
    if kind == 'halfnoise':                    # second half of the clip replaced by white noise
        t2 = X[m].shape[1] // 2
        X[m][:, t2:] = white_noise(X[m][:, t2:], st[m], rng); return X, mask
    if kind == 'swap_spk':                     # swapped with a clip of the most similar speaker
        X[m] = X[m][same_speaker_perm(spk, rng)]; return X, mask
    if kind == 'white':
        bad = white_noise(X[m], st[m], rng)
    elif kind == 'ar':
        bad = ar_noise(X[m], st[m], rng)
    elif kind == 'swap':
        bad = swap(X[m], rng)
    else:
        raise ValueError(kind)
    mu = st[m][0]
    X[m] = (mu + np.sqrt(1 - lam) * (X[m] - mu) + np.sqrt(lam) * (bad - mu)).astype(np.float32) if lam < 1 else bad
    return X, mask


def conditions():
    out = [('clean',)]
    for sub in (('t', 'a'), ('t', 'v'), ('a', 'v'), ('t',), ('a',), ('v',)):
        out.append(('missing', frozenset(sub)))
    for kind in ('white', 'ar', 'swap'):
        for m in MODS:
            out.append((kind, m, 1.0))
    for m in MODS:
        out.append(('white', m, 0.5))
    return out


HELDOUT_KINDS = ('gain', 'dead', 'shuffle', 'halfnoise', 'swap_spk')


def heldout_conditions():
    out = [('clean',)]
    for kind in HELDOUT_KINDS:
        for m in MODS:
            out.append((kind, m, 1.0))
    for pair in ('ta', 'tv', 'av'):
        out.append(('double', pair, 1.0))
    return out


def cond_name(c):
    if c[0] == 'clean':
        return 'clean'
    if c[0] == 'missing':
        return 'miss[' + ''.join(m for m in MODS if m in c[1]) + ']'
    return f'{c[0]}[{c[1]}]' + ('' if c[2] == 1.0 else f'@{c[2]}')
