"""CMU-MOSEI (MMSA aligned_50.pkl) -> cleaned, pooled, standardised arrays, cached as mosei_compact.npz.

Preprocessing is identical to the first full-experiment run (see docs/walkthrough), so results stay comparable.
"""
import hashlib
import math
import os
import pickle
import re
import time

import numpy as np
import torch

from .runtime import BASE, RT, attach

KNOWN_SHA = {
    'ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f': 'MOSEI unaligned_50.pkl',
    '45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791': 'MOSEI aligned_50.pkl',
}
COMPACT = 'mosei_compact.npz'


# ----------------------------------------------------------------------------- download helpers
def check_download(path):
    """Fail loudly if Google Drive returned an HTML page instead of the pickle."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(2000)
    if size < 500e6 or head.lstrip().startswith(b'<'):
        txt = head.decode('utf-8', 'ignore').lower()
        os.remove(path)
        if 'quota' in txt or 'too many users' in txt:
            reason = 'Drive download quota exceeded: retry later or copy the file to your own Drive'
        elif 'sign in' in txt or 'access' in txt:
            reason = 'no access: share the FILE with "Anyone with the link"'
        else:
            reason = 'unknown (an HTML page was returned)'
        raise RuntimeError(f'Download is not the dataset ({size / 1e6:.2f} MB). {reason}. The bad file was deleted.')
    print(f'download OK: {size / 1e9:.2f} GB')


def gdrive_download(url, dest='./data/MOSEI/aligned_50.pkl'):
    if os.path.exists(dest):
        return dest
    if '/folders/' in url:
        raise ValueError('That is a folder link: share the file aligned_50.pkl itself.')
    m = re.search(r'/d/([\w-]{20,})', url) or re.search(r'[?&]id=([\w-]{20,})', url)
    fid = m.group(1) if m else url.strip()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.system('pip -q install -U gdown')
    os.system(f'gdown {fid} -O {dest}')
    check_download(dest)
    return dest


# ----------------------------------------------------------------------------- conversion
def find_file(names, roots=('/kaggle/input', '.', './data')):
    hits = []
    for root in roots:
        if os.path.isdir(root):
            for dp, _, fs in os.walk(root):
                hits += [os.path.join(dp, f) for f in fs if f in names]
    return hits


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _clean(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _pool(x, mask, max_len):
    """Masked average-pool along time so that length <= max_len."""
    N, L, D = x.shape
    if L <= max_len:
        return x, mask
    k = math.ceil(L / max_len)
    Lp = (L // k) * k
    xr = x[:, :Lp].reshape(N, L // k, k, D)
    mr = mask[:, :Lp].reshape(N, L // k, k).astype(np.float32)
    s = mr.sum(2, keepdims=True)
    return ((xr * mr[..., None]).sum(2) / np.maximum(s, 1)).astype(np.float32), (s[..., 0] > 0)


def build_arrays(path, max_len, text_pca):
    with open(path, 'rb') as f:
        D = pickle.load(f)
    out = {}
    for split in ['train', 'valid', 'test']:
        d = D[split]
        text = _clean(d['text'])
        if text.ndim == 2:
            text = text[:, None, :]
        audio, vision = _clean(d['audio']), _clean(d['vision'])
        y = np.asarray(d['regression_labels'], dtype=np.float32).reshape(-1)
        if 'text_bert' in d:
            tm = (np.asarray(d['text_bert'])[:, 1, :] > 0)[:, :text.shape[1]]      # BERT attention mask
        else:
            tm = np.abs(text).sum(-1) > 0
        am, vm = np.abs(audio).sum(-1) > 0, np.abs(vision).sum(-1) > 0
        audio, am = _pool(audio, am, max_len)
        vision, vm = _pool(vision, vm, max_len)
        for m in (tm, am, vm):
            m[m.sum(1) == 0, 0] = True                                   # never an all-padding row
        out[split] = dict(t=text, a=audio, v=vision, tm=tm, am=am, vm=vm, y=y)
        print(f'{split:5s} n={len(y):5d} text{text.shape} audio{audio.shape} vision{vision.shape}')
    del D
    for k, mk in (('a', 'am'), ('v', 'vm')):                             # standardise with train statistics
        tr = out['train'][k][out['train'][mk]]
        mu, sd = tr.mean(0), tr.std(0) + 1e-6
        for s in out:
            out[s][k] = np.clip((out[s][k] - mu) / sd, -10, 10) * out[s][mk][..., None]
    tr_tok = out['train']['t'][out['train']['tm']]                        # text: PCA on train tokens
    mu = tr_tok.mean(0)
    if text_pca and text_pca < tr_tok.shape[1]:
        rng = np.random.default_rng(0)
        sub = tr_tok[rng.choice(len(tr_tok), min(len(tr_tok), 200_000), replace=False)] - mu
        _, S, Vt = np.linalg.svd(sub, full_matrices=False)
        P = Vt[:text_pca].T
        kept = float(np.sum(S[:text_pca] ** 2) / np.sum(S ** 2))
        print(f'text PCA {tr_tok.shape[1]}->{text_pca}: keeps {kept:.1%} of variance')
    else:
        P, kept = np.eye(tr_tok.shape[1], dtype=np.float32), 1.0
    scale = ((tr_tok - mu) @ P).std() + 1e-6
    for s in out:
        out[s]['t'] = (((out[s]['t'] - mu) @ P) / scale).astype(np.float32) * out[s]['tm'][..., None]
    out['meta'] = dict(pca_var_kept=np.array(kept, dtype=np.float32))
    return out


def save_compact(arrs, path):
    flat = {}
    for s, d in arrs.items():
        for k, v in d.items():
            flat[f'{s}/{k}'] = v.astype(np.float16) if k in BASE else v
    np.savez(path, **flat)
    print(f'saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)')


def load_compact(path):
    z = np.load(path)
    arrs = {}
    for key in z.files:
        s, k = key.split('/')
        arrs.setdefault(s, {})[k] = z[key].astype(np.float32) if k in BASE else z[key]
    return arrs


def load_mosei(cfg, gdrive_url=''):
    """Find (or build) mosei_compact.npz, load it as tensors and attach it to the runtime."""
    t0 = time.time()
    os.makedirs(cfg['out_dir'], exist_ok=True)
    compact = None if cfg['pkl_path'] else next(iter(find_file({COMPACT}, ('/kaggle/input', '.', './data',
                                                                          cfg['out_dir']))), None)
    if compact:
        print('loading', compact)
        arrs = load_compact(compact)
    else:
        if gdrive_url:
            gdrive_download(gdrive_url)
        pkl = cfg['pkl_path'] or sorted(find_file({'aligned_50.pkl', 'unaligned_50.pkl'}),
                                        key=lambda p: ('unaligned' in p, p))[0]
        if not cfg['quick']:
            h = sha256(pkl)
            print('sha256', h[:16], KNOWN_SHA.get(h, 'unknown file (not an MMSA MOSEI release?)'))
        arrs = build_arrays(pkl, cfg['max_len'], cfg['text_pca'])
        save_compact(arrs, os.path.join(cfg['out_dir'], COMPACT))
    meta = arrs.pop('meta', {})
    if 'pca_var_kept' in meta:
        print(f'text PCA keeps {float(meta["pca_var_kept"]):.1%} of BERT variance')
    return attach_arrays(cfg, arrs, t0)


def attach_arrays(cfg, arrs, t0=None):
    data = {s: {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items() if k != 'ids'}
            for s, d in arrs.items() if s != 'meta'}
    attach(cfg, data)
    print({s: len(data[s]['y']) for s in data}, f'data ready on {RT.DEV}' +
          (f' in {time.time() - t0:.0f}s' if t0 else ''))
    return data


def load_npz(cfg, path):
    """Load any cache in the compact format (e.g. coco_compact.npz) and attach it to the runtime."""
    arrs = load_compact(path)
    meta = arrs.pop('meta', {})
    print('meta:', {k: float(v) for k, v in meta.items()})
    return attach_arrays(cfg, arrs, time.time())
