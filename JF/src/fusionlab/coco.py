"""MS-COCO 2017 as a second multimodal benchmark, in the same array format as CMU-MOSEI.

Three "sensors" describe each scene (so a lying sensor can be localised, which needs M >= 3):
  v : the image        DINOv2-small patch tokens (7x7 pooled grid + CLS = 50 tokens x 384)
  t : caption #1       all-MiniLM-L6-v2 token states (<= 32 tokens x 384), annotator 1
  a : caption #2       same text encoder, a different annotator's caption
The image and text encoders are trained separately (not CLIP), so the modalities are NOT pre-aligned.
Task (regression, label-free pretraining is unaffected): y = standardised log2(1 + number of annotated object
instances in the image), from instances_*2017.json. Every modality carries evidence about it ("two dogs and a man").
Splits: train = N_TRAIN images of train2017, valid = N_VALID other train2017 images, test = all val2017 images (5,000).
Every feature is PCA-reduced to 128 dims (fitted on train) to keep the cache small.
"""
import glob
import json
import os
import time
from collections import Counter

import numpy as np
import torch

from .data import save_compact

IMG_MODEL = 'facebook/dinov2-small'
TXT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
COMPACT = 'coco_compact.npz'


def find_coco(roots=('/kaggle/input', '.', './data')):
    """Locate the directory holding annotations/captions_train2017.json and the train2017 / val2017 image folders."""
    for r in roots:
        for p in glob.glob(os.path.join(r, '**', 'captions_train2017.json'), recursive=True):
            ann = os.path.dirname(p)
            root = os.path.dirname(ann)
            for cand in (root, os.path.dirname(root)):
                tr = glob.glob(os.path.join(cand, '**', 'train2017'), recursive=True)
                va = glob.glob(os.path.join(cand, '**', 'val2017'), recursive=True)
                tr = [d for d in tr if os.path.isdir(d)]
                va = [d for d in va if os.path.isdir(d)]
                if tr and va:
                    return ann, tr[0], va[0]
    raise FileNotFoundError('COCO 2017 not found: attach the Kaggle dataset "awsaf49/coco-2017-dataset".')


def _load_split(ann_dir, split):
    caps = json.load(open(os.path.join(ann_dir, f'captions_{split}2017.json')))
    inst = json.load(open(os.path.join(ann_dir, f'instances_{split}2017.json')))
    files = {im['id']: im['file_name'] for im in caps['images']}
    by_img = {}
    for c in sorted(caps['annotations'], key=lambda c: c['id']):
        by_img.setdefault(c['image_id'], []).append(c['caption'].strip())
    count = Counter(a['image_id'] for a in inst['annotations'] if not a.get('iscrowd', 0))
    ids = sorted(i for i in files if len(by_img.get(i, [])) >= 2)
    return [dict(id=i, file=files[i], cap1=by_img[i][0], cap2=by_img[i][1], n=count.get(i, 0)) for i in ids]


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths):
        from torchvision import transforms as T
        self.paths = paths
        self.tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])     # DINOv2 preprocessing

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert('RGB'))


@torch.no_grad()
def _image_tokens(paths, dev, bs=128, workers=4):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(IMG_MODEL).to(dev).eval().half()
    dl = torch.utils.data.DataLoader(_Images(paths), batch_size=bs, num_workers=workers, shuffle=False)
    out = np.zeros((len(paths), 50, 384), np.float32)
    t0, i = time.time(), 0
    for x in dl:
        h = model(pixel_values=x.to(dev).half()).last_hidden_state.float()     # (B, 1 + 16*16, 384)
        cls, patches = h[:, :1], h[:, 1:].reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
        grid = torch.nn.functional.adaptive_avg_pool2d(patches, 7).flatten(2).transpose(1, 2)   # (B, 49, 384)
        out[i:i + len(x)] = torch.cat([cls, grid], 1).cpu().numpy()
        i += len(x)
        if (i // bs) % 40 == 0:
            print(f'  images {i}/{len(paths)} ({time.time() - t0:.0f}s)')
    return out


@torch.no_grad()
def _text_tokens(texts, dev, max_len=32, bs=256):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TXT_MODEL)
    model = AutoModel.from_pretrained(TXT_MODEL).to(dev).eval()
    out = np.zeros((len(texts), max_len, 384), np.float32)
    mask = np.zeros((len(texts), max_len), bool)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        h = model(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state
        out[i:i + bs] = h.cpu().numpy()
        mask[i:i + bs] = enc['attention_mask'].numpy().astype(bool)
    return out, mask


def build_coco(out_path, n_train=20000, n_valid=2500, pca=128, seed=0):
    ann, tr_dir, va_dir = find_coco()
    print('COCO annotations:', ann, '| images:', tr_dir, va_dir)
    train_all, test = _load_split(ann, 'train'), _load_split(ann, 'val')
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(train_all))
    splits = {'train': [train_all[i] for i in pick[:n_train]],
              'valid': [train_all[i] for i in pick[n_train:n_train + n_valid]], 'test': test}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw = {}
    for s, items in splits.items():
        d = tr_dir if s != 'test' else va_dir
        print(f'{s}: {len(items)} images')
        v = _image_tokens([os.path.join(d, it['file']) for it in items], dev)
        t, tm = _text_tokens([it['cap1'] for it in items], dev)
        a, am = _text_tokens([it['cap2'] for it in items], dev)
        y = np.log2(1 + np.array([it['n'] for it in items], np.float32))
        raw[s] = dict(t=t, a=a, v=v, tm=tm, am=am, vm=np.ones(v.shape[:2], bool), y=y, ids=np.array([it['id'] for it in items]))
    mu_y, sd_y = raw['train']['y'].mean(), raw['train']['y'].std()
    meta = {}
    for m in ('t', 'a', 'v'):                                      # PCA per modality on real train tokens
        X = raw['train'][m][raw['train'][m + 'm']]
        mu = X.mean(0)
        sub = X[rng.choice(len(X), min(len(X), 200_000), replace=False)] - mu
        _, S, Vt = np.linalg.svd(sub, full_matrices=False)
        P = Vt[:pca].T
        scale = ((X - mu) @ P).std() + 1e-6
        meta[f'pca_var_{m}'] = np.array(float((S[:pca] ** 2).sum() / (S ** 2).sum()), np.float32)
        for s in raw:
            raw[s][m] = (((raw[s][m] - mu) @ P) / scale).astype(np.float32) * raw[s][m + 'm'][..., None]
    for s in raw:
        raw[s]['y'] = ((raw[s]['y'] - mu_y) / sd_y).astype(np.float32)
    meta.update(y_mean=np.array(mu_y, np.float32), y_std=np.array(sd_y, np.float32))
    raw['meta'] = meta
    print({k: float(v) for k, v in meta.items()})
    save_compact(raw, out_path)
    return out_path
