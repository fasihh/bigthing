"""Shared runtime state: device, config, loaded data and derived constants.

Every module reads data through `RT` so that a notebook only has to call `fusionlab.data.load_mosei(cfg)` once.
"""
import math
import os
import random

import numpy as np
import torch

BASE = ['t', 'a', 'v']                       # text, audio, video
NAMES = {'t': 'T', 'a': 'A', 'v': 'V', 'n': 'N'}


class _Runtime:
    def __init__(self):
        self.DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CFG = None
        self.DATA = None          # {split: {key: tensor}}
        self.DIMS = {}
        self.LENS = {}
        self.N_SPLIT = {}
        self.FEAT_STD = {}
        self.DIST_MIX = {}
        self.ENV = None           # Noisy TV D distractor settings while a distractor grid runs

    @property
    def ready(self):
        return self.DATA is not None


RT = _Runtime()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def attach(cfg, data):
    """Register config + data and compute the constants every module needs."""
    RT.CFG, RT.DATA = cfg, data
    RT.DIMS = {m: data['train'][m].shape[-1] for m in BASE}
    RT.LENS = {m: data['train'][m].shape[1] for m in BASE}
    RT.DIMS['n'], RT.LENS['n'] = cfg['noise_dim'], cfg['max_len']       # Noisy TV B noise channel
    RT.N_SPLIT = {s: len(data[s]['y']) for s in data}
    # per-dimension std over real (non-padding) training frames: test-time noise is matched to it
    RT.FEAT_STD = {m: data['train'][m][data['train'][m + 'm']].std(0).to(RT.DEV) for m in BASE}
    g = torch.Generator().manual_seed(1234)
    RT.DIST_MIX = {m: (torch.randn(cfg['dist_k'], RT.DIMS[m], generator=g) / math.sqrt(cfg['dist_k'])).to(RT.DEV)
                   for m in BASE}                     # fixed random mixing matrix per modality (Noisy TV D)
    os.makedirs(cfg['out_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['out_dir'], 'ckpt'), exist_ok=True)
