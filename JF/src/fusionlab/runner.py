"""Experiment grid runner: resumable, checkpointed, shared across notebooks.

Expert checkpoints (stage 1) and trained models (stage 2) are looked up in this run's ckpt folder and in every
attached Kaggle input, so a model trained in one notebook is reused, not retrained, by the next.
"""
import copy
import glob
import json
import os
import shutil
import time

import pandas as pd
import torch

from .evaluate import evaluate_model, probe_metrics
from .fusion import make_fusion
from .models import Head, make_experts
from .pretrain import HAS_TARGET, find_ckpt, pretrain_or_load
from .runtime import RT, set_seed
from .stage2 import stage2, train_d0


def paths():
    out = RT.CFG['out_dir']
    return {k: os.path.join(out, f) for k, f in (('results', 'results.csv'), ('curves', 'curves.jsonl'),
                                                 ('diag', 'diagnostics.csv'))}


def done_keys():
    p = paths()['results']
    return set(pd.read_csv(p, usecols=['key'])['key'].unique()) if os.path.exists(p) else set()


def append_rows(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path, nrows=0).columns.tolist()
        if set(df.columns) - set(old):
            pd.concat([pd.read_csv(path), df], ignore_index=True).to_csv(path, index=False)
            return
        df.reindex(columns=old).to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)


def resume_from(src_root):
    """Copy results and checkpoints from an earlier (unfinished) version of the same notebook."""
    if not src_root:
        return
    hits = glob.glob(os.path.join(src_root, '**', 'results.csv'), recursive=True)
    src = os.path.dirname(hits[0]) if hits else src_root
    for f in ('results.csv', 'curves.jsonl', 'diagnostics.csv'):
        s, d = os.path.join(src, f), os.path.join(RT.CFG['out_dir'], f)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy(s, d)
    for f in glob.glob(os.path.join(src, 'ckpt', '*.pt')):
        d = os.path.join(RT.CFG['out_dir'], 'ckpt', os.path.basename(f))
        if not os.path.exists(d):
            shutil.copy(f, d)
    print('resumed:', len(done_keys()), 'finished models found')


def build_model(experts0, obj, rg, fk, mods, c, seed, marginal, pgroup):
    """Stage 2 for one (objective, regime, fusion): load a saved model if one exists, else train it."""
    name = f'model_{pgroup}_{obj}_{rg}_{fk}_s{seed}.pt'
    if obj == 'none':
        set_seed(seed)
        experts = make_experts(mods, c, 'none').to(RT.DEV)
    else:
        experts = copy.deepcopy(experts0)
    path = find_ckpt(name)
    if path:
        ck = torch.load(path, map_location=RT.DEV)
        fusion = make_fusion(fk, c['d_lat'], len(mods), marginal).to(RT.DEV)
        head, d0 = Head(c['d_lat']).to(RT.DEV), torch.nn.Linear(c['d_lat'], 1).to(RT.DEV)
        for mod, k in ((experts, 'experts'), (fusion, 'fusion'), (head, 'head'), (d0, 'd0')):
            mod.load_state_dict(ck[k])
        info = ck['info']
        print(f'    loaded {path}')
    else:
        fusion, head, info = stage2(experts, fk, mods, c, seed, frozen=(rg == 'frozen'), marginal=marginal)
        d0 = train_d0(experts, fusion, head, mods, c, seed)
        torch.save({'experts': experts.state_dict(), 'fusion': fusion.state_dict(), 'head': head.state_dict(),
                    'd0': d0.state_dict(), 'info': info}, os.path.join(c['out_dir'], 'ckpt', name))
    return dict(experts=experts.eval(), fusion=fusion.eval(), head=head.eval(), d0=d0.eval(), mods=mods), info


def run_grid(name, mods, objectives, conds, frozen=(), finetuned=(), supervised=(), env=None, use_me=False,
             pgroup='main', probes=False, seeds=None):
    """name: label of this grid inside the notebook; pgroup: which pretrained experts to use ('main' is shared
    by notebooks 03, 04, 06 and 07)."""
    c = RT.CFG
    RT.ENV = env
    try:
        done, t_start, diag = done_keys(), time.time(), []
        for seed in (seeds or c['seeds']):
            for obj in objectives:
                plan = [('supervised', f) for f in supervised] if obj == 'none' else \
                       [('frozen', f) for f in frozen] + [('finetuned', f) for f in finetuned]
                plan = [(rg, f) for rg, f in plan if not (f == 'mpoe' and obj not in HAS_TARGET)]
                plan = [(rg, f) for rg, f in plan if f'{name}|{obj}|{rg}|{f}|{seed}' not in done]
                if not plan:
                    continue
                print(f'=== [{name}] objective={obj} seed={seed}: {len(plan)} model(s)')
                experts0, target, marginal = (None, None, None) if obj == 'none' else \
                    pretrain_or_load(obj, mods, pgroup, seed, c, diag)
                append_rows(paths()['diag'], diag); diag.clear()
                for rg, fk in plan:
                    key = f'{name}|{obj}|{rg}|{fk}|{seed}'
                    t0 = time.time()
                    model, info = build_model(experts0, obj, rg, fk, mods, c, seed, marginal, pgroup)
                    model['target'] = target if rg == 'frozen' else None
                    tag = dict(key=key, grid=name, objective=obj, regime=rg, fusion=fk, seed=seed, **info,
                               dist_pattern=env['pattern'] if env else 'none',
                               dist_sigma=env['sigma'] if env else 0.0)
                    if probes:
                        tag.update(probe_metrics(model))
                    rows, curves = evaluate_model(model, tag, conds, use_me)
                    append_rows(paths()['results'], rows)
                    with open(paths()['curves'], 'a') as f:
                        for cv in curves:
                            f.write(json.dumps(cv, default=float) + '\n')
                    print(f'    {key}: {(time.time() - t0) / 60:.1f} min (total {(time.time() - t_start) / 60:.0f} min)')
    finally:
        RT.ENV = None
