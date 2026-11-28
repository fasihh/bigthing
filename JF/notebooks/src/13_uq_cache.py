# %% [markdown]
# # 13 · Real-MOSEI cache for fast uncertainty experiments (CPU, no training)
#
# Dumps everything the small experiments need, so each one runs in seconds instead of re-running the experts:
# for three trained models (frozen V-JEPA PoE, frozen variational-target PoE, fine-tuned V-JEPA PoE) and every
# test condition, the label belief of **every availability pattern** (all three modalities, each pair, each single
# modality), plus the pooled input statistics (model-independent).
#
# * train / valid: clean + self-generated corruptions (white noise λ = 1 and random swap on each modality), for
#   techniques that learn from synthetic corruptions;
# * test: clean, the same corruptions, and held-out ones (noise λ = 0.5, half-clip noise, gain fault, dead sensor,
#   time shuffle, nearest-neighbour swap, two noisy modalities).
#
# Attach the outputs of 01, 03 and 07. Output: `uq_cache.npz` (float16).

# %% [include] _setup.py

# %%
import time
import numpy as np, torch
from fusionlab import BASE, config, data, gate, pretrain, runner
from fusionlab.runtime import RT

cfg = config.load_config(os.path.join(CONFIGS, '12_mosei_gate.yaml'))
data.load_mosei(cfg)
OUT = cfg['out_dir']
MODELS = [('vjepa', 'frozen'), ('vjepa_vt', 'frozen'), ('vjepa', 'finetuned')]
SEED = 0

tr_mask = {m: RT.DATA['train'][m + 'm'] for m in BASE}
feat_mean = {m: RT.DATA['train'][m][tr_mask[m]].mean(0).to(RT.DEV) for m in BASE}
self_gen = [dict(name='clean', kind='clean')] + \
    [dict(name=f'noise[{m}]', kind='noise', m=m, lam=1.0) for m in BASE] + \
    [dict(name=f'swap[{m}]', kind='swap', m=m) for m in BASE]
test_conds = [c for c in gate.gate_conditions(include_heldout=True)]

# %%
store = {}
perms = {s: dict(swap=torch.randperm(RT.N_SPLIT[s], generator=torch.Generator().manual_seed(123 + 7))) for s in RT.N_SPLIT}
for obj, rg in MODELS:
    if not cfg['quick'] and not pretrain.find_ckpt(f'v2_model_main_{obj}_{rg}_poe_s{SEED}.pt'):
        print('skip', obj, rg); continue
    t0 = time.time()
    experts, _, marg = pretrain.pretrain_or_load(obj, BASE, 'main', SEED, cfg)
    model, _ = runner.build_model(experts, obj, rg, 'poe', BASE, cfg, SEED, marg, 'main')
    tag = f'{obj}_{rg}'
    for split, conds in (('train', self_gen), ('valid', self_gen), ('test', test_conds)):
        clean = None
        for cd in conds:
            if cd['kind'] == 'swapnn' and 'swapnn' not in perms[split]:
                perms[split]['swapnn'] = {m: gate.nn_perm(clean['F'], m) for m in BASE}
            r = gate.run_condition(model, split, cd, perms[split], feat_mean, singles=True)
            if cd['kind'] == 'clean':
                clean = r
            key = f'{split}/{cd["name"]}'
            pats = list(r['pred'])
            store[f'{key}/{tag}/mu'] = np.stack([r['pred'][p][0] for p in pats], 1).astype(np.float32)
            store[f'{key}/{tag}/var'] = np.stack([r['pred'][p][1] for p in pats], 1).astype(np.float32)
            if f'{key}/F' not in store:
                store[f'{key}/F'] = np.concatenate([r['F'][m] for m in BASE], 1).astype(np.float16)
            store[f'{split}/y'] = r['y'].astype(np.float32)
    store[f'patterns'] = np.array([''.join(m for m, a in zip(BASE, p) if a) for p in pats])
    print(tag, f'{time.time() - t0:.0f}s', flush=True)

store['fdims'] = np.array([RT.DIMS[m] * 2 for m in BASE])
np.savez_compressed(os.path.join(OUT, 'uq_cache.npz'), **store)
print('saved', os.path.getsize(os.path.join(OUT, 'uq_cache.npz')) / 1e6, 'MB,', len(store), 'arrays')
