# %% [markdown]
# # 10 · Second dataset: MS-COCO 2017 — Cross-Examined Fusion vs other fusion methods
#
# **Why COCO:** a second, very different multimodal benchmark (natural images + human captions instead of video
# interviews), to check that the findings and the FYP-2 method are not specific to CMU-MOSEI.
#
# **Three sensors per scene** (three are needed so that a lying sensor can be *localised*):
#
# | Name in tables | Modality | Encoder (frozen, run once) |
# | --- | --- | --- |
# | **V** | the image | DINOv2-small patch tokens (7×7 grid + CLS = 50 × 384) |
# | **T** | caption written by annotator 1 | all-MiniLM-L6-v2 token states (≤ 32 × 384) |
# | **A** | caption written by annotator 2 | same text encoder |
#
# Image and text encoders are trained separately (not CLIP), so the senses are **not** pre-aligned.
# **Task:** regress how many objects are in the scene (standardised log2(1 + instances), from COCO's instance
# annotations). **Splits:** 20,000 train / 2,500 validation images from train2017; test = all 5,000 val2017 images.
#
# **Compared:** PoE, MoE, attention, MoNIG (evidential), robust PoE, disagreement-aware PoE — and **CXF**, which
# replaces each sensor's self-reported precision with a label-free posterior over which sensors are lying, fitted
# only on clean, unlabelled data, then averages the subset predictions (corruption treated as inferred missingness).
#
# **Test conditions:** the 7 sensor subsets (missing), noise on one sensor (λ = 0.5, 1), and a sensor swapped in
# from another image (a mismatched caption is a realistic failure).
#
# Attach: **awsaf49/coco-2017-dataset**. Turn on Internet (encoder weights) and a GPU. First run builds
# `coco_compact.npz` (~20 min); save it as a dataset to skip that next time. Resumable.

# %% [include] _setup.py

# %%
import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fusionlab import BASE, batching, coco, config, cxf, data, pretrain, report, runner

RESUME_FROM = ''
cfg = config.load_config(os.path.join(CONFIGS, '10_coco_cxf.yaml'))
os.makedirs(cfg['out_dir'], exist_ok=True)
hits = glob.glob('/kaggle/input/**/' + coco.COMPACT, recursive=True) + glob.glob(os.path.join(cfg['out_dir'], coco.COMPACT))
path = hits[0] if hits else coco.build_coco(os.path.join(cfg['out_dir'], coco.COMPACT), cfg['n_train'], cfg['n_valid'])
data.load_npz(cfg, path)
runner.resume_from(RESUME_FROM)
NAME = {'t': 'caption 1', 'a': 'caption 2', 'v': 'image'}

conds = (batching.conditions(BASE, ['subset', 'noise'], cfg)
         + batching.swap_conditions(BASE, (('t',), ('a',), ('v',))))
conds = [c for c in conds if c['kind'] != 'noise' or len(c['noise']) == 1]      # one corrupted sensor at a time
print(len(conds), 'conditions:', [c['name'] for c in conds])

# %% [markdown]
# ## 1 · Comparison fusions (all trained on random sensor subsets)

# %%
runner.run_grid('coco', BASE, cfg['objectives'], conds, frozen=cfg['frozen'], finetuned=cfg['finetuned'],
                supervised=cfg['supervised'], pgroup='coco')

# %% [markdown]
# ## 2 · Cross-Examined Fusion on the same PoE workers
# The auditor is a frozen stage-1 model (it never sees labels or corruptions). For fine-tuned and supervised
# workers the auditor is the frozen copy of the stage-1 experts (V-JEPA for the supervised worker).

# %%
done = runner.done_keys()
for seed in cfg['seeds']:
    for aud in cfg['auditors']:
        auditor, _, marg = pretrain.pretrain_or_load(aud, BASE, 'coco', seed, cfg)
        witness = cxf.Witness().fit(auditor, BASE)
        workers = [(aud, 'frozen'), (aud, 'finetuned')] + ([('none', 'supervised')] if aud == 'vjepa' else [])
        for obj, rg in workers:
            key = f'coco|{obj}|{rg}|cxfv2[{aud}]|{seed}'
            if key in done:
                continue
            model, info = runner.build_model(auditor, obj, rg, 'poe', BASE, cfg, seed, marg, 'coco')
            rows = cxf.evaluate_cxf(model, auditor, witness, conds,
                                    dict(key=key, grid='coco', objective=obj, regime=rg, seed=seed, auditor=aud, **info))
            for r in rows:
                r['fusion'] = f"{r['fusion']}[{aud}]"
            runner.append_rows(runner.paths()['results'], rows)
            print(key, 'T =', round(witness.T, 2))

# %% [markdown]
# # Results

# %%
OUT = cfg['out_dir']
R = pd.read_csv(runner.paths()['results'])
R['sensor'] = R['corrupted'].fillna('').map({'T': 'caption 1', 'A': 'caption 2', 'V': 'image', '': ''})
D2 = R[(R.decoder == 'D2') & (R.unc == 'base')]
SUB = D2[D2.cond.isin(report.SUBSETS)]
ONE = D2[(D2.kind.isin(['noise', 'swap'])) & ((D2.lam == 1.0) | (D2.kind == 'swap'))]
fz = lambda d: d[d.regime == 'frozen']

# %% [markdown]
# ## Q1 · Clean data with missing sensors: accuracy and calibration (no harm from CXF?)

# %%
report.scoreboard(SUB, ['MAE', 'NLL', 'Cov90', 'CalErr'], by=['regime', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_10_q1_clean.csv'))

# %% [markdown]
# ## Q2 · One sensor corrupted (pure noise or swapped): accuracy and 90% coverage

# %%
report.scoreboard(ONE, ['MAE', 'Cov90', 'NLL'], by=['regime', 'fusion'], save=os.path.join(OUT, 'scoreboard_10_q2_corrupt.csv'),
                  sort='MAE')

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
report.bar(fz(ONE), 'MAE', x='fusion', hue='kind', ax=axes[0], title='MAE with one sensor corrupted (frozen experts)')
report.bar(fz(ONE), 'Cov90', x='fusion', hue='kind', ax=axes[1], title='90% coverage with one sensor corrupted')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_10_q2.png'), dpi=150); plt.show()

# %% [markdown]
# ## Q3 · Calibrated ranges under corruption without corrupted calibration data (D3x = CXF-C)

# %%
C = R[(R.unc == 'base') & R.kind.isin(['noise', 'swap']) & R.decoder.isin(['D2', 'D3b', 'D3m', 'D3x'])]
report.scoreboard(C[C.regime == 'finetuned'], ['Cov90', 'Width90', 'IS90'], by=['fusion', 'decoder'],
                  save=os.path.join(OUT, 'scoreboard_10_q3_ranges.csv'))

# %% [markdown]
# ## Q4 · Which sensor is lying? Detection AUROC (0.5 = chance) and localisation accuracy (chance = 1/3)

# %%
det = []
for _, r in ONE.iterrows():
    for col, nm in (('det_AUROC_sd', 'own sigma'), ('det_AUROC_dis', 'disagreement'), ('det_AUROC_cxf', 'CXF posterior')):
        f = str(r['fusion'])
        if col in r and pd.notna(r[col]) and ((nm == 'CXF posterior' and f.startswith(('cxf[', 'hard_drop['))) or
                                              (nm != 'CXF posterior' and f == 'poe')):
            sig = nm if nm != 'CXF posterior' else ('marginal detector (Lee/CARAT)' if f.startswith('hard_drop') else 'CXF posterior')
            det.append(dict(regime=r['regime'], seed=r['seed'], case=f"{r['kind']} {r['sensor']}", signal=sig,
                            AUROC=r[col], fusion=sig))
DET = pd.DataFrame(det)
report.HEADLINE['AUROC'] = ('Detection AUROC', 'higher', 0.5)
report.scoreboard(DET, ['AUROC'], by=['regime', 'case', 'signal'], save=os.path.join(OUT, 'scoreboard_10_q4_detect.csv'))

# %%
report.bar(DET[DET.regime == 'frozen'], 'AUROC', x='case', hue='signal', target=0.5,
           title='Can the system tell that a sensor is corrupted? (frozen experts)', save=os.path.join(OUT, 'fig_10_q4.png'))

# %%
LOC = ONE[ONE.fusion.astype(str).str.startswith(('cxf', 'hard_drop')) & ONE.localise_acc.notna()]
report.scoreboard(LOC.assign(case=LOC.kind + ' ' + LOC.sensor), ['localise_acc', 'false_alarm'], by=['fusion', 'case'],
                  save=os.path.join(OUT, 'scoreboard_10_q4_localise.csv'))

# %% [markdown]
# ## Headline numbers

# %%
q2 = report.per_seed(fz(ONE), ['MAE', 'Cov90'], by=['fusion']).groupby('fusion').mean()
dd = DET[DET.regime == 'frozen'].groupby('signal')['AUROC'].mean()
rows = [dict(question=f'One sensor corrupted, {f}: MAE / 90% coverage', number=f'{r.MAE:.3f} / {r.Cov90:.3f}')
        for f, r in q2.iterrows()]
rows += [dict(question=f'Corruption detection AUROC, {s}', number=f'{v:.3f}') for s, v in dd.items()]
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_10.csv'), index=False); H
