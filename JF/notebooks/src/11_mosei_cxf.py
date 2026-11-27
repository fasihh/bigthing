# %% [markdown]
# # 11 · Cross-Examined Fusion on CMU-MOSEI vs the other fusion methods
#
# The FYP-2 method on the primary dataset. CXF replaces each modality's self-reported precision with a label-free
# posterior over which modalities are lying (fitted on clean, unlabelled data only), then averages the subset
# predictions of the same PoE worker. Compared on identical models and inputs with PoE, MoE, attention, MoNIG,
# robust PoE and disagreement-aware PoE.
#
# **Test conditions:** 7 modality subsets (missing), noise on one modality (λ = 0.5, 1), one modality swapped in
# from another clip. **Auditors:** frozen V-JEPA and frozen alignment experts (Part I: disagreement AUROC 0.56 / 0.84).
#
# Attach `mosei_compact.npz` (output of 00 or 01), the output of **01** (experts) and of **03** (stage-2 models), so
# nothing is retrained. Resumable.

# %% [include] _setup.py

# %%
import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fusionlab import BASE, batching, coco, config, cxf, data, pretrain, report, runner

RESUME_FROM = ''
cfg = config.load_config(os.path.join(CONFIGS, '11_mosei_cxf.yaml'))
data.load_mosei(cfg)
runner.resume_from(RESUME_FROM)
NAME = {'t': 'text', 'a': 'audio', 'v': 'video'}

conds = (batching.conditions(BASE, ['subset', 'noise'], cfg)
         + batching.swap_conditions(BASE, (('t',), ('a',), ('v',))))
conds = [c for c in conds if c['kind'] != 'noise' or len(c['noise']) == 1]      # one corrupted sensor at a time
print(len(conds), 'conditions:', [c['name'] for c in conds])

# %% [markdown]
# ## 1 · Comparison fusions (loaded from 03 when attached)

# %%
runner.run_grid('mosei_cxf', BASE, cfg['objectives'], conds, frozen=cfg['frozen'], finetuned=cfg['finetuned'],
                supervised=cfg['supervised'], pgroup='main')

# %% [markdown]
# ## 2 · Cross-Examined Fusion on the same PoE workers
# The auditor is a frozen stage-1 model (it never sees labels or corruptions). For fine-tuned and supervised
# workers the auditor is the frozen copy of the stage-1 experts (V-JEPA for the supervised worker).

# %%
done = runner.done_keys()
for seed in cfg['seeds']:
    for aud in cfg['auditors']:
        auditor, _, marg = pretrain.pretrain_or_load(aud, BASE, 'main', seed, cfg)
        witness = cxf.Witness().fit(auditor, BASE)
        workers = [(aud, 'frozen'), (aud, 'finetuned')] + ([('none', 'supervised')] if aud == 'vjepa' else [])
        for obj, rg in workers:
            key = f'mosei_cxf|{obj}|{rg}|cxfv2[{aud}]|{seed}'
            if key in done:
                continue
            model, info = runner.build_model(auditor, obj, rg, 'poe', BASE, cfg, seed, marg, 'main')
            rows = cxf.evaluate_cxf(model, auditor, witness, conds,
                                    dict(key=key, grid='mosei_cxf', objective=obj, regime=rg, seed=seed, auditor=aud, **info))
            for r in rows:
                r['fusion'] = f"{r['fusion']}[{aud}]"
            runner.append_rows(runner.paths()['results'], rows)
            print(key, 'T =', round(witness.T, 2))

# %% [markdown]
# # Results

# %%
OUT = cfg['out_dir']
R = pd.read_csv(runner.paths()['results'])
R['sensor'] = R['corrupted'].fillna('').map({'T': 'text', 'A': 'audio', 'V': 'video', '': ''})
D2 = R[(R.decoder == 'D2') & (R.unc == 'base')]
SUB = D2[D2.cond.isin(report.SUBSETS)]
ONE = D2[(D2.kind.isin(['noise', 'swap'])) & ((D2.lam == 1.0) | (D2.kind == 'swap'))]
fz = lambda d: d[d.regime == 'frozen']

# %% [markdown]
# ## Q1 · Clean data with missing sensors: accuracy and calibration (no harm from CXF?)

# %%
report.scoreboard(SUB, ['MAE', 'NLL', 'Cov90', 'CalErr'], by=['regime', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_11_q1_clean.csv'))

# %% [markdown]
# ## Q2 · One sensor corrupted (pure noise or swapped): accuracy and 90% coverage

# %%
report.scoreboard(ONE, ['MAE', 'Cov90', 'NLL'], by=['regime', 'fusion'], save=os.path.join(OUT, 'scoreboard_11_q2_corrupt.csv'),
                  sort='MAE')

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
report.bar(fz(ONE), 'MAE', x='fusion', hue='kind', ax=axes[0], title='MAE with one sensor corrupted (frozen experts)')
report.bar(fz(ONE), 'Cov90', x='fusion', hue='kind', ax=axes[1], title='90% coverage with one sensor corrupted')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_11_q2.png'), dpi=150); plt.show()

# %% [markdown]
# ## Q3 · Calibrated ranges under corruption without corrupted calibration data (D3x = CXF-C)

# %%
C = R[(R.unc == 'base') & R.kind.isin(['noise', 'swap']) & R.decoder.isin(['D2', 'D3b', 'D3m', 'D3x'])]
report.scoreboard(C[C.regime == 'finetuned'], ['Cov90', 'Width90', 'IS90'], by=['fusion', 'decoder'],
                  save=os.path.join(OUT, 'scoreboard_11_q3_ranges.csv'))

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
report.scoreboard(DET, ['AUROC'], by=['regime', 'case', 'signal'], save=os.path.join(OUT, 'scoreboard_11_q4_detect.csv'))

# %%
report.bar(DET[DET.regime == 'frozen'], 'AUROC', x='case', hue='signal', target=0.5,
           title='Can the system tell that a sensor is corrupted? (frozen experts)', save=os.path.join(OUT, 'fig_11_q4.png'))

# %%
LOC = ONE[ONE.fusion.astype(str).str.startswith(('cxf', 'hard_drop')) & ONE.localise_acc.notna()]
report.scoreboard(LOC.assign(case=LOC.kind + ' ' + LOC.sensor), ['localise_acc', 'false_alarm'], by=['fusion', 'case'],
                  save=os.path.join(OUT, 'scoreboard_11_q4_localise.csv'))

# %% [markdown]
# ## Headline numbers

# %%
q2 = report.per_seed(fz(ONE), ['MAE', 'Cov90'], by=['fusion']).groupby('fusion').mean()
dd = DET[DET.regime == 'frozen'].groupby('signal')['AUROC'].mean()
rows = [dict(question=f'One sensor corrupted, {f}: MAE / 90% coverage', number=f'{r.MAE:.3f} / {r.Cov90:.3f}')
        for f, r in q2.iterrows()]
rows += [dict(question=f'Corruption detection AUROC, {s}', number=f'{v:.3f}') for s, v in dd.items()]
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_11.csv'), index=False); H
