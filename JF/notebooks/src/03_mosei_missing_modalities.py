# %% [markdown]
# # 03 · CMU-MOSEI with missing modalities: every fusion method compared
#
# **Questions**
# 1. Which fusion method is most accurate and best calibrated when modalities are missing?
# 2. When text (the strongest modality) is missing, which methods still give honest 90% ranges?
# 3. Does the predicted uncertainty rise as much as the error does when text is lost (tracking ratio)?
# 4. Which pretraining objective gives the best experts (S-JEPA included)?
# 5. What happens as more and more of the text is dropped?
#
# **Compared:** fusion = concat, late average, MoE, attention, PoE, prior-corrected PoE, MoNIG ·
# objective = V-JEPA, S-JEPA, reconstruction, alignment, supervised · regime = frozen experts / fine-tuned.
#
# **Test conditions:** the 7 modality subsets (T, A, V, TA, TV, AV, TAV) and 25–90% of text tokens dropped.
# Metrics use decoder D2 (the model's own Gaussian, no recalibration) unless stated; calibrated decoders are in 06.
#
# Attach: `mosei_compact.npz` and the output of 01 (pretrained experts). Resumable.

# %% [include] _setup.py

# %%
from fusionlab import BASE, batching, config, data, report, runner
import pandas as pd, numpy as np, matplotlib.pyplot as plt

RESUME_FROM = ''
cfg = config.load_config(os.path.join(CONFIGS, '03_mosei_missing.yaml'))
data.load_mosei(cfg)
runner.resume_from(RESUME_FROM)
conds = batching.conditions(BASE, ['subset', 'textdrop'], cfg)
print(len(conds), 'test conditions:', [c['name'] for c in conds])

# %%
runner.run_grid('missing', BASE, cfg['objectives'], conds, frozen=cfg['frozen'], finetuned=cfg['finetuned'],
                supervised=cfg['supervised'], probes=True)

# %% [markdown]
# # Results

# %%
OUT = cfg['out_dir']
R = pd.read_csv(runner.paths()['results'])
D2 = R[R.decoder == 'D2']
SUB = D2[D2.cond.isin(report.SUBSETS)]
M = ['MAE', 'Acc2', 'NLL', 'Cov90', 'CalErr', 'Spearman_unc_err']
fz = lambda d, obj='vjepa': d[(d.objective == obj) & (d.regime == 'frozen')]

# %% [markdown]
# ## Q1 · Fusion methods, averaged over the 7 modality subsets (V-JEPA experts, frozen)

# %%
report.scoreboard(fz(SUB), M, by=['fusion'], save=os.path.join(OUT, 'scoreboard_03_q1_fusion.csv'), sort='NLL')

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, met in zip(axes, ['MAE', 'NLL', 'Cov90']):
    report.bar(fz(SUB), met, x='fusion', ax=ax, title=met)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_03_q1.png'), dpi=150); plt.show()

# %% [markdown]
# Same comparison for every regime (frozen, fine-tuned, supervised), V-JEPA or no pretraining:

# %%
report.scoreboard(SUB[SUB.objective.isin(['vjepa', 'none'])], M, by=['regime', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_03_q1_regimes.csv'))

# %% [markdown]
# ## Q2 · Text missing (A, V, AV only): coverage of the 90% range (target 0.90)

# %%
NOTXT = SUB[SUB.cond.isin(report.NOTXT)]
report.scoreboard(fz(NOTXT), ['Cov90', 'Width90', 'NLL', 'MAE'], by=['fusion'],
                  save=os.path.join(OUT, 'scoreboard_03_q2_notext.csv'))

# %%
report.bar(NOTXT[NOTXT.objective == 'vjepa'], 'Cov90', x='fusion', hue='regime',
           title='Coverage of the 90% range when text is missing (target 0.90)',
           save=os.path.join(OUT, 'fig_03_q2.png'))

# %%
w = [(f, *report.wins(fz(NOTXT), 'Cov90', 'poe', f, pair_on=('seed',))) for f in cfg['frozen'] if f != 'poe']
pd.DataFrame(w, columns=['vs', 'poe_closer_to_0.90', 'pairs'])

# %% [markdown]
# ## Q3 · Does sigma rise with the error when text is lost? (tracking ratio, target 1.0)

# %%
TR = report.tracking(SUB)
TR.to_csv(os.path.join(OUT, 'tracking_03.csv'), index=False)
report.scoreboard(TR, ['tracking'], by=['objective', 'regime', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_03_q3_tracking.csv'))

# %%
report.bar(TR[TR.regime != 'finetuned'], 'tracking', x='fusion', hue='objective',
           title='Tracking ratio, frozen experts (1.0 = sigma rises exactly as much as the error)',
           save=os.path.join(OUT, 'fig_03_q3.png'))

# %% [markdown]
# ## Q4 · Pretraining objectives (PoE fusion), including S-JEPA

# %%
POE = SUB[SUB.fusion == 'poe']
report.scoreboard(POE, M + ['probe_signal_R2'], by=['objective', 'regime'],
                  save=os.path.join(OUT, 'scoreboard_03_q4_objectives.csv'))

# %%
report.bar(POE, 'NLL', x='objective', hue='regime', title='NLL by pretraining objective (PoE fusion)',
           save=os.path.join(OUT, 'fig_03_q4.png'))

# %% [markdown]
# ## Q5 · Dropping more and more text tokens (text only, V-JEPA frozen)

# %%
TD = D2[(D2.kind == 'textdrop') & D2.cond.str.startswith('T~')]
TD = pd.concat([TD, D2[D2.cond == 'T'].assign(text_drop=0.0)])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, met in zip(axes, ['Cov90', 'MAE']):
    s = fz(TD).groupby(['fusion', 'text_drop', 'seed'])[met].mean().groupby(['fusion', 'text_drop']).mean().unstack(0)
    s.plot(ax=ax, marker='o')
    ax.set_xlabel('share of text tokens dropped'); ax.set_ylabel(met)
    if met == 'Cov90':
        ax.axhline(0.9, color='#444', ls='--', lw=1)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_03_q5.png'), dpi=150); plt.show()

# %% [markdown]
# ## Per-subset accuracy of the best configuration (for comparison with published MOSEI numbers)

# %%
best = report.per_seed(SUB, ['NLL']).groupby(['objective', 'regime', 'fusion'])['NLL'].mean().idxmin()
print('best (lowest NLL):', best)
b = SUB[(SUB.objective == best[0]) & (SUB.regime == best[1]) & (SUB.fusion == best[2])]
b.groupby('cond')[['MAE', 'Corr', 'Acc2', 'Acc2_has0', 'Acc7', 'F1']].mean().reindex(report.SUBSETS).round(3)

# %% [markdown]
# ## Headline numbers

# %%
q1 = report.per_seed(fz(SUB), ['Cov90', 'NLL', 'MAE'], by=['fusion']).groupby('fusion').mean()
q2 = report.per_seed(fz(NOTXT), ['Cov90'], by=['fusion']).groupby('fusion')['Cov90'].mean()
tr = TR[(TR.objective == 'vjepa') & (TR.regime == 'frozen')].groupby('fusion')['tracking'].mean()
rows = [dict(question=f'90% coverage, all subsets, {f}', number=f'{q1.loc[f, "Cov90"]:.3f}') for f in q1.index]
rows += [dict(question=f'90% coverage, text missing, {f}', number=f'{q2.loc[f]:.3f}') for f in q2.index]
rows += [dict(question=f'Tracking ratio, {f}', number=f'{tr.loc[f]:.2f}') for f in tr.index]
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_03.csv'), index=False); H
