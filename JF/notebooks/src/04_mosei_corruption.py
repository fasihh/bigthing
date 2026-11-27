# %% [markdown]
# # 04 · Corrupted and mismatched modalities at test time (Noisy TV A and C)
#
# A missing modality is easy: the system knows it is gone. A **corrupted** one is still "available", so the
# uncertainty has to notice by itself.
#
# * **Noisy TV A:** x′ = √(1−λ)·x + √λ·ε, with ε Gaussian noise matched to each feature's spread. λ = 1 is pure
#   noise of the right size. Applied to every combination of modalities.
# * **Noisy TV C:** audio and/or video taken from a different random clip (a real but wrong signal).
#
# **Questions**
# 1. Does the predicted sigma rise when an input is corrupted, and by as much as the error?
# 2. Do the 90% ranges stay honest under corruption?
# 3. Can sigma, or the disagreement between experts, flag corrupted clips (AUROC; 0.5 = chance)?
# 4. How does each method degrade as the noise level grows?
# 5. Which pretraining objective helps most?
#
# Attach `mosei_compact.npz`, 01's output, and 03's output (its trained models are reused, not retrained).

# %% [include] _setup.py

# %%
from fusionlab import BASE, batching, config, data, report, runner
import pandas as pd, numpy as np, matplotlib.pyplot as plt

RESUME_FROM = ''
cfg = config.load_config(os.path.join(CONFIGS, '04_mosei_corruption.yaml'))
data.load_mosei(cfg)
runner.resume_from(RESUME_FROM)
conds = batching.conditions(BASE, ['noise', 'swap'], cfg)            # the clean TAV reference is included
print(len(conds), 'test conditions')

# %%
runner.run_grid('corruption', BASE, cfg['objectives'], conds, frozen=cfg['frozen'], finetuned=cfg['finetuned'],
                supervised=cfg['supervised'])

# %% [markdown]
# # Results

# %%
OUT = cfg['out_dir']
R = pd.read_csv(runner.paths()['results'])
D2 = R[R.decoder == 'D2']
CLEAN = D2[D2.cond == 'TAV']
fz = lambda d, obj='vjepa': d[(d.objective == obj) & (d.regime == 'frozen')]
LMAX = max(cfg['noise_lambdas'])
NZ = D2[(D2.kind == 'noise') & (D2.lam == LMAX)]
SW = D2[D2.kind == 'swap']

# %% [markdown]
# ## Q1 · Sigma and error under pure noise (λ = 1), relative to clean input

# %%
sdr = report.ratio_to_clean(NZ, CLEAN, 'MeanSD', 'sd_ratio')
maer = report.ratio_to_clean(NZ, CLEAN, 'MAE', 'mae_ratio')
Q1 = sdr.merge(maer[['objective', 'regime', 'fusion', 'seed', 'cond', 'mae_ratio']])
Q1['corrupted'] = Q1.cond.str.extract(r'noise\[(\w+)\]')
report.scoreboard(fz(Q1[Q1.corrupted == 'T']), ['sd_ratio', 'mae_ratio'], by=['fusion'],
                  title='Text replaced by noise', save=os.path.join(OUT, 'scoreboard_04_q1_text.csv'))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
report.bar(fz(Q1[Q1.corrupted == 'T']), 'sd_ratio', x='fusion', ax=axes[0], target=1.0,
           title='Sigma when text is pure noise ÷ sigma on clean input')
report.bar(fz(Q1[Q1.corrupted == 'T']), 'mae_ratio', x='fusion', ax=axes[1], target=1.0,
           title='Error when text is pure noise ÷ error on clean input')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_04_q1.png'), dpi=150); plt.show()

# %% [markdown]
# ## Q2 · Coverage of the 90% range under pure noise, by corrupted modality (target 0.90)

# %%
NZ2 = NZ.assign(corrupted=NZ.cond.str.extract(r'noise\[(\w+)\]')[0])
report.scoreboard(fz(NZ2[NZ2.corrupted.isin(['T', 'A', 'V', 'TAV'])]), ['Cov90', 'MAE', 'NLL'],
                  by=['corrupted', 'fusion'], save=os.path.join(OUT, 'scoreboard_04_q2.csv'))

# %%
report.bar(fz(NZ2[NZ2.corrupted.isin(['T', 'A', 'V', 'TAV'])]), 'Cov90', x='fusion', hue='corrupted',
           title='90% coverage with one or all modalities replaced by noise (target 0.90)',
           save=os.path.join(OUT, 'fig_04_q2.png'))

# %% [markdown]
# ## Q3 · Flagging corrupted clips: AUROC of sigma vs of expert disagreement (0.5 = chance)

# %%
DET = pd.concat([NZ2[NZ2.corrupted.isin(['T', 'A', 'V'])].assign(case=lambda d: 'noise ' + d.corrupted),
                 SW.assign(case=lambda d: d.cond)])
report.scoreboard(fz(DET), ['det_AUROC_sd', 'det_AUROC_dis'], by=['case', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_04_q3.csv'))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
report.bar(fz(DET), 'det_AUROC_sd', x='case', hue='fusion', ax=axes[0], title='Flag by sigma')
report.bar(fz(DET), 'det_AUROC_dis', x='case', hue='fusion', ax=axes[1], title='Flag by expert disagreement')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_04_q3.png'), dpi=150); plt.show()

# %% [markdown]
# ## Q4 · Degradation as the text noise level grows

# %%
TN = D2[(D2.kind == 'noise') & (D2.corrupted == 'T')]
TN = pd.concat([TN, CLEAN.assign(lam=0.0)])
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, met in zip(axes, ['MAE', 'MeanSD', 'Cov90']):
    s = fz(TN).groupby(['fusion', 'lam', 'seed'])[met].mean().groupby(['fusion', 'lam']).mean().unstack(0)
    s.plot(ax=ax, marker='o'); ax.set_xlabel('text noise level λ'); ax.set_ylabel(met)
    if met == 'Cov90':
        ax.axhline(0.9, color='#444', ls='--', lw=1)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_04_q4.png'), dpi=150); plt.show()

# %% [markdown]
# ## Q5 · Pretraining objectives (PoE fusion)

# %%
POE = Q1[(Q1.fusion == 'poe') & (Q1.corrupted == 'T')]
report.scoreboard(POE, ['sd_ratio', 'mae_ratio'], by=['objective', 'regime'],
                  save=os.path.join(OUT, 'scoreboard_04_q5_sd.csv'))

# %%
report.scoreboard(DET[DET.fusion == 'poe'], ['det_AUROC_sd', 'det_AUROC_dis'], by=['objective', 'regime'],
                  save=os.path.join(OUT, 'scoreboard_04_q5_det.csv'))

# %% [markdown]
# ## Headline numbers

# %%
a = fz(Q1[Q1.corrupted == 'T']).groupby('fusion')[['sd_ratio', 'mae_ratio']].mean()
c = fz(NZ2[NZ2.corrupted == 'T']).groupby('fusion')['Cov90'].mean()
dd = fz(DET).groupby('fusion')[['det_AUROC_sd', 'det_AUROC_dis']].mean()
rows = []
for f in a.index:
    rows += [dict(question=f'{f}: sigma ×{a.loc[f, "sd_ratio"]:.2f} while error ×{a.loc[f, "mae_ratio"]:.2f} '
                           '(text = noise)', number=f'{a.loc[f, "sd_ratio"]:.2f} vs {a.loc[f, "mae_ratio"]:.2f}'),
             dict(question=f'{f}: 90% coverage, text = noise', number=f'{c.loc[f]:.3f}'),
             dict(question=f'{f}: corruption-flag AUROC, sigma / disagreement',
                  number=f'{dd.loc[f, "det_AUROC_sd"]:.2f} / {dd.loc[f, "det_AUROC_dis"]:.2f}')]
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_04.csv'), index=False); H
