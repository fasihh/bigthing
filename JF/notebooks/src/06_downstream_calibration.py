# %% [markdown]
# # 06 · Downstream use of the uncertainty: decoders, calibration, selective prediction
#
# **Question:** given the same predictions, how much better are prediction ranges and decisions when the model's
# uncertainty is used, and which calibration step makes them honest everywhere?
#
# Calibration uses the validation split **pooled over every condition** (clean subsets, missing text, noise,
# swaps): a deployed system does not know in advance which input will be missing or corrupted.
#
# | Decoder | Range at level p | Uses the model's sigma? |
# | --- | --- | --- |
# | D0 | point decoder ± constant | no |
# | D1 | head mean ± constant | no |
# | D2 | ± z_p · σ (raw) | yes |
# | D3a | ± z_p · τ · σ (one rescaling factor) | yes |
# | D3b | ± q_p · σ (normalised split conformal) | yes |
# | D3m | ± q_p(mask) · σ: one quantile per modality-availability pattern (Mondrian, as in MCCF 2026) | yes |
# | D3c | ± q_p · σ calibrated per condition (oracle reference, not deployable) | yes |
#
# Attach `mosei_compact.npz` and the outputs of 01, 03 and 04 (models are reused).

# %% [include] _setup.py

# %%
from fusionlab import BASE, batching, config, data, report, runner
import pandas as pd, numpy as np, matplotlib.pyplot as plt

RESUME_FROM = ''
cfg = config.load_config(os.path.join(CONFIGS, '06_downstream_calibration.yaml'))
data.load_mosei(cfg)
runner.resume_from(RESUME_FROM)
conds = batching.conditions(BASE, ['subset', 'textdrop', 'noise', 'swap'], cfg)
print(len(conds), 'conditions pooled for calibration')

# %%
runner.run_grid('calibration', BASE, cfg['objectives'], conds, frozen=cfg['frozen'], finetuned=cfg['finetuned'],
                supervised=cfg['supervised'])

# %% [markdown]
# # Results

# %%
OUT = cfg['out_dir']
R = pd.read_csv(runner.paths()['results'])
GROUPS = {'clean subsets': R.kind == 'subset', 'text missing': R.cond.isin(report.NOTXT),
          'text partly dropped': R.kind == 'textdrop', 'noise': R.kind == 'noise', 'swap': R.kind == 'swap'}
DEC = ['D0', 'D1', 'D2', 'D3a', 'D3b', 'D3m', 'D3c']
V = R[(R.objective == 'vjepa') & (R.regime == 'frozen') & (R.fusion == 'poe')]

# %% [markdown]
# ## Q1 · Coverage of the 90% range per decoder and condition group (V-JEPA + PoE, target 0.90)

# %%
tab = pd.DataFrame({g: V[m & V.decoder.isin(DEC)].groupby(['decoder', 'seed'])['Cov90'].mean()
                    .groupby('decoder').mean() for g, m in GROUPS.items() if (m & V.decoder.isin(DEC)).any()}).reindex(DEC)
tab.round(3).to_csv(os.path.join(OUT, 'scoreboard_06_q1_cov.csv')); tab.round(3)

# %%
d = pd.concat([V[m].assign(group=g) for g, m in GROUPS.items()])
report.bar(d[d.decoder.isin(DEC)], 'Cov90', x='decoder', hue='group',
           title='90% coverage per decoder and condition group (target 0.90)', save=os.path.join(OUT, 'fig_06_q1.png'))

# %% [markdown]
# ## Q2 · Width and interval score: the price of honest coverage (lower is better)

# %%
report.scoreboard(V[V.decoder.isin(DEC)], ['Cov90', 'Width90', 'IS90'], by=['decoder'],
                  save=os.path.join(OUT, 'scoreboard_06_q2.csv'))

# %% [markdown]
# ## Q3 · Worst-case coverage gap over the 7 modality subsets (lower is better)
# For each model: the largest |coverage − 0.90| across the subsets. Pooled conformal (D3b) is accurate on
# average but can miss per subset; the per-mask quantiles (D3m) target exactly this.

# %%
S = R[(R.kind == 'subset') & R.decoder.isin(['D2', 'D3a', 'D3b', 'D3m'])]
gap = S.assign(g=(S.Cov90 - 0.9).abs()).groupby(['objective', 'regime', 'fusion', 'decoder', 'seed'])['g'].max()
gap = gap.rename('cov_gap').reset_index()
report.scoreboard(gap, ['cov_gap'], by=['fusion', 'decoder'], save=os.path.join(OUT, 'scoreboard_06_q3.csv'))

# %%
report.bar(gap[gap.objective == 'vjepa'], 'cov_gap', x='fusion', hue='decoder', target=0.0,
           title='Worst per-subset 90% coverage gap', save=os.path.join(OUT, 'fig_06_q3.png'))

# %% [markdown]
# ## Q4 · Selective prediction: keep only the most confident clips

# %%
SEL = R[(R.decoder == 'D2') & (R.kind == 'subset')]
report.scoreboard(SEL, ['MAE', 'selMAE@80', 'selMAE@50', 'AUROC_bigerr', 'E_AURC'], by=['objective', 'regime', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_06_q4.csv'), sort='E_AURC')

# %%
import json
cur = [json.loads(l) for l in open(runner.paths()['curves'])]
cur = pd.DataFrame([c for c in cur if c.get('grid') == 'calibration' and c['cond'] == 'TAV'])
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for i, (f, g) in enumerate(cur[(cur.objective == 'vjepa') & (cur.regime == 'frozen')].groupby('fusion')):
    ax.plot(np.linspace(0.05, 1, 20), np.mean(np.stack(g.rc_sd.values), 0), label=f, color=report.PALETTE[i])
ax.set_xlabel('share of clips kept (most confident first)'); ax.set_ylabel('MAE of kept clips')
ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_06_q4_rc.png'), dpi=150); plt.show()

# %% [markdown]
# ## Q5 · Sign-or-defer: answer positive/negative only when the 90% range excludes 0 (D3b)

# %%
SD = R[(R.decoder == 'D3b') & (R.kind == 'subset')]
report.scoreboard(SD, ['Acc2', 'defer_answer_rate', 'defer_acc2_answered'], by=['objective', 'regime', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_06_q5.csv'))

# %% [markdown]
# ## Headline numbers

# %%
rows = [dict(question=f'90% coverage, {dec}, {g}', number=f'{tab.loc[dec, g]:.3f}')
        for dec in ['D1', 'D2', 'D3b', 'D3m'] for g in tab.columns]
gg = gap[(gap.objective == 'vjepa') & (gap.fusion == 'poe')].groupby('decoder')['cov_gap'].mean()
rows += [dict(question=f'Worst per-subset coverage gap, {k}', number=f'{v:.3f}') for k, v in gg.items()]
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_06.csv'), index=False); H
