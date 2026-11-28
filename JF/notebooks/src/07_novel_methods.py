# %% [markdown]
# # 07 · Novel methods vs the best baselines
#
# Each method targets a weakness found in the first experiments:
#
# | Method | Weakness it targets | Where it acts |
# | --- | --- | --- |
# | `vjepa_vt` / `vjepa_vts`: **variational target** | the EMA target was a point, not a distribution | pretraining |
# | `vjepa_pe`: **predictor ensemble** (5 heads) | learned sigma is aleatoric only; does not rise on corrupted input | experts |
# | `me`: **masking-ensemble uncertainty** | same, without retraining (K = 8 random maskings at test time) | test time |
# | `dapoe`: **disagreement-aware PoE** | PoE's variance ignores conflicts between experts | fusion |
# | `rpoe`: **robust PoE** | a corrupted expert keeps its full weight | fusion |
# | `rdapoe` | both of the above | fusion |
# | `mpoe`: **marginal-corrected PoE** | an uninformative expert is counted as information | fusion |
#
# Compared with plain `vjepa` + PoE (the paired baseline), S-JEPA, and the best baselines from 03/04 (MoE, MoNIG),
# on the same seeds and conditions. Attach `mosei_compact.npz` and 01's output. Expected time: about 4–5 hours.

# %% [include] _setup.py

# %%
from fusionlab import BASE, batching, config, data, report, runner
import pandas as pd, numpy as np, matplotlib.pyplot as plt

RESUME_FROM = ''
RUN_MAIN, RUN_NOISE_CHANNEL, RUN_DISTRACTOR = True, True, True
cfg = config.load_config(os.path.join(CONFIGS, '07_novel_methods.yaml'))
data.load_mosei(cfg)
runner.resume_from(RESUME_FROM)
conds = batching.conditions(BASE, ['subset', 'textdrop', 'noise', 'swap'], cfg)

# %%
if RUN_MAIN:
    runner.run_grid('novel', BASE, cfg['objectives'], conds, frozen=cfg['frozen'], finetuned=cfg['finetuned'],
                    use_me=True)
if RUN_NOISE_CHANNEL:
    NM = BASE + ['n']
    runner.run_grid('novel_noisech', NM, cfg['nc_objectives'], batching.subset_conditions(NM),
                    frozen=cfg['nc_fusions'], pgroup='noisech')
if RUN_DISTRACTOR:
    env = dict(pattern=cfg['dist_pattern'], sigma=cfg['dist_sigma'], shared=cfg['dist_pattern'].endswith('shared'),
               mods=list(BASE))
    g = f'dist_{cfg["dist_pattern"]}_s{cfg["dist_sigma"]:g}'
    runner.run_grid('novel_' + g, BASE, cfg['dist_objectives'], batching.subset_conditions(BASE), frozen=['poe'],
                    env=env, pgroup=g, probes=True)

# %% [markdown]
# # Results

# %%
OUT = cfg['out_dir']
R = pd.read_csv(runner.paths()['results'])
N = R[(R.grid == 'novel') & (R.decoder == 'D2')]
SUB = N[N.cond.isin(report.SUBSETS)]
base = lambda d: d[d.unc == 'base']
fz = lambda d: d[d.regime == 'frozen']

# %% [markdown]
# ## Q1 · Variational target and predictor ensemble vs V-JEPA and S-JEPA (PoE, 7 subsets)

# %%
report.scoreboard(base(SUB[SUB.fusion == 'poe']), ['MAE', 'NLL', 'Cov90', 'CalErr', 'Spearman_unc_err'],
                  by=['objective', 'regime'], save=os.path.join(OUT, 'scoreboard_07_q1.csv'))

# %%
TR = report.tracking(base(SUB))
report.bar(TR[(TR.fusion == 'poe')], 'tracking', x='objective', hue='regime',
           title='Tracking ratio by objective, PoE (target 1.0)', save=os.path.join(OUT, 'fig_07_q1.png'))

# %% [markdown]
# ## Q2 · Novel fusions vs PoE, MoE and MoNIG (V-JEPA experts, frozen)

# %%
V = base(fz(SUB[SUB.objective == 'vjepa']))
report.scoreboard(V, ['MAE', 'NLL', 'Cov90', 'CalErr'], by=['fusion'], save=os.path.join(OUT, 'scoreboard_07_q2.csv'),
                  sort='NLL')

# %%
w = [(f, *report.wins(V, 'NLL', f, 'poe', pair_on=('seed',))) for f in cfg['frozen'] if f != 'poe']
pd.DataFrame(w, columns=['method', 'beats_poe_on_NLL', 'pairs'])

# %% [markdown]
# ## Q3 · Does the uncertainty now notice corruption? Sigma ratio and flagging AUROC (text = noise, λ = 1)

# %%
LMAX = max(cfg['noise_lambdas'])
CL = N[N.cond == 'TAV']
NZ = N[(N.kind == 'noise') & (N.corrupted == 'T') & (N.lam == LMAX)]
Q3 = report.ratio_to_clean(NZ, CL, 'MeanSD', 'sd_ratio', by=('objective', 'regime', 'fusion', 'unc'))
report.scoreboard(fz(Q3), ['sd_ratio'], by=['objective', 'fusion', 'unc'], save=os.path.join(OUT, 'scoreboard_07_q3_sd.csv'))

# %%
DET = N[N.kind.isin(['noise', 'swap']) & ((N.lam == LMAX) | (N.kind == 'swap'))]
cols = ['det_AUROC_sd', 'det_AUROC_dis', 'det_AUROC_epi', 'det_AUROC_pe']
report.scoreboard(fz(DET[DET.fusion == 'poe']), cols, by=['objective', 'unc'],
                  save=os.path.join(OUT, 'scoreboard_07_q3_det.csv'))

# %%
melt = fz(DET[DET.fusion == 'poe']).melt(id_vars=['objective', 'unc', 'seed'], value_vars=[c for c in cols if c in DET],
                                         var_name='signal', value_name='AUROC').dropna()
melt['signal'] = melt['signal'].str.replace('det_AUROC_', '') + ' (' + melt['unc'] + ')'
report.HEADLINE['AUROC'] = ('Corruption-flag AUROC', 'higher', 0.5)
report.bar(melt, 'AUROC', x='objective', hue='signal', title='Flagging corrupted clips (0.5 = chance)',
           save=os.path.join(OUT, 'fig_07_q3.png'))

# %% [markdown]
# ## Q4 · Coverage under corruption, all methods (text = noise, λ = 1; target 0.90)

# %%
report.scoreboard(fz(NZ), ['Cov90', 'MAE'], by=['objective', 'fusion', 'unc'], save=os.path.join(OUT, 'scoreboard_07_q4.csv'))

# %% [markdown]
# ## Q5 · Noise channel and distractor with the novel methods

# %%
NC = R[(R.grid == 'novel_noisech') & (R.decoder == 'D2') & (R.cond == 'TAV+N')]
report.scoreboard(NC, ['share_n', 'MAE', 'Cov90'], by=['objective', 'fusion'], save=os.path.join(OUT, 'scoreboard_07_q5_nc.csv'))

# %%
DI = R[R.grid.str.startswith('novel_dist') & (R.decoder == 'D2') & (R.cond == 'TAV')].drop_duplicates('key')
report.scoreboard(DI, ['probe_signal_R2', 'probe_noise_R2', 'MAE', 'Cov90'], by=['objective'],
                  save=os.path.join(OUT, 'scoreboard_07_q5_dist.csv'))

# %% [markdown]
# ## Headline numbers

# %%
q2 = report.per_seed(V, ['NLL', 'Cov90'], by=['fusion']).groupby('fusion').mean()
q3 = fz(DET[DET.fusion == 'poe']).groupby(['objective', 'unc'])[[c for c in cols if c in DET]].mean()
rows = [dict(question=f'NLL / 90% coverage, {f}', number=f'{r.NLL:.3f} / {r.Cov90:.3f}') for f, r in q2.iterrows()]
for (o, u), r in q3.iterrows():
    best = r.dropna().idxmax()
    rows.append(dict(question=f'Best corruption flag, {o} ({u})', number=f'{best.replace("det_AUROC_", "")} = {r[best]:.2f}'))
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_07.csv'), index=False); H
