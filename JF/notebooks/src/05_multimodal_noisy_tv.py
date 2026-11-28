# %% [markdown]
# # 05 · Multimodal noisy TV: a noise source present during training
#
# Huang's noisy TV asks whether a model ignores a high-variance distractor. Here it is made multimodal, on real
# CMU-MOSEI clips (only the noise is synthetic, as the experiment requires).
#
# * **B · Noise channel:** a 4th "modality" of pure Gaussian noise (50 steps × 32 values, fresh every batch).
#   *Does the fusion learn to give it no weight, and stay accurate and calibrated with it present?*
# * **D · Sticky distractor (Huang-style):** an AR(1) process (ρ = 0.9, 8 dims) mixed into every modality's
#   features at σ = 2 or 8 times the feature spread (4× or 64× the variance), during training and testing.
#   *independent* = a different distractor per modality; *shared* = the same one in all three.
#   *Does the latent keep the sentiment (signal R², higher is better) and drop the distractor (lower is better)?*
#
# **Compared:** objective = V-JEPA, S-JEPA, reconstruction, alignment, supervised · fusion = every baseline in B,
# PoE / MoE / attention in D. These experts are trained here (the inputs differ from the clean ones in 01).
# Expected time: about 4 hours on a P100; resumable.

# %% [include] _setup.py

# %%
from fusionlab import BASE, batching, config, data, report, runner
import pandas as pd, numpy as np, matplotlib.pyplot as plt

RESUME_FROM = ''
RUN_B, RUN_D = True, True
cfg = config.load_config(os.path.join(CONFIGS, '05_multimodal_noisy_tv.yaml'))
data.load_mosei(cfg)
runner.resume_from(RESUME_FROM)

# %% [markdown]
# ## B · Noise channel

# %%
NMODS = BASE + ['n']
if RUN_B:
    runner.run_grid('noisech', NMODS, cfg['objectives'], batching.subset_conditions(NMODS),
                    frozen=cfg['nc_frozen'], supervised=cfg['nc_supervised'], pgroup='noisech')

# %% [markdown]
# ## D · Sticky distractor

# %%
SUBS = batching.subset_conditions(BASE)
if RUN_D:
    for pat in cfg['dist_patterns']:
        for sig in cfg['dist_sigmas']:
            env = dict(pattern=pat, sigma=sig, shared=pat.endswith('shared'), mods=list(BASE))
            g = f'dist_{pat}_s{sig:g}'
            runner.run_grid(g, BASE, cfg['objectives'], SUBS, frozen=cfg['dist_frozen'],
                            supervised=cfg['dist_supervised'], env=env, pgroup=g, probes=True)

# %% [markdown]
# # Results

# %%
OUT = cfg['out_dir']
R = pd.read_csv(runner.paths()['results'])
D2 = R[R.decoder == 'D2']
B = D2[D2.grid == 'noisech']
Dd = D2[D2.grid.str.startswith('dist_')]

# %% [markdown]
# ## B1 · How much weight does each fusion give the noise channel? (all four inputs available; lower is better)

# %%
BN = B[B.cond == 'TAV+N']
report.scoreboard(BN[BN.regime == 'frozen'], ['share_n', 'MAE', 'Cov90'], by=['objective', 'fusion'],
                  save=os.path.join(OUT, 'scoreboard_05_b1.csv'))

# %%
report.bar(BN, 'share_n', x='fusion', hue='objective', target=0.0,
           title='Weight on the pure-noise channel (0 = ignored)', save=os.path.join(OUT, 'fig_05_b1.png'))

# %% [markdown]
# ## B2 · Cost of the noise channel: MAE and 90% coverage with vs without it (averaged over the 7 subsets)

# %%
B2 = B.assign(with_noise=B.cond.str.endswith('+N'))
report.scoreboard(B2[B2.objective == 'vjepa'], ['MAE', 'Cov90', 'NLL'], by=['fusion', 'with_noise'],
                  save=os.path.join(OUT, 'scoreboard_05_b2.csv'))

# %%
report.bar(B2[(B2.objective == 'vjepa') & B2.with_noise], 'Cov90', x='fusion', hue='regime',
           title='90% coverage with the noise channel present (target 0.90)', save=os.path.join(OUT, 'fig_05_b2.png'))

# %% [markdown]
# ## D1 · Signal vs distractor kept in the latent (Huang's probes, multimodal)
# Signal R²: ridge probe of the sentiment from the fused latent. Distractor R²: same probe for the distractor.

# %%
P = Dd[Dd.cond == 'TAV'].drop_duplicates('key')
P = P.assign(setting=P.dist_pattern + ' σ=' + P.dist_sigma.astype(str))
report.scoreboard(P, ['probe_signal_R2', 'probe_noise_R2'], by=['setting', 'objective'],
                  save=os.path.join(OUT, 'scoreboard_05_d1.csv'))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
pp = P[P.fusion == 'poe']
report.bar(pp, 'probe_signal_R2', x='setting', hue='objective', ax=axes[0], title='Signal R² (higher is better)')
report.bar(pp, 'probe_noise_R2', x='setting', hue='objective', ax=axes[1], title='Distractor R² (lower is better)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_05_d1.png'), dpi=150); plt.show()

# %% [markdown]
# ## D2 · Task accuracy and calibration under the distractor (all modalities, PoE)

# %%
DT = Dd[Dd.cond.isin(report.SUBSETS)]
DT = DT.assign(setting=DT.dist_pattern + ' σ=' + DT.dist_sigma.astype(str))
report.scoreboard(DT[DT.fusion == 'poe'], ['MAE', 'Cov90', 'NLL'], by=['setting', 'objective'],
                  save=os.path.join(OUT, 'scoreboard_05_d2.csv'))

# %%
report.bar(DT[(DT.fusion == 'poe') & (DT.dist_sigma == DT.dist_sigma.max())], 'MAE', x='objective', hue='setting',
           title='MAE under the strongest distractor (σ = 8)', save=os.path.join(OUT, 'fig_05_d2.png'))

# %% [markdown]
# ## Headline numbers

# %%
s = BN[BN.regime == 'frozen'].groupby(['objective', 'fusion'])['share_n'].mean()
p = P[P.fusion == 'poe'].groupby(['setting', 'objective'])[['probe_signal_R2', 'probe_noise_R2']].mean()
rows = [dict(question=f'Weight on noise channel, {o} / {f}', number=f'{v:.3f}') for (o, f), v in s.items()]
rows += [dict(question=f'{st}, {o}: signal R² / distractor R²',
              number=f'{r.probe_signal_R2:.3f} / {r.probe_noise_R2:.3f}') for (st, o), r in p.iterrows()]
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_05.csv'), index=False); H
