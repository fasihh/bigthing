# %% [markdown]
# # 01 · Pretrain the modality experts (stage 1, no labels)
#
# **Question:** do the experts learn a usable, non-collapsed representation under each objective?
#
# Trains one text, audio and video expert per objective and seed, and saves them to `ckpt/`. Notebooks 03, 04, 06
# and 07 load these instead of retraining: after this run, *Save Version* and attach its output there.
#
# | Objective | What the experts learn to predict | Source |
# | --- | --- | --- |
# | `vjepa` | the EMA text latent, as a Gaussian belief | this project (after Huang 2026) |
# | `sjepa` | the EMA text latent through a stochastic latent with learned prior | Giakoumoglou et al. 2026 |
# | `recon` | the pooled input features (reconstruction, MVAE) | Wu & Goodman 2018 |
# | `align` | the matching clip's text latent among the batch (InfoNCE) | CLIP-style |
# | `vjepa_vt`, `vjepa_vts` | a variational (distribution) target | novel |
# | `vjepa_pe` | as `vjepa`, with a 5-head predictor ensemble | novel |
#
# Expected time on a Kaggle P100: about 3–5 minutes per objective and seed (21 runs). Resumable.

# %% [include] _setup.py

# %%
from fusionlab import BASE, config, data, pretrain, report, runner
import pandas as pd

RESUME_FROM = ''           # e.g. '/kaggle/input/<this-notebook>' to continue an unfinished run
cfg = config.load_config(os.path.join(CONFIGS, '01_pretrain_experts.yaml'))
data.load_mosei(cfg)
runner.resume_from(RESUME_FROM)

# %%
diag = []
for seed in cfg['seeds']:
    for obj in cfg['objectives']:
        print(f'=== {obj} seed {seed}')
        pretrain.pretrain_or_load(obj, BASE, 'main', seed, cfg, diag)
        runner.append_rows(runner.paths()['diag'], diag)
        diag.clear()

# %% [markdown]
# ## Scoreboard: representation health
# Effective rank (out of 64) and per-dimension spread of the fused latent on the validation split. A collapsed
# representation has an effective rank near 1 and a spread near 0.

# %%
D = pd.read_csv(runner.paths()['diag']).drop_duplicates(['objective', 'seed'], keep='last')
cols = ['fused_erank', 'fused_std_mean', 'fused_std_min', 'target_erank', 'pretrain_min']
tab = D.groupby('objective')[[c for c in cols if c in D]].agg(['mean', 'std']).round(3)
tab.to_csv(os.path.join(cfg['out_dir'], 'scoreboard_01_collapse.csv'))
tab

# %%
D['fusion'] = D['objective']
report.HEADLINE['fused_erank'] = ('Effective rank of fused latent (of 64)', 'higher', None)
report.bar(D, 'fused_erank', x='objective', title='No objective collapses: effective rank of the fused latent',
           save=os.path.join(cfg['out_dir'], 'fig_01_erank.png'))
