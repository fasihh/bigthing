# %% [markdown]
# # 09 · FYP-2 pilot: can cross-modal consistency find the corrupted modality?
#
# **Question:** FYP-1 showed that an expert's own sigma does not rise when its input is replaced by noise or swapped
# with another clip's (AUROC about 0.5). Can a *label-free* score that compares each expert with what the *other*
# experts imply find (a) that a clip is corrupted and (b) which modality it is?
#
# Frozen V-JEPA experts from notebook 01 (attach its output). No labels, no retraining; minutes on CPU or GPU.
#
# | Score | Uses | From |
# | --- | --- | --- |
# | sigma (own) | the expert's own uncertainty | FYP-1 baseline |
# | disagreement | symmetric KL to the other experts | FYP-1 baseline |
# | conditional Mahalanobis | the expert's belief vs the clean joint distribution of all beliefs | FYP-2 core idea |
# | cross-prediction | ridge prediction of the expert's mean from the others' means | FYP-2 core idea (linear) |
# | liar posterior | P(modality corrupted) over all 8 corruption configurations | FYP-2 core idea (Bayesian) |
#
# **Go / no-go:** go if the consistency scores reach detection AUROC ≥ 0.80 on text noise and ≥ 0.70 on swaps,
# where sigma is near 0.5.

# %% [include] _setup.py

# %%
from fusionlab import BASE, config, data, pretrain, pilot
import numpy as np, pandas as pd, matplotlib.pyplot as plt

cfg = config.load_config(os.path.join(CONFIGS, '01_pretrain_experts.yaml'))
data.load_mosei(cfg)
OBJECTIVES = ['vjepa', 'sjepa', 'align', 'recon']

# %%
rows = []
for obj in OBJECTIVES:
    for seed in cfg['seeds']:
        experts, _, _ = pretrain.pretrain_or_load(obj, BASE, 'main', seed, cfg)
        cm = pilot.ConsistencyModel(pilot.beliefs(experts, 'train')[0])
        clean = pilot.beliefs(experts, 'test')
        cases = {'text = noise': ([0], pilot.beliefs(experts, 'test', ('noise', ['t'], 1.0))),
                 'audio = noise': ([1], pilot.beliefs(experts, 'test', ('noise', ['a'], 1.0))),
                 'video = noise': ([2], pilot.beliefs(experts, 'test', ('noise', ['v'], 1.0))),
                 'text 50% noise': ([0], pilot.beliefs(experts, 'test', ('noise', ['t'], 0.5))),
                 'audio swapped': ([1], pilot.beliefs(experts, 'test', ('swap', ['a']))),
                 'video swapped': ([2], pilot.beliefs(experts, 'test', ('swap', ['v']))),
                 'text swapped': ([0], pilot.beliefs(experts, 'test', ('swap', ['t'])))}
        rows += [dict(r, objective=obj, seed=seed) for r in pilot.evaluate(cm, clean, None, cases)]
        print(obj, seed, 'done')
R = pd.DataFrame(rows)
R.to_csv(os.path.join(cfg['out_dir'], 'pilot_fyp2.csv'), index=False)

# %% [markdown]
# ## Scoreboard: detection AUROC (0.5 = chance) and localisation accuracy (chance = 1/3)

# %%
tab = R.groupby(['objective', 'case', 'score'])[['detect_AUROC', 'localise_acc']].agg(['mean', 'std']).round(3)
tab.to_csv(os.path.join(cfg['out_dir'], 'scoreboard_09_pilot.csv'))
tab.loc['vjepa']

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
order = ['sigma (own)', 'disagreement', 'cond. Mahalanobis', 'cross-prediction', 'liar posterior']
for ax, met, tgt in zip(axes, ['detect_AUROC', 'localise_acc'], [0.5, 1 / 3]):
    p = R[R.objective == 'vjepa'].groupby(['case', 'score'])[met].mean().unstack()[order]
    p.plot.bar(ax=ax, width=0.8)
    ax.axhline(tgt, color='k', ls='--', lw=1)
    ax.set_title(f'{met} (V-JEPA experts; dashed = chance)'); ax.legend(fontsize=7, frameon=False)
    ax.tick_params(axis='x', rotation=25)
plt.tight_layout(); plt.savefig(os.path.join(cfg['out_dir'], 'fig_09_pilot.png'), dpi=150); plt.show()

# %% [markdown]
# ## Go / no-go

# %%
v = R[R.objective == 'vjepa'].groupby(['case', 'score'])['detect_AUROC'].mean().unstack()
best = v[['cond. Mahalanobis', 'cross-prediction', 'liar posterior']].max(1)
print('text = noise: best consistency AUROC', round(best['text = noise'], 3), 'vs sigma', round(v.loc['text = noise', 'sigma (own)'], 3))
sw = best[[c for c in best.index if 'swapped' in c]].mean()
print('swaps: best consistency AUROC', round(sw, 3))
print('GO' if best['text = noise'] >= 0.8 and sw >= 0.7 else 'NO-GO (see fallback in the proposal)')
