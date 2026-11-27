# %% [markdown]
# # 02 · Reproducing Huang (2026): VJEPA/BJEPA on the noisy TV, with more seeds
#
# **Question:** do the paper's noisy TV results hold when the experiment is rerun with new random seeds?
#
# * The author's code is **downloaded at run time** from github.com/YongchaoHuang/VJEPA (no licence file is
#   published, so it is not copied into this repository). We run his own multi-seed script
#   (`exp1_multiseed_variance`) **unchanged**; we only choose which seeds to pass in.
# * **Part A** reruns his 5 seeds (42, 111, 123, 456, 789) and compares with the table printed in his notebook.
# * **Part B** runs 10 new seeds (0–9) with identical settings.
# * Setup (his): 4-d signal, 4-d sticky distractor (AR(1), ρ = 0.9) mixed into 20-d observations at scale 0, 4 or 8;
#   linear VAE, AR, JEPA, VJEPA, BJEPA; 3,000 epochs; linear probes of the next signal state (higher is better)
#   and of the distractor (lower is better).
#
# CPU only, about 2 minutes per seed (15 seeds ≈ 30 minutes).

# %% [include] _setup.py

# %%
from fusionlab import huang
import numpy as np, pandas as pd, matplotlib.pyplot as plt

OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else './out_02'
os.makedirs(OUT, exist_ok=True)
NEW_SEEDS = list(range(10))
repo, sha = huang.fetch(os.path.join(OUT, 'huang_vjepa'))
ns = huang.load_exp1(repo)
print('epochs', ns['N_EPOCHS'], '| scales', ns['SCALES'], '| models', huang.MODELS)
paper = huang.published_table(repo)
paper

# %% [markdown]
# ## Part A: the author's seeds

# %%
A_csv = os.path.join(OUT, 'huang_paper_seeds.csv')
A = pd.read_csv(A_csv) if os.path.exists(A_csv) else huang.run_seeds(ns, huang.PAPER_SEEDS)
A.to_csv(A_csv, index=False)
sA = huang.summary(A)
chk = sA[sA.metric == 'sig_te'].merge(paper, on=['scale', 'model'])
chk['abs_diff_vs_paper'] = (chk['mean'] - chk['sig_te']).abs()
print('Reproduction check (signal R2 test): largest difference from the published table =',
      round(chk['abs_diff_vs_paper'].max(), 3))
chk[['scale', 'model', 'mean', 'sig_te', 'abs_diff_vs_paper']].round(3)

# %% [markdown]
# ## Part B: 10 new seeds

# %%
B_csv = os.path.join(OUT, 'huang_new_seeds.csv')
B = pd.read_csv(B_csv) if os.path.exists(B_csv) else huang.run_seeds(ns, NEW_SEEDS)
B.to_csv(B_csv, index=False)
sB = huang.summary(B)
tab = sB.pivot_table(index=['scale', 'model'], columns='metric',
                     values=['mean', 'median', 'std', 'worst', 'ci_lo', 'ci_hi']).round(3)
tab.to_csv(os.path.join(OUT, 'scoreboard_02_new_seeds.csv'))
tab

# %% [markdown]
# ## Scoreboard: does VJEPA beat each baseline, seed by seed?
# Counts are out of the number of new seeds. Signal: higher R² wins. Distractor: lower R² wins.

# %%
W = huang.win_table(B)
W.to_csv(os.path.join(OUT, 'scoreboard_02_wins.csv'), index=False)
W

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, met, lab in zip(axes, ['sig_te', 'noi_te'], ['Signal R² on test (higher is better)',
                                                     'Distractor R² on test (lower is better)']):
    for j, (src, S) in enumerate((('author seeds (5)', A), (f'new seeds ({len(NEW_SEEDS)})', B))):
        q = S[S.scale == 8.0].groupby('model')[met].quantile([0.25, 0.5, 0.75]).unstack().reindex(huang.MODELS)
        pos = np.arange(5) + (j - 0.5) * 0.38
        ax.bar(pos, q[0.5], 0.36, label=src, color=['#2a6f97', '#e07a1f'][j])
        ax.errorbar(pos, q[0.5], yerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]], fmt='none', ecolor='#222', capsize=3)
        for p_, v, top in zip(pos, q[0.5], q[0.75]):
            ax.text(p_, max(v, top) + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(5)); ax.set_xticklabels(huang.MODELS)
    ax.set_title(lab + ', noise scale 8\nbar = median over seeds, whiskers = middle 50% of seeds', fontsize=9)
    ax.axhline(0, color='#444', lw=0.8); ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_02_huang_scale8.png'), dpi=150); plt.show()

# %% [markdown]
# ## Per-seed spread at the hardest noise level
# One dot per seed (blue = author's seeds, orange = new seeds). A wide spread means a single-seed result can land
# anywhere in this range; a dot far below the rest is a failed run.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.6))
d = pd.concat([A.assign(src='author'), B.assign(src='new')])
d = d[d.scale == 8.0]
for i, mdl in enumerate(huang.MODELS):
    v = d[d.model == mdl]
    ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v['sig_te'], s=14,
               c=np.where(v['src'] == 'author', '#2a6f97', '#e07a1f'))
ax.set_xticks(range(5)); ax.set_xticklabels(huang.MODELS); ax.set_ylabel('Signal R² (test), scale 8')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_02_per_seed.png'), dpi=150); plt.show()

# %% [markdown]
# ## Headline numbers (filled automatically)

# %%
def ms(S, model, met, scale=8.0):
    r = S[(S.model == model) & (S.metric == met) & (S.scale == scale)].iloc[0]
    return f"median {r['median']:.3f} · mean {r['mean']:.3f} ± {r['std']:.3f} · worst seed {r['worst']:.3f}"

rows = [dict(question=f'Signal R² at scale 8, {m} (new seeds)', number=ms(sB, m, 'sig_te')) for m in huang.MODELS]
rows += [dict(question=f'Signal R² at scale 8, {m} (author seeds, rerun)', number=ms(sA, m, 'sig_te'))
         for m in huang.MODELS]
w8 = W[W.scale == 8.0].set_index('vs')
rows += [dict(question=f'VJEPA beats {o} on signal R² at scale 8 (new seeds)',
              number=f"{w8.loc[o, 'signal_wins']}/{w8.loc[o, 'n']} seeds") for o in w8.index]
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, 'headline_02.csv'), index=False)
print('Huang repo commit:', sha)
H
