"""Figures for the FYP-1 paper, from the first full run (results tables transcribed from the run log
JF/results/run1/tables.txt) and the Huang reproduction (JF/results/02_huang_reproduction)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'savefig.dpi': 300})
C = ['#2a6f97', '#e07a1f', '#3a9d5d', '#c03a3a', '#7b5ea7', '#6b7280']
OUT = 'figures/'

# Fig: 90% coverage by condition kind and decoder (PoE, V-JEPA, frozen vs fine-tuned)
kinds = ['text\npresent', 'text\nmissing', 'text partly\ndropped', 'A/V\nswapped', 'pure\nnoise']
cov = {('frozen', 'D1'): [0.897, 0.831, 0.892, 0.895, 0.853], ('frozen', 'D2'): [0.899, 0.884, 0.895, 0.897, 0.854],
       ('frozen', 'D3b'): [0.895, 0.878, 0.890, 0.893, 0.849], ('finetuned', 'D1'): [0.925, 0.792, 0.894, 0.918, 0.832],
       ('finetuned', 'D2'): [0.900, 0.862, 0.864, 0.890, 0.789], ('finetuned', 'D3b'): [0.920, 0.886, 0.886, 0.912, 0.813]}
lab = {'D1': 'constant width (D1)', 'D2': 'raw Gaussian (D2)', 'D3b': 'split conformal (D3b)'}
fig, axes = plt.subplots(1, 2, figsize=(7, 2.3), sharey=True)
for ax, rg in zip(axes, ['frozen', 'finetuned']):
    for j, d in enumerate(['D1', 'D2', 'D3b']):
        ax.bar(np.arange(5) + (j - 1) * 0.26, cov[(rg, d)], 0.25, color=C[j], label=lab[d])
    ax.axhline(0.9, color='k', ls='--', lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(kinds, fontsize=6.5); ax.set_ylim(0.75, 0.95)
    ax.set_title(f'V-JEPA experts, {"frozen" if rg == "frozen" else "fine-tuned"}, PoE')
axes[0].set_ylabel('coverage of 90% range')
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12))
plt.savefig(OUT + 'coverage_by_condition.pdf'); plt.close()

# Fig: Noisy TV A, text replaced progressively by noise (PoE)
lam = [0, 0.25, 0.5, 0.75, 1.0]
sd = {'V-JEPA frozen': [0.068, 0.066, 0.064, 0.062, 0.061], 'V-JEPA fine-tuned': [0.578, 0.570, 0.559, 0.536, 0.464],
      'recon frozen': [0.988, 1.026, 1.076, 1.148, 1.262], 'supervised': [0.453, 0.444, 0.435, 0.422, 0.377]}
mae = {'V-JEPA frozen': [0.727, 0.730, 0.740, 0.760, 0.848], 'V-JEPA fine-tuned': [0.563, 0.580, 0.611, 0.676, 0.870],
       'recon frozen': [0.713, 0.717, 0.725, 0.741, 0.826], 'supervised': [0.558, 0.581, 0.621, 0.692, 0.855]}
fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))
for i, k in enumerate(sd):
    axes[0].plot(lam, np.array(sd[k]) / sd[k][0], marker='o', ms=3, color=C[i], label=k)
    axes[1].plot(lam, np.array(mae[k]) / mae[k][0], marker='o', ms=3, color=C[i], label=k)
axes[0].set_title("text expert's predicted $\\sigma$ (relative to clean)"); axes[1].set_title('MAE (relative to clean)')
for ax in axes:
    ax.set_xlabel('share of noise in the text stream, $\\lambda$'); ax.axhline(1, color='k', lw=0.6)
axes[1].legend(fontsize=6.5, frameon=False)
plt.savefig(OUT + 'noisytv_text.pdf'); plt.close()

# Fig: Noisy TV D probes at sigma=8
obj = ['V-JEPA', 'recon', 'align', 'supervised']
sig = {'independent': [0.071, 0.001, 0.066, 0.468], 'shared': [0.068, 0.003, 0.009, 0.480]}
noi = {'independent': [0.181, 0.465, 0.189, 0.151], 'shared': [0.517, 0.524, 0.531, 0.338]}
clean = [0.326, 0.324, 0.345, 0.555]
fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))
x = np.arange(4)
axes[0].bar(x - 0.27, clean, 0.26, color=C[5], label='no distractor')
axes[0].bar(x, sig['independent'], 0.26, color=C[0], label='independent, $\\sigma$=8')
axes[0].bar(x + 0.27, sig['shared'], 0.26, color=C[1], label='shared, $\\sigma$=8')
axes[1].bar(x - 0.14, noi['independent'], 0.26, color=C[0]); axes[1].bar(x + 0.14, noi['shared'], 0.26, color=C[1])
axes[0].set_title('sentiment probe $R^2$ (higher is better)', fontsize=8)
axes[1].set_title('distractor probe $R^2$ (lower is better)', fontsize=8)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(obj)
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
plt.savefig(OUT + 'distractor_probes.pdf'); plt.close()

# Fig: Huang reproduction, median + IQR at scale 8
A = pd.read_csv('../results/02_huang_reproduction/huang_paper_seeds.csv')
B = pd.read_csv('../results/02_huang_reproduction/huang_new_seeds.csv')
M = ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']
fig, axes = plt.subplots(1, 2, figsize=(7, 2.3), gridspec_kw={'width_ratios': [1.1, 1]})
for j, (nm, D) in enumerate((('author seeds (5)', A), ('new seeds (10)', B))):
    q = D[D.scale == 8.0].groupby('model')['sig_te'].quantile([.25, .5, .75]).unstack().reindex(M)
    pos = np.arange(5) + (j - 0.5) * 0.38
    axes[0].bar(pos, q[0.5], 0.36, color=C[j], label=nm)
    axes[0].errorbar(pos, q[0.5], yerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]], fmt='none', ecolor='k', capsize=2, lw=0.8)
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(M); axes[0].axhline(0, color='k', lw=0.6)
axes[0].set_title('signal $R^2$ at noise scale 8: median, middle 50% of seeds', fontsize=7.5)
axes[0].legend(fontsize=6.5, frameon=False)
d = pd.concat([A.assign(s='a'), B.assign(s='b')]); d = d[d.scale == 8.0]
rng = np.random.default_rng(0)
for i, m in enumerate(M):
    v = d[d.model == m]
    axes[1].scatter(i + rng.uniform(-.15, .15, len(v)), v.sig_te, s=8, c=[C[0] if s == 'a' else C[1] for s in v.s])
axes[1].set_xticks(range(5)); axes[1].set_xticklabels(M); axes[1].set_title('every seed (blue: author, orange: new)', fontsize=7.5)
axes[1].axhline(0, color='k', lw=0.6)
plt.savefig(OUT + 'huang_reproduction.pdf'); plt.close()

# Fig: tracking ratio and swap detection
fus = ['PoE', 'MoE', 'attention']
trk = {'frozen': [1.007, 0.966, 1.043], 'fine-tuned': [0.958, 0.832, 0.960]}
trk_noise = {'frozen': [0.855, 0.901, 0.835], 'fine-tuned': [0.627, 0.778, 0.704]}
fig, axes = plt.subplots(1, 2, figsize=(7, 2.1))
x = np.arange(3)
for j, rg in enumerate(['frozen', 'fine-tuned']):
    axes[0].bar(x + (j - .5) * .38, trk[rg], .36, color=C[j], label=rg)
    axes[1].bar(x + (j - .5) * .38, trk_noise[rg], .36, color=C[j])
for ax, t in zip(axes, ['text missing (declared)', 'text replaced by pure noise (undeclared)']):
    ax.axhline(1, color='k', ls='--', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(fus); ax.set_ylim(0.5, 1.1)
    ax.set_title(f'tracking ratio, {t}', fontsize=7.5)
axes[0].legend(fontsize=6.5, frameon=False); axes[0].set_ylabel('1 = $\\sigma$ rises as much as error')
plt.savefig(OUT + 'tracking.pdf'); plt.close()
print('figures done')
