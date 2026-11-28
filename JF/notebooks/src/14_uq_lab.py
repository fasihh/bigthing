# %% [markdown]
# # 14 · Uncertainty-tracking lab on real CMU-MOSEI (CPU, minutes)
#
# Runs every method in `lab/methods.py` on the cache written by notebook 13 (attach its output). Each method may
# learn only from the self-generated corruptions (white noise λ = 1, random swap) in train / valid; the test set adds
# held-out corruptions (noise λ = 0.5, half-clip noise, gain fault, dead sensor, time shuffle, nearest-neighbour swap,
# two noisy modalities). Scores: NLL, coverage gap of 90% ranges, tracking ratio (σ rise / error rise, ideal 1) and
# AUROC of σ for corrupted vs clean clips.

# %% [include] _setup.py

# %%
import glob, subprocess
sys.path.insert(0, JF)
cache = glob.glob('/kaggle/input/**/uq_cache.npz', recursive=True)[0]
print(cache)
r = subprocess.run([sys.executable, '-m', 'lab.run', 'real', '--cache', cache, '--tags', 'vjepa_frozen', 'vjepa_vt_frozen',
                    'vjepa_finetuned', '--out', '/kaggle/working/lab_real.csv'], cwd=JF, capture_output=True, text=True)
print(r.stdout[-20000:]); print(r.stderr[-5000:])

# %%
import pandas as pd
from lab.run import summary
df = pd.read_csv('/kaggle/working/lab_real.csv')
pd.set_option('display.width', 250)
for tag in df.tag.unique():
    print('=====', tag); print(summary(df[df.tag == tag]).round(3).to_string())

# %%
# per held-out kind, NLL
print(df[df.heldout].groupby(['tag', 'kind', 'method']).NLL.mean().unstack('kind').round(3).to_string())
