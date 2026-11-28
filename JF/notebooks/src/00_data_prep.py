# %% [markdown]
# # 00 · Data preparation (CMU-MOSEI)
#
# **Run once.** Converts MMSA's `aligned_50.pkl` (4.7 GB) into `mosei_compact.npz` (about 0.55 GB), which every other
# notebook loads in seconds.
#
# 1. Turn on *Internet*. Paste the Google Drive link of `aligned_50.pkl` into `GDRIVE_URL` (or attach the pkl).
# 2. Run all. Then *Save Version* → open the version's *Output* → *New Dataset* from `mosei_compact.npz`.
# 3. Attach that dataset to notebooks 01–07.
#
# | | |
# | --- | --- |
# | Labels used | 1: sentiment score in [−3, 3] (the MMSA file has no emotion labels) |
# | Splits | 16,326 train / 1,871 validation / 4,659 test |
# | Text | BERT 50 × 768 → PCA to 128 dims (fitted on train) |
# | Audio | COVAREP 50 × 74, standardised with train statistics |
# | Video | FACET 50 × 35, standardised with train statistics |

# %% [include] _setup.py

# %%
from fusionlab import config, data
import numpy as np

GDRIVE_URL = ''          # link to the FILE aligned_50.pkl, or leave empty if the pkl / compact file is attached
cfg = config.load_config(os.path.join(CONFIGS, '00_data_prep.yaml'))
D = data.load_mosei(cfg, GDRIVE_URL)

# %% [markdown]
# ## Numbers to report

# %%
import pandas as pd
rows = []
for s in ('train', 'valid', 'test'):
    y = D[s]['y'].numpy()
    rows.append(dict(split=s, clips=len(y), label_mean=y.mean(), label_std=y.std(), positive=(y > 0).mean(),
                     neutral=(y == 0).mean(), negative=(y < 0).mean(),
                     text_tokens=D[s]['tm'].float().sum(1).mean().item(),
                     audio_frames=D[s]['am'].float().sum(1).mean().item(),
                     video_frames=D[s]['vm'].float().sum(1).mean().item()))
stats_df = pd.DataFrame(rows).round(3)
os.makedirs(os.path.join(cfg['out_dir'], 'tables'), exist_ok=True)
stats_df.to_csv(os.path.join(cfg['out_dir'], 'tables', 'dataset_stats.csv'), index=False)
stats_df
