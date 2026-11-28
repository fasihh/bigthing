# %% [markdown]
# # 08 · Final scoreboard (all notebooks)
#
# Collects every `headline_*.csv`, `scoreboard_*.csv` and `fig_*.png` from the outputs of notebooks 01–07
# (attach them as inputs) and writes one file, `final_numbers.md`, with every number the report quotes.
# CPU only; no training.

# %% [include] _setup.py

# %%
import glob, shutil
import pandas as pd
from IPython.display import Image, Markdown, display

OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else './out_08'
ROOTS = ['/kaggle/input', os.path.join(JF, 'results')]
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)


def find(pattern):
    hits = {}
    for r in ROOTS:
        for p in glob.glob(os.path.join(r, '**', pattern), recursive=True):
            hits.setdefault(os.path.basename(p), p)          # first copy of each file wins
    return dict(sorted(hits.items()))


heads, boards, figs = find('headline_*.csv'), find('scoreboard_*.csv'), find('fig_*.png')
print(len(heads), 'headline files,', len(boards), 'scoreboards,', len(figs), 'figures')

# %%
TITLES = {'02': 'Huang (2026) reproduction with new seeds', '03': 'Missing modalities',
          '04': 'Corrupted and mismatched modalities', '05': 'Multimodal noisy TV (noise channel, distractor)',
          '06': 'Downstream decoders and calibration', '07': 'Novel methods'}
md = ['# Final numbers\n', 'Mean ± standard deviation over seeds unless stated. `*` marks the best value in a column.\n']
for name, p in heads.items():
    nb = name.split('_')[1].split('.')[0]
    md += [f'\n## {nb} · {TITLES.get(nb, "")}\n', pd.read_csv(p).to_markdown(index=False), '\n']
    for bn, bp in boards.items():
        if bn.startswith(f'scoreboard_{nb}') and not bn.endswith('_raw.csv'):
            md += [f'\n**{bn.replace(".csv", "")}**\n', pd.read_csv(bp).to_markdown(index=False), '\n']
    for fn, fp in figs.items():
        if fn.startswith(f'fig_{nb}'):
            shutil.copy(fp, os.path.join(OUT, 'figures', fn))
            md.append(f'\n![{fn}](figures/{fn})\n')
open(os.path.join(OUT, 'final_numbers.md'), 'w').write('\n'.join(md))
display(Markdown('\n'.join(md[:60])))
