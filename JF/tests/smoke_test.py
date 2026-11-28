"""Crash check only: tiny epochs on placeholder MOSEI-shaped arrays. The numbers it prints are NOT results.
Usage: python tests/smoke_test.py /path/to/placeholder_or_real/aligned_50.pkl
"""
import sys
import time, pandas as pd
from fusionlab import config, data, batching, runner, report, BASE
cfg = config.load_config(None, quick=True, out_dir='./smoke_out', pkl_path=sys.argv[1],
                         text_drops=[0.5], noise_lambdas=[1.0])
data.load_mosei(cfg)
conds = batching.conditions(BASE, ['subset','noise','swap'], cfg)
runner.run_grid('smoke', BASE, ['sjepa','recon','align','none'], conds, frozen=['poe','monig','mpoe'], finetuned=['poe'], supervised=['concat'], probes=True)
NM = BASE+['n']
runner.run_grid('nc', NM, ['vjepa'], batching.subset_conditions(NM), frozen=['poe','moe'], pgroup='noisech')
env = dict(pattern='TAV_shared', sigma=8.0, shared=True, mods=list(BASE))
runner.run_grid('dist', BASE, ['vjepa'], batching.subset_conditions(BASE), frozen=['poe'], env=env, pgroup='dist', probes=True)
# reuse check: second call must load, not retrain
t=time.time(); runner.run_grid('reuse', BASE, ['sjepa'], conds[:2], frozen=['poe']); print('reuse took', round(time.time()-t,1),'s')
R = pd.read_csv('./smoke_out/results.csv'); print(R.shape)
print(R[R.decoder=='D2'].groupby(['grid','objective','regime','fusion'])[['MAE','Cov90','MeanSD']].mean().round(3))
print(report.scoreboard(R[(R.decoder=='D2')&(R.grid=='smoke')], ['MAE','Cov90','NLL'], by=['objective','fusion']))
print(report.tracking(R[(R.decoder=='D2')&(R.grid=='smoke')]).head())
