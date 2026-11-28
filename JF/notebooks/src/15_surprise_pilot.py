# %% [markdown]
# # 15 · Pilot: surprise-tempered product of experts (CPU, no retraining)
#
# Each expert's belief is compared with what the OTHER experts predict for the same target latent (the cross-modal
# prediction that V-JEPA training already learns). An expert that is unusually surprising has its precision scaled
# down inside the product, so the prediction leans on the others and the fused uncertainty rises.
# Thresholds come from clean training clips; the strength k is either fixed or tuned on validation clips with
# self-generated corruptions (noise, random swap). Held-out at test: noise λ = 0.5, dead sensor, two noisy inputs.
#
# Runs on **MOSEI** (attach 01, 03, 07) and/or **MS-COCO** (attach 10). Datasets whose inputs are missing are skipped.

# %% [include] _setup.py

# %%
import glob, time
import numpy as np, pandas as pd, torch
from fusionlab import BASE, coco, config, data, pretrain, runner, surprise as S
from fusionlab.runtime import RT

DATASETS = {'mosei': ('12_mosei_gate.yaml', 'main', ['vjepa', 'vjepa_vt', 'align']),
            'coco': ('10_coco_cxf.yaml', 'coco', ['vjepa', 'align'])}
KS = [0.5, 1, 2, 4]
rows = []


def load(ds, cfgname):
    cfg = config.load_config(os.path.join(CONFIGS, cfgname))
    if ds == 'mosei':
        data.load_mosei(cfg)
    else:
        hits = glob.glob('/kaggle/input/**/' + coco.COMPACT, recursive=True)
        if not hits:
            return None
        data.load_npz(cfg, hits[0])
    return cfg


for ds, (cfgname, group, objectives) in DATASETS.items():
    cfg = load(ds, cfgname)
    if cfg is None:
        print('skip', ds, '(no data attached)'); continue
    probe = f'v2_model_{group}_vjepa_frozen_poe_s0.pt'
    if not cfg['quick'] and not pretrain.find_ckpt(probe):
        print('skip', ds, '(attach its model outputs)'); continue
    feat_mean = {m: RT.DATA['train'][m][RT.DATA['train'][m + 'm']].mean(0).to(RT.DEV) for m in BASE}
    perm = {s: torch.randperm(RT.N_SPLIT[s], generator=torch.Generator().manual_seed(123 + 7)) for s in RT.N_SPLIT}
    for obj in objectives:
        for seed in cfg['seeds']:
            if not cfg['quick'] and not pretrain.find_ckpt(f'v2_model_{group}_{obj}_frozen_poe_s{seed}.pt'):
                continue
            t0 = time.time()
            experts, _, marg = pretrain.pretrain_or_load(obj, BASE, group, seed, cfg)
            model, _ = runner.build_model(experts, obj, 'frozen', 'poe', BASE, cfg, seed, marg, group)
            clean = dict(name='clean', kind='clean')
            mu_tr, lv_tr, _ = S.beliefs(model, 'train', clean, perm['train'], feat_mean)
            det = S.Surprise().fit(mu_tr, lv_tr)
            # validation: clean for the temperature, self-generated corruptions for tuning k
            val = {c['name']: S.beliefs(model, 'valid', c, perm['valid'], feat_mean)
                   for c in [clean] + [dict(name=f'noise[{m}]', kind='noise', m=m, lam=1.0) for m in BASE]
                   + [dict(name=f'swap[{m}]', kind='swap', m=m) for m in BASE]}
            def tau_for(w_fn):
                mu, lv, y = val['clean']
                m_, v_ = S.fused(model, mu, lv, w_fn(mu, lv))
                return float(np.mean((y - m_) ** 2 / v_))
            variants = {'poe': lambda mu, lv: None}
            for k in KS:
                variants[f'stp-k{k}'] = (lambda kk: lambda mu, lv: det.weights(mu, lv, kk))(k)
            taus = {n: tau_for(f) for n, f in variants.items()}
            # tune k on validation with self-generated corruptions (mean NLL over clean + corrupted)
            best = None
            for k in KS:
                nl = []
                for name, (mu, lv, y) in val.items():
                    m_, v_ = S.fused(model, mu, lv, det.weights(mu, lv, k))
                    nl.append(S.metrics(y, m_, v_, taus[f'stp-k{k}'])['NLL'])
                best = min(best or (1e9, k), (np.mean(nl), k))
            variants['stp-tuned'] = variants[f'stp-k{best[1]}']; taus['stp-tuned'] = taus[f'stp-k{best[1]}']
            clean_ref = {}
            for cd in S.conditions():
                mu, lv, y = S.beliefs(model, 'test', cd, perm['test'], feat_mean)
                z = det.z(mu, lv).numpy()
                for name, fn in variants.items():
                    m_, v_ = S.fused(model, mu, lv, fn(mu, lv))
                    r = dict(dataset=ds, objective=obj, seed=seed, cond=cd['name'], kind=cd['kind'],
                             heldout=not (cd['kind'] == 'swap' or (cd['kind'] == 'noise' and cd.get('lam') == 1.0)),
                             method=name, k=best[1], **S.metrics(y, m_, v_, taus[name]))
                    sd = np.sqrt(v_ * taus[name])
                    if cd['kind'] == 'clean':
                        clean_ref[name] = (sd, r['MAE'], r['SD'], z.max(1))
                    else:
                        r['AUROC_sd'] = S.auroc(clean_ref[name][0], sd)
                        r['AUROC_surprise'] = S.auroc(clean_ref[name][3], z.max(1))
                        r['TR'] = (r['SD'] / clean_ref[name][2]) / (r['MAE'] / clean_ref[name][1])
                        if 'm' in cd:
                            r['loc_acc'] = np.mean(z.argmax(1) == BASE.index(cd['m']))
                    rows.append(r)
            pd.DataFrame(rows).to_csv(os.path.join(cfg['out_dir'], 'surprise_pilot.csv'), index=False)
            print(ds, obj, seed, f'k*={best[1]}', f'{time.time() - t0:.0f}s', flush=True)

# %%
R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
c = R[R.kind != 'clean']
g = lambda d, col: d.groupby(['dataset', 'objective', 'method', 'seed'])[col].mean().groupby(['dataset', 'objective', 'method']).mean()
B = pd.DataFrame({'clean_NLL': g(R[R.kind == 'clean'], 'NLL'), 'clean_MAE': g(R[R.kind == 'clean'], 'MAE'),
                  'seen_NLL': g(c[~c.heldout], 'NLL'), 'held_NLL': g(c[c.heldout], 'NLL'),
                  'corr_MAE': g(c, 'MAE'), 'corr_Cov90': g(c, 'Cov90'), 'TR': g(c, 'TR'),
                  'AUROC_sd': g(c, 'AUROC_sd'), 'AUROC_surprise': g(c, 'AUROC_surprise'), 'loc_acc': g(c, 'loc_acc')})
print(B.round(3).to_string())

# %%
# per corruption kind (NLL), poe vs tuned
print(c[c.method.isin(['poe', 'stp-tuned'])].groupby(['dataset', 'objective', 'kind', 'method']).NLL.mean()
      .unstack('method').round(3).to_string())
