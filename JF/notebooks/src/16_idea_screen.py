# %% [markdown]
# # 16 · Idea screen: learned redundancy checks between experts (CPU, no retraining)
#
# Notebook 15 found that "down-weight the expert that disagrees" only works when the experts share an aligned
# space. Here the clean relations between experts are LEARNED from data (ideas from sensor fault diagnosis), and
# four ways of acting on a flagged expert are compared. Detectors: loo (15's baseline), parity (ridge prediction of
# one expert from the others, whitened residual), spe (PCA residual share), self (own-belief density), max
# (parity or self). Actions: temper, widen, impute, temper+widen. Strengths are tuned on validation clips with
# self-made corruptions (noise, random swap); held-out at test: noise λ = 0.5, dead sensor, two noisy inputs.
#
# Runs on **MOSEI** (attach 01, 03, 07) and/or **MS-COCO** (attach 10). Datasets whose inputs are missing are skipped.

# %% [include] _setup.py

# %%
import glob, time
import numpy as np, pandas as pd, torch
from fusionlab import BASE, coco, config, data, pretrain, runner, surprise as S, screen16 as X
from fusionlab.runtime import RT

DATASETS = {'mosei': ('12_mosei_gate.yaml', 'main', ['vjepa', 'vjepa_vt', 'align']),
            'coco': ('10_coco_cxf.yaml', 'coco', ['vjepa', 'align'])}
KS, AS = [1, 2, 4], [0.5, 1, 2, 4]
rows = []


def load(ds, cfgname):
    cfg = config.load_config(os.path.join(CONFIGS, cfgname))
    if ds == 'mosei':
        data.load_mosei(cfg)
    else:
        hits = glob.glob('/kaggle/input/**/' + coco.COMPACT, recursive=True) + glob.glob('/tmp/**/' + coco.COMPACT, recursive=True)
        if not hits:
            return None
        data.load_npz(cfg, hits[0])
    return cfg


def predict(model, mus, lvs, Z, base, spec):
    det, act, k, a = spec
    if act == 'poe':
        return base
    z, c = Z[det]
    if act in ('temper', 'temper+widen'):
        m_, v_ = S.fused(model, mus, lvs, X.temper_w(z, c, k))
    elif act == 'impute':
        m_, v_ = S.fused(model, *X.impute(mus, lvs, z, c, dets.parity))
    else:
        m_, v_ = base
    if act in ('widen', 'temper+widen'):
        v_ = v_ * X.widen_factor(z, c, a)
    return m_, v_


def valid_score(model, val, spec):
    """(tau from clean valid, mean NLL over clean + self-made corruptions)."""
    outs = {n: predict(model, mu, lv, Z, base, spec) for n, (mu, lv, y, Z, base) in val.items()}
    y = val['clean'][2]
    tau = float(np.mean((y - outs['clean'][0]) ** 2 / outs['clean'][1]))
    return tau, np.mean([S.metrics(val[n][2], m_, v_, tau)['NLL'] for n, (m_, v_) in outs.items()])


for ds, (cfgname, group, objectives) in DATASETS.items():
    cfg = load(ds, cfgname)
    if cfg is None:
        print('skip', ds, '(no data attached)'); continue
    if not cfg['quick'] and not pretrain.find_ckpt(f'v2_model_{group}_vjepa_frozen_poe_s0.pt'):
        print('skip', ds, '(attach its model outputs)'); continue
    feat_mean = {m: RT.DATA['train'][m][RT.DATA['train'][m + 'm']].mean(0).to(RT.DEV) for m in BASE}
    perm = {s: torch.randperm(RT.N_SPLIT[s], generator=torch.Generator().manual_seed(130)) for s in RT.N_SPLIT}
    for obj in objectives:
        for seed in cfg['seeds']:
            if not cfg['quick'] and not pretrain.find_ckpt(f'v2_model_{group}_{obj}_frozen_poe_s{seed}.pt'):
                continue
            t0 = time.time()
            experts, _, marg = pretrain.pretrain_or_load(obj, BASE, group, seed, cfg)
            model, _ = runner.build_model(experts, obj, 'frozen', 'poe', BASE, cfg, seed, marg, group)
            clean = dict(name='clean', kind='clean')
            mu_tr, lv_tr, _ = S.beliefs(model, 'train', clean, perm['train'], feat_mean)
            dets = X.Detectors().fit(mu_tr, lv_tr)
            val = {}
            for c_ in [clean] + [dict(name=f'noise[{m}]', kind='noise', m=m, lam=1.0) for m in BASE] \
                    + [dict(name=f'swap[{m}]', kind='swap', m=m) for m in BASE]:
                mu, lv, y = S.beliefs(model, 'valid', c_, perm['valid'], feat_mean)
                val[c_['name']] = (mu, lv, y, dets.z(mu, lv), S.fused(model, mu, lv))
            # tune each detector x action on validation
            specs, taus = {'poe': ('loo', 'poe', 0, 0)}, {}
            taus['poe'] = valid_score(model, val, specs['poe'])[0]
            for d in X.Detectors.NAMES:
                best_k = min(KS, key=lambda k: valid_score(model, val, (d, 'temper', k, 0))[1])
                best_a = min(AS, key=lambda a: valid_score(model, val, (d, 'widen', 0, a))[1])
                best_ka = min(AS, key=lambda a: valid_score(model, val, (d, 'temper+widen', best_k, a))[1])
                for name, sp in [(f'{d}|temper', (d, 'temper', best_k, 0)), (f'{d}|widen', (d, 'widen', 0, best_a)),
                                 (f'{d}|impute', (d, 'impute', 0, 0)), (f'{d}|temper+widen', (d, 'temper+widen', best_k, best_ka))]:
                    specs[name] = sp
                    taus[name] = valid_score(model, val, sp)[0]
            ref = {}
            for cd in S.conditions():
                mu, lv, y = S.beliefs(model, 'test', cd, perm['test'], feat_mean)
                Z = dets.z(mu, lv)
                base = S.fused(model, mu, lv)
                for name, sp in specs.items():
                    m_, v_ = predict(model, mu, lv, Z, base, sp)
                    r = dict(dataset=ds, objective=obj, seed=seed, cond=cd['name'], kind=cd['kind'],
                             heldout=not (cd['kind'] == 'swap' or (cd['kind'] == 'noise' and cd.get('lam') == 1.0)),
                             method=name, det=sp[0], act=sp[1], k=sp[2], a=sp[3], **S.metrics(y, m_, v_, taus[name]))
                    sd = np.sqrt(v_ * taus[name])
                    zmax = (Z[sp[0]][0] / Z[sp[0]][1]).max(1).values.numpy()
                    if cd['kind'] == 'clean':
                        ref[name] = (sd, r['MAE'], r['SD'], zmax)
                    else:
                        r['AUROC_sd'] = S.auroc(ref[name][0], sd)
                        r['AUROC_det'] = S.auroc(ref[name][3], zmax)
                        r['TR'] = (r['SD'] / ref[name][2]) / (r['MAE'] / ref[name][1])
                        if 'm' in cd:
                            r['loc_acc'] = float(np.mean(Z[sp[0]][0].argmax(1).numpy() == BASE.index(cd['m'])))
                    rows.append(r)
            pd.DataFrame(rows).to_csv(os.path.join(cfg['out_dir'], 'idea_screen.csv'), index=False)
            print(ds, obj, seed, f'{time.time() - t0:.0f}s', flush=True)

# %%
R = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
c = R[R.kind != 'clean']
g = lambda d, col: d.groupby(['dataset', 'objective', 'method', 'seed'])[col].mean().groupby(['dataset', 'objective', 'method']).mean()
B = pd.DataFrame({'clean_NLL': g(R[R.kind == 'clean'], 'NLL'), 'seen_NLL': g(c[~c.heldout], 'NLL'),
                  'held_NLL': g(c[c.heldout], 'NLL'), 'corr_MAE': g(c, 'MAE'), 'Cov90': g(c, 'Cov90'), 'TR': g(c, 'TR'),
                  'AUROC_sd': g(c, 'AUROC_sd'), 'AUROC_det': g(c, 'AUROC_det'), 'loc': g(c, 'loc_acc')})
print(B.round(3).to_string())

# %%
# held-out NLL gain over poe (positive = better), per dataset/objective, best methods first
H = B['held_NLL'].unstack('method')
gain = (H.rsub(H['poe'], axis=0)).drop(columns='poe')
print(gain.round(3).T.assign(mean=lambda d: d.mean(1)).sort_values('mean', ascending=False).to_string())
print('clean NLL cost (method - poe):')
Cn = B['clean_NLL'].unstack('method')
print((Cn.sub(Cn['poe'], axis=0)).drop(columns='poe').round(3).T.assign(worst=lambda d: d.max(1)).sort_values('worst').to_string())

# %%
# per corruption kind (NLL) for poe and the three best by mean held-out gain
top = list(gain.mean().sort_values(ascending=False).index[:3])
print(c[c.method.isin(['poe'] + top)].groupby(['dataset', 'objective', 'kind', 'method']).NLL.mean()
      .unstack('method').round(3).to_string())
