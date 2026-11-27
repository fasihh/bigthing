"""Prediction under each test condition, decoders (D0-D3m), and all metrics.

Decoders (calibration always on the validation split, pooled over the notebook's conditions):
  D0  point decoder, constant +/- width                  D1  head mean, constant +/- width
  D2  head mean +/- z_p * sigma (raw)                    D3a +/- z_p * tau * sigma (post-hoc rescaling)
  D3b +/- q_p * sigma (normalised split conformal)       D3c as D3b but per condition (oracle reference)
  D3m as D3b but one quantile per modality-availability pattern (Mondrian conformal, as in MCCF, 2026)
"""
import math

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from .batching import batches, get_batch
from .fusion import poe
from .models import expert_beliefs
from .runtime import RT, set_seed
from .stage2 import fused_output, me_beliefs

CAL_GRID = np.linspace(0.1, 0.9, 9)
COV_GRID = np.linspace(0.05, 1.0, 20)


def levels():
    return RT.CFG['levels']


def z_lev(p):
    return stats.norm.ppf(0.5 + p / 2)


def pairwise_skl(mus, lvs, av):
    """Mean symmetric KL between every pair of available experts' beliefs (disagreement)."""
    tot = torch.zeros(mus.shape[0], device=RT.DEV)
    cnt = torch.zeros_like(tot)
    for i in range(mus.shape[1]):
        for j in range(i + 1, mus.shape[1]):
            vi, vj = lvs[:, i].exp(), lvs[:, j].exp()
            skl = 0.25 * (vi / vj + vj / vi - 2 + (mus[:, i] - mus[:, j]) ** 2 * (1 / vi + 1 / vj)).sum(-1)
            w = av[:, i] * av[:, j]
            tot += skl * w
            cnt += w
    return tot / cnt.clamp(min=1)


def head_spread(experts, b, masks=None, xs=None):
    """Predictor-ensemble epistemic variance per expert: variance of the K head means, averaged over dims."""
    out = []
    for m in experts:
        e = experts[m]
        if getattr(e, 'K', 1) > 1:
            x = xs[m] if xs and m in xs else b[m]
            mk = masks[m] if masks and m in masks else b[m + 'm']
            mu_k, _ = e.heads(e.backbone(x, mk))
            out.append(mu_k.var(0, unbiased=False).mean(-1))
        else:
            out.append(torch.zeros(b['y'].shape[0], device=RT.DEV))
    return torch.stack(out, 1)


@torch.no_grad()
def predict(model, split, cond, seed=123, me=False):
    experts, fusion, head, d0, mods = (model[k] for k in ('experts', 'fusion', 'head', 'd0', 'mods'))
    for e in (experts, fusion, head, d0):
        e.eval()
    set_seed(seed)                                  # S-JEPA prior samples are drawn from the global generator
    g = torch.Generator(device=RT.DEV).manual_seed(seed)
    gm = torch.Generator(device=RT.DEV).manual_seed(seed + 1)
    n = RT.N_SPLIT[split]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 7)) if cond.get('swap') else None
    av1 = torch.tensor(cond['avail'], device=RT.DEV, dtype=torch.float32)
    keys = ['m', 'v', 'vl', 'm0', 'esd', 'share', 'dis', 'epi', 'pe']
    out = {k: [] for k in keys}
    FM, FV = [], []
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(n, i + 1024))
        b = get_batch(split, idx, mods, g)
        masks, xs = {}, {}
        if cond.get('text_drop'):
            tm = b['tm']
            keep = (torch.rand(tm.shape, device=RT.DEV, generator=g) > cond['text_drop']) & tm
            empty = keep.sum(1) == 0
            keep[empty] = tm[empty] & (torch.cumsum(tm.int(), 1) == 1)[empty]
            masks['t'] = keep
        for m, lam in cond.get('noise', {}).items():
            eps = torch.randn(b[m].shape, device=RT.DEV, generator=g) * RT.FEAT_STD[m]
            xs[m] = (math.sqrt(1 - lam) * b[m] + math.sqrt(lam) * eps) * b[m + 'm'].unsqueeze(-1)
        for m in cond.get('swap', ()):
            j = perm[idx]
            xs[m] = RT.DATA[split][m][j].to(RT.DEV)
            masks[m] = RT.DATA[split][m + 'm'][j].to(RT.DEV)
        if me:
            mus, lvs, epi = me_beliefs(experts, b, RT.CFG['me_K'], RT.CFG['me_ratio'], gm, masks, xs)
        else:
            mus, lvs = expert_beliefs(experts, b, masks, xs)
            epi = torch.zeros_like(mus)
        av = av1.expand(len(idx), -1)
        m_, v_, vl, fm, fv, share = fused_output(fusion, head, mus, lvs, av)
        out['m'].append(m_); out['v'].append(v_); out['vl'].append(vl)
        out['m0'].append(d0(fm).squeeze(-1))
        out['esd'].append(torch.exp(0.5 * lvs).mean(-1))
        out['share'].append(share)
        out['dis'].append(pairwise_skl(mus, lvs, av))
        out['epi'].append((epi * av.unsqueeze(-1)).mean(-1))
        out['pe'].append(head_spread(experts, b, masks, xs) * av)
        FM.append(fm)
        FV.append(fv if fv is not None else torch.full_like(fm, float('nan')))
    res = {k: torch.cat(v).cpu().numpy().astype(np.float64) for k, v in out.items()}
    res['y'] = RT.DATA[split]['y'].numpy().astype(np.float64)
    res['fm'], res['fv'] = torch.cat(FM), torch.cat(FV)
    return res


# ----------------------------------------------------------------------------- metrics
def conf_q(scores, p):
    s = np.sort(scores)
    k = min(len(s) - 1, int(math.ceil((len(s) + 1) * p)) - 1)
    return s[k]


def calibrate(vals, conds):
    """vals: {cond name: validation prediction}. Pooled calibration + per-availability (Mondrian) quantiles."""
    L = levels()
    y = np.concatenate([v['y'] for v in vals.values()])
    m = np.concatenate([v['m'] for v in vals.values()])
    v_ = np.concatenate([v['v'] for v in vals.values()])
    m0 = np.concatenate([v['m0'] for v in vals.values()])
    z = np.abs(y - m) / np.sqrt(v_)
    cal = dict(tau=math.sqrt(np.mean(z ** 2)), sc0=math.sqrt(np.mean((y - m0) ** 2)),
               sc1=math.sqrt(np.mean((y - m) ** 2)),
               qabs0={p: conf_q(np.abs(y - m0), p) for p in L}, qabs1={p: conf_q(np.abs(y - m), p) for p in L},
               qn={p: conf_q(z, p) for p in L}, qn_mask={})
    by_mask = {}
    for cd in conds:
        by_mask.setdefault(cd['avail'], []).append(vals[cd['name']])
    for mask, vs in by_mask.items():
        zz = np.concatenate([np.abs(v['y'] - v['m']) / np.sqrt(v['v']) for v in vs])
        cal['qn_mask'][mask] = {p: conf_q(zz, p) for p in L}
    return cal


def point_metrics(y, m):
    nz = y != 0
    return dict(MAE=np.mean(np.abs(y - m)), Corr=stats.pearsonr(y, m)[0] if np.std(m) > 1e-9 else 0.0,
                Acc2=np.mean((y[nz] > 0) == (m[nz] > 0)), Acc2_has0=np.mean((y >= 0) == (m >= 0)),
                Acc7=np.mean(np.round(np.clip(y, -3, 3)) == np.round(np.clip(m, -3, 3))),
                F1=f1_score(y[nz] > 0, m[nz] > 0, average='weighted'))


def prob_metrics(y, m, sd):
    z = (y - m) / sd
    cov = [np.mean(np.abs(z) <= z_lev(p)) for p in CAL_GRID]
    return dict(NLL=np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2),
                CRPS=np.mean(sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))),
                PIT_KS=stats.kstest(stats.norm.cdf(z), 'uniform').statistic,
                CalErr=np.mean(np.abs(np.array(cov) - CAL_GRID)),
                z_mean=np.mean(z), z_var=np.var(z), frac_absz_gt2=np.mean(np.abs(z) > 2), MeanSD=np.mean(sd))


def interval_metrics(y, m, hw):
    out = {}
    for p in levels():
        h = np.broadcast_to(hw[p], y.shape)
        out[f'Cov{int(p * 100)}'] = np.mean(np.abs(y - m) <= h)
        out[f'Width{int(p * 100)}'] = np.mean(2 * h)
    h = np.broadcast_to(hw[0.9], y.shape)
    lo, hi = m - h, m + h
    out['IS90'] = np.mean((hi - lo) + 20 * (lo - y) * (y < lo) + 20 * (y - hi) * (y > hi))
    return out


def failure_metrics(y, m, unc):
    err = np.abs(y - m)
    nz = y != 0
    order = np.argsort(unc, kind='stable')
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    curve = [float(risk[max(0, int(round(cv * len(err))) - 1)]) for cv in COV_GRID]
    oracle = np.cumsum(np.sort(err)) / np.arange(1, len(err) + 1)
    big = err > 1.0
    out = dict(Spearman_unc_err=stats.spearmanr(unc, err)[0],
               AUROC_bigerr=roc_auc_score(big, unc) if 0 < big.sum() < len(big) else np.nan,
               AURC=risk.mean(), E_AURC=risk.mean() - oracle.mean())
    for cv in (0.9, 0.8, 0.7, 0.5):
        keep = order[:int(round(cv * len(y)))]
        k_nz = keep[nz[keep]]
        out[f'selMAE@{int(cv * 100)}'] = err[keep].mean()
        out[f'selAcc2@{int(cv * 100)}'] = np.mean((y[k_nz] > 0) == (m[k_nz] > 0))
    return out, curve


def sign_or_defer(y, m, hw90):
    h = np.broadcast_to(hw90, y.shape)
    ans = (m - h > 0) | (m + h < 0)
    sel = ans & (y != 0)
    return dict(defer_answer_rate=ans.mean(),
                defer_acc2_answered=np.mean((y[sel] > 0) == (m[sel] > 0)) if sel.any() else np.nan)


def ppc(y, m, sd, R, seed=0):
    rng = np.random.default_rng(seed)
    t1, t2 = np.mean(np.abs(y - m)), np.mean(np.abs(y - m) > 1.645 * sd)
    yr = rng.standard_normal((R, len(y))) * sd
    return dict(PPC_p_MAE=np.mean(np.abs(yr).mean(1) >= t1), PPC_p_out90=np.mean((np.abs(yr) > 1.645 * sd).mean(1) >= t2))


@torch.no_grad()
def latent_maha(model, fm, fv, split):
    """Mahalanobis^2 / d of the target under the fused latent belief (frozen experts with an EMA target only)."""
    if model.get('target') is None or torch.isnan(fv).any():
        return {}
    Z, V = [], []
    for _, b in batches(split, 1024, False, model['mods'], torch.Generator(device=RT.DEV).manual_seed(1)):
        z, vt = model['target'](b['t'], b['tm'])
        Z.append(z); V.append(vt)
    Z, V = torch.cat(Z), torch.cat(V)
    d2 = (((Z - fm) ** 2) / (fv + V)).sum(-1).cpu().numpy()
    d = RT.CFG['d_lat']
    return dict(Maha_over_d=d2.mean() / d, LatCov90=np.mean(d2 <= stats.chi2.ppf(0.9, d)))


def evaluate_model(model, tag, conds, use_me=False):
    """Returns result rows (one per condition x decoder x uncertainty source) and calibration curves."""
    L = levels()
    rows, curves = [], []
    for unc in (['base', 'me'] if use_me else ['base']):
        me = unc == 'me'
        vals = {cd['name']: predict(model, 'valid', cd, me=me) for cd in conds}
        cal = calibrate(vals, conds)
        clean = None
        tests = {cd['name']: predict(model, 'test', cd, me=me) for cd in conds}
        if 'TAV' in tests:
            t0 = tests['TAV']
            clean = dict(dis=t0['dis'], sd=np.sqrt(t0['v']), epi=t0['epi'].sum(1), pe=t0['pe'].sum(1),
                         esd_t=t0['esd'][:, 0])
        for cd in conds:
            te = tests[cd['name']]
            y, m, sd, m0 = te['y'], te['m'], np.sqrt(te['v']), te['m0']
            vv = vals[cd['name']]
            qn_or = {p: conf_q(np.abs(vv['y'] - vv['m']) / np.sqrt(vv['v']), p) for p in L}
            qm = cal['qn_mask'][cd['avail']]
            ones = np.ones_like(y)
            decs = {'D0': (m0, cal['sc0'] * ones, {p: cal['qabs0'][p] for p in L}),
                    'D1': (m, cal['sc1'] * ones, {p: cal['qabs1'][p] for p in L}),
                    'D2': (m, sd, {p: z_lev(p) * sd for p in L}),
                    'D3a': (m, sd * cal['tau'], {p: z_lev(p) * sd * cal['tau'] for p in L}),
                    'D3b': (m, None, {p: cal['qn'][p] * sd for p in L}),
                    'D3c': (m, None, {p: qn_or[p] * sd for p in L}),
                    'D3m': (m, None, {p: qm[p] * sd for p in L})}
            base = dict(tag, unc=unc, cond=cd['name'], kind=cd['kind'], lam=cd.get('lam', 0.0),
                        corrupted=cd.get('corrupted', ''), text_drop=cd.get('text_drop', 0.0),
                        tau=cal['tau'], disagree=te['dis'].mean(),
                        latent_var_share=np.mean(np.minimum(te['vl'] * model['head'].log_alpha.exp().item()
                                                            / te['v'], 1.0)))
            for k, mm in enumerate(model['mods']):
                base[f'sd_{mm}'] = te['esd'][:, k].mean()
                base[f'share_{mm}'] = te['share'][:, k].mean()
                base[f'epi_{mm}'] = te['epi'][:, k].mean()
                base[f'pe_{mm}'] = te['pe'][:, k].mean()
            per = {}
            for dn, (mean, sdd, hw) in decs.items():
                r = dict(base, decoder=dn, **point_metrics(y, mean), **interval_metrics(y, mean, hw))
                if sdd is not None:
                    r.update(prob_metrics(y, mean, sdd))
                per[dn] = r
            fd2, curve_sd = failure_metrics(y, m, sd)
            per['D2'].update(fd2)
            per['D2'].update(ppc(y, m, sd, RT.CFG['ppc_reps']))
            per['D3b'].update(sign_or_defer(y, m, cal['qn'][0.9] * sd))
            if cd['kind'] in ('subset', 'subset+N'):
                per['D2'].update(latent_maha(model, te['fm'], te['fv'], 'test'))
            if cd['kind'] in ('noise', 'swap', 'textdrop') and clean is not None:
                lab = np.r_[np.zeros(len(clean['sd'])), np.ones(len(sd))]
                per['D2']['det_AUROC_sd'] = roc_auc_score(lab, np.r_[clean['sd'], sd])
                per['D2']['det_AUROC_dis'] = roc_auc_score(lab, np.r_[clean['dis'], te['dis']])
                per['D2']['det_AUROC_esd_t'] = roc_auc_score(lab, np.r_[clean['esd_t'], te['esd'][:, 0]])
                if me:
                    per['D2']['det_AUROC_epi'] = roc_auc_score(lab, np.r_[clean['epi'], te['epi'].sum(1)])
                if np.any(clean['pe'] > 0):
                    per['D2']['det_AUROC_pe'] = roc_auc_score(lab, np.r_[clean['pe'], te['pe'].sum(1)])
            rows += list(per.values())
            if cd['kind'] in ('subset', 'subset+N'):
                z2, z3 = (y - m) / sd, (y - m) / (sd * cal['tau'])
                curves.append(dict(tag, unc=unc, cond=cd['name'],
                                   pit_D2=np.histogram(stats.norm.cdf(z2), bins=10, range=(0, 1))[0].tolist(),
                                   pit_D3a=np.histogram(stats.norm.cdf(z3), bins=10, range=(0, 1))[0].tolist(),
                                   rc_sd=curve_sd))
    return rows, curves


# ----------------------------------------------------------------------------- probes (Huang-style)
@torch.no_grad()
def collect_latents(model, split):
    experts, mods = model['experts'], model['mods']
    experts.eval()
    set_seed(77)
    Z, E, Y, DP = [], [], [], []
    for idx, b in batches(split, 1024, False, mods, torch.Generator(device=RT.DEV).manual_seed(77)):
        mus, lvs = expert_beliefs(experts, b)
        Z.append(poe(mus, lvs, torch.ones(len(idx), len(mods), device=RT.DEV))[0])
        E.append(mus); Y.append(b['y'])
        if 'dpool' in b:
            DP.append(b['dpool'])
    return torch.cat(Z), torch.cat(E), torch.cat(Y)[:, None], (torch.cat(DP) if DP else None)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    one = lambda X: torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=RT.DEV)], 1)
    A, Bt = one(Xtr), one(Xte)
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=RT.DEV), A.T @ Ytr)
    P = Bt @ W
    return (1 - ((Yte - P) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)).mean().item()


def probe_metrics(model):
    """How much of the sentiment (signal) and of the distractor each latent keeps (ridge R^2 on test)."""
    Ztr, Etr, Ytr, Dtr = collect_latents(model, 'train')
    Zte, Ete, Yte, Dte = collect_latents(model, 'test')
    out = {'probe_signal_R2': ridge_r2(Ztr, Ytr, Zte, Yte)}
    for k, m in enumerate(model['mods']):
        out[f'probe_signal_R2_{m}'] = ridge_r2(Etr[:, k], Ytr, Ete[:, k], Yte)
    if Dtr is not None:
        out['probe_noise_R2'] = ridge_r2(Ztr, Dtr, Zte, Dte)
        for k, m in enumerate(model['mods']):
            out[f'probe_noise_R2_{m}'] = ridge_r2(Etr[:, k], Dtr, Ete[:, k], Dte)
    return out
