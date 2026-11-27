"""Stage 1: label-free expert pretraining.

Objectives
  vjepa     : V-JEPA with a deterministic EMA text target (the project's main method)
  vjepa_vt  : variational target, closed-form expected NLL under the target distribution        (novel)
  vjepa_vts : variational target, one target sample per step                                    (novel)
  vjepa_pe  : vjepa with a predictor ensemble (K heads, bootstrap-weighted)                      (novel)
  sjepa     : S-JEPA, stochastic latent with learned prior and posterior (Giakoumoglou et al. 2026)
  recon     : reconstruction (MVAE with subset sampling)
  align     : contrastive alignment to the text latent (InfoNCE)
  none      : no pretraining (fully supervised experts, stage 2 only)
"""
import copy
import glob
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batching import batches, cosine_lr, pooled, random_subsets, random_token_mask
from .fusion import poe
from .models import Target, expected_nll, expert_beliefs, gauss_nll, make_experts
from .runtime import RT, set_seed

OBJECTIVES = ['vjepa', 'vjepa_vt', 'vjepa_vts', 'vjepa_pe', 'sjepa', 'recon', 'align', 'none']
HAS_TARGET = {'vjepa', 'vjepa_vt', 'vjepa_vts', 'vjepa_pe', 'sjepa', 'align'}


def vicreg_terms(h):
    h = h - h.mean(0)
    std = torch.sqrt(h.var(0) + 1e-4)
    cov = (h.T @ h) / (h.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return F.relu(1 - std).mean() + (off ** 2).sum() / h.size(1)


def _target_kind(obj):
    return 'vt' if obj in ('vjepa_vt', 'vjepa_vts') else 'det'


def _online(experts, obj):
    return experts['t'] if _target_kind(obj) == 'vt' else experts['t'].backbone


def pretrain(objective, mods, c, seed):
    set_seed(seed)
    DEV, M, d = RT.DEV, len(mods), c['d_lat']
    experts = make_experts(mods, c, objective).to(DEV)
    if objective == 'none':
        return experts.eval(), None
    target = Target(copy.deepcopy(_online(experts, objective)), _target_kind(objective)).to(DEV) \
        if objective not in ('recon', 'align') else None
    extra, stats_ = nn.ModuleDict(), {}
    if objective == 'recon':
        for m in mods:
            if m == 'n':
                stats_[m] = (torch.zeros(RT.DIMS[m], device=DEV),
                             torch.full((RT.DIMS[m],), 1 / math.sqrt(RT.LENS[m]), device=DEV))
            else:
                pv = pooled(RT.DATA['train'][m], RT.DATA['train'][m + 'm'])
                stats_[m] = (pv.mean(0).to(DEV), pv.std(0).to(DEV) + 1e-6)
        extra = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, RT.DIMS[m]))
                               for m in mods}).to(DEV)
    params = list(experts.parameters()) + list(extra.parameters())
    opt = torch.optim.AdamW(params, lr=c['pre_lr'], weight_decay=0.01)
    total = c['pre_epochs'] * math.ceil(RT.N_SPLIT['train'] / c['batch'])
    step = 0
    for ep in range(c['pre_epochs']):
        experts.train(); extra.train()
        tot, nb = 0.0, 0
        for idx, b in batches('train', c['batch'], True, mods):
            cosine_lr(opt, c['pre_lr'], step, total)
            B = len(idx)
            ctx = {m: random_token_mask(b[m + 'm'], c['mask_ratio']) for m in mods}
            if objective in ('vjepa', 'vjepa_vt', 'vjepa_vts'):
                loss = _loss_vjepa(objective, experts, target, b, ctx, B, M, c)
            elif objective == 'vjepa_pe':
                loss = _loss_vjepa_pe(experts, target, b, ctx, B, M, c)
            elif objective == 'sjepa':
                beta = c['sjepa_beta'] * min(1.0, step / max(1, c['sjepa_warmup'] * total))
                loss = _loss_sjepa(experts, target, b, ctx, beta, c)
            elif objective == 'align':
                loss = _loss_align(experts, b, ctx, B, mods, c)
            else:
                kl_w = c['beta_recon'] * min(1.0, step / max(1, c['kl_warmup'] * total))
                loss = _loss_recon(experts, extra, stats_, b, ctx, B, M, mods, kl_w, d)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if target is not None:
                mom = 1 - (1 - c['ema_start']) * (math.cos(math.pi * step / total) + 1) / 2
                target.ema_from(_online(experts, objective), mom)
            step += 1
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == c['pre_epochs'] - 1:
            print(f'    [pre {objective}{"+N" if "n" in mods else ""}] ep {ep:3d} loss {tot / nb:.4f}')
    if objective == 'align':
        target = Target(copy.deepcopy(experts['t'].backbone), 'det').to(DEV)
    return experts.eval(), target


def _loss_vjepa(obj, experts, target, b, ctx, B, M, c):
    mus, lvs = expert_beliefs(experts, b, ctx)
    z, vt = target(b['t'], b['tm'])
    if obj == 'vjepa_vts':
        z, vt = z + vt.sqrt() * torch.randn_like(z), torch.zeros_like(vt)
    nll = expected_nll(z.unsqueeze(1), vt.unsqueeze(1), mus, lvs.exp()).sum(-1).mean()
    kl = 0.5 * (mus ** 2 + lvs.exp() - lvs - 1).sum(-1).mean()
    fm, fv, _ = poe(mus, lvs, random_subsets(B, M))
    fused = expected_nll(z, vt, fm, fv).sum(-1).mean()
    h_full = experts['t'].backbone(b['t'], b['tm'])
    vic = vicreg_terms(h_full)
    if obj != 'vjepa':                     # the variational target is the predictor's own output: guard it too
        vic = vic + vicreg_terms(experts['t'].head(h_full)[0])
    return nll + c['beta_kl'] * kl + c['w_fused'] * fused + c['w_vic'] * vic


def _loss_vjepa_pe(experts, target, b, ctx, B, M, c):
    """Each head sees a bootstrap re-weighting of the batch, so the heads disagree where data is thin."""
    z, _ = target(b['t'], b['tm'])
    nll, kl, mus, lvs = 0.0, 0.0, [], []
    for m in experts:
        h = experts[m].backbone(b[m], ctx[m])
        mu_k, lv_k = experts[m].heads(h)                                    # (K, B, d)
        w = torch.poisson(torch.ones(mu_k.shape[0], B, device=RT.DEV))       # bootstrap weights
        nll = nll + (w * gauss_nll(z.unsqueeze(0), mu_k, lv_k.exp()).sum(-1)).mean()
        kl = kl + (0.5 * (mu_k ** 2 + lv_k.exp() - lv_k - 1).sum(-1)).mean()
        mu, lv = experts[m].head(h)
        mus.append(mu); lvs.append(lv)
    mus, lvs = torch.stack(mus, 1), torch.stack(lvs, 1)
    fm, fv, _ = poe(mus, lvs, random_subsets(B, M))
    fused = gauss_nll(z, fm, fv).sum(-1).mean()
    vic = vicreg_terms(experts['t'].backbone(b['t'], b['tm']))
    return nll + c['beta_kl'] * kl + c['w_fused'] * fused + c['w_vic'] * vic


def _loss_sjepa(experts, target, b, ctx, beta, c):
    """S-JEPA: E_q ||g(h, z) - sg(target)||^2 + beta * KL(q(z | h, target) || p(z | h))."""
    zt, _ = target(b['t'], b['tm'])
    loss = 0.0
    for m in experts:
        e = experts[m]
        h = e.backbone(b[m], ctx[m])
        qm, qlv = e.post_params(h, zt)
        pm, plv = e.prior_params(h)
        z = qm + (0.5 * qlv).exp() * torch.randn_like(qm)
        pred = e.predict(h, z)
        rec = ((pred - zt) ** 2).sum(-1).mean()
        kl = 0.5 * (plv - qlv + (qlv.exp() + (qm - pm) ** 2) / plv.exp() - 1).sum(-1).mean()
        loss = loss + rec + beta * kl
    return loss + c['w_vic'] * vicreg_terms(experts['t'].backbone(b['t'], b['tm']))


def _loss_align(experts, b, ctx, B, mods, c):
    d = c['d_lat']
    z = F.layer_norm(experts['t'].backbone(b['t'], b['tm']), (d,))
    zn, lab = F.normalize(z, dim=-1), torch.arange(B, device=RT.DEV)
    loss = 0.0
    for m in mods:
        h = experts[m].backbone(b[m], ctx[m])
        mu, lv = experts[m].head(h, detach_var=True)
        logits = F.normalize(mu, dim=-1) @ zn.T / c['align_temp']
        loss = loss + 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
        loss = loss + gauss_nll(z.detach(), mu.detach(), lv.exp()).sum(-1).mean()   # variance head only
    return loss


def _loss_recon(experts, extra, stats_, b, ctx, B, M, mods, kl_w, d):
    mus, lvs = expert_beliefs(experts, b, ctx)
    tgt = {m: (pooled(b[m], b[m + 'm']) - stats_[m][0]) / stats_[m][1] for m in mods}
    subsets = [torch.ones(B, M, device=RT.DEV)] + \
              [torch.eye(M, device=RT.DEV)[i].expand(B, M) for i in range(M)] + [random_subsets(B, M)]
    loss = 0.0
    for av in subsets:
        fm, fv, _ = poe(mus, lvs, av)
        zs = fm + fv.sqrt() * torch.randn_like(fm)
        rec = sum(0.5 * ((extra[m](zs) - tgt[m]) ** 2).sum(-1).mean() / RT.DIMS[m] * d for m in mods)
        kl = 0.5 * (fm ** 2 + fv - torch.log(fv) - 1).sum(-1).mean()
        loss = loss + rec + kl_w * kl
    return loss / len(subsets)


# ----------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def latent_stats(Z):
    Z = Z - Z.mean(0)
    std = Z.std(0)
    s = torch.linalg.svdvals(Z)
    p = s / s.sum()
    return dict(std_mean=std.mean().item(), std_min=std.min().item(),
                erank=torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())


@torch.no_grad()
def target_marginal(target, mods):
    """Per-dimension marginal of the target over the training split: N(mean z, var z + mean target var)."""
    Z, V = [], []
    for _, b in batches('train', 1024, False, mods, torch.Generator(device=RT.DEV).manual_seed(9)):
        z, vt = target(b['t'], b['tm'])
        Z.append(z); V.append(vt)
    Z, V = torch.cat(Z), torch.cat(V)
    return Z.mean(0), Z.var(0) + V.mean(0)


@torch.no_grad()
def collapse_diag(experts, target, mods):
    experts.eval()
    set_seed(5)
    Zt, Zf = [], []
    for idx, b in batches('valid', 1024, False, mods, torch.Generator(device=RT.DEV).manual_seed(5)):
        mus, lvs = expert_beliefs(experts, b)
        Zf.append(poe(mus, lvs, torch.ones(len(idx), len(mods), device=RT.DEV))[0])
        if target is not None:
            Zt.append(target(b['t'], b['tm'])[0])
    out = {f'fused_{k}': v for k, v in latent_stats(torch.cat(Zf)).items()}
    if Zt:
        out.update({f'target_{k}': v for k, v in latent_stats(torch.cat(Zt)).items()})
    return out


# ----------------------------------------------------------------------------- checkpoints
def ckpt_dirs():
    dirs = [os.path.join(RT.CFG['out_dir'], 'ckpt')]
    dirs += sorted(set(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/ckpt/*.pt', recursive=True)))
    return dirs


def find_ckpt(name):
    for d in ckpt_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def pretrain_or_load(obj, mods, group, seed, c, diag_rows=None):
    """Returns experts, target, target marginal. Looks in this run's ckpt folder and in every attached
    Kaggle input (e.g. the output of 01_pretrain_experts) before training."""
    name = f'v2_experts_{group}_{obj}_s{seed}.pt'   # v2: never confused with run-1 checkpoints
    path = find_ckpt(name)
    if path:
        ck = torch.load(path, map_location=RT.DEV)
        experts = make_experts(mods, c, obj).to(RT.DEV)
        experts.load_state_dict(ck['experts'])
        target, marginal = None, None
        if ck.get('target') is not None:
            tmod = copy.deepcopy(_online(experts, obj))
            tmod.load_state_dict(ck['target'])
            target = Target(tmod, _target_kind(obj)).to(RT.DEV)
            marginal = (ck['marg_mu'].to(RT.DEV), ck['marg_var'].to(RT.DEV))
        print(f'    loaded {path}')
        return experts.eval(), target, marginal
    t0 = time.time()
    experts, target = pretrain(obj, mods, c, seed)
    marginal = target_marginal(target, mods) if target is not None else None
    torch.save({'experts': experts.state_dict(), 'target': target.m.state_dict() if target else None,
                'marg_mu': marginal[0].cpu() if marginal else None,
                'marg_var': marginal[1].cpu() if marginal else None},
               os.path.join(c['out_dir'], 'ckpt', name))
    diag = dict(group=group, objective=obj, seed=seed, pretrain_min=(time.time() - t0) / 60,
                **collapse_diag(experts, target, mods))
    if diag_rows is not None:
        diag_rows.append(diag)
    print('    collapse check:', {k: round(v, 3) for k, v in diag.items() if isinstance(v, float)})
    return experts.eval(), target, marginal
