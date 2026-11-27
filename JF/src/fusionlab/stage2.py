"""Stage 2: train the fusion + head with the label (Gaussian NLL; MoNIG uses its own evidential loss), and the
point decoder D0. Every clip gets a random non-empty modality subset, so one model covers all missing patterns."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .batching import all_subsets, batches, get_batch, random_subsets, random_token_mask
from .fusion import make_fusion
from .models import Head, expert_beliefs, gauss_nll
from .runtime import RT, set_seed


@torch.no_grad()
def cache_beliefs(experts, split, mods):
    experts.eval()
    set_seed(11 + len(split))
    M, L = [], []
    for _, b in batches(split, 1024, False, mods, torch.Generator(device=RT.DEV).manual_seed(11 + len(split))):
        mu, lv = expert_beliefs(experts, b)
        M.append(mu); L.append(lv)
    return torch.cat(M), torch.cat(L)


def fused_output(fusion, head, mus, lvs, av):
    """-> label mean, label variance, latent part of the variance, fused latent mean, fused latent var, shares."""
    if fusion.output_level:
        m, v, share, _, _ = fusion(mus, lvs, av)
        w = av / av.sum(1, keepdim=True).clamp(min=1)
        fm = (w.unsqueeze(-1) * mus).sum(1)
        return m, v, v, fm, None, share
    fm, fv, share = fusion(mus, lvs, av)
    m, v, vl = head(fm, fv)
    return m, v, vl, fm, fv, share


def stage2(experts, fusion_kind, mods, c, seed, frozen, marginal=None):
    set_seed(seed + 1)
    M = len(mods)
    fusion = make_fusion(fusion_kind, c['d_lat'], M, marginal).to(RT.DEV)
    head = Head(c['d_lat']).to(RT.DEV)
    params = list(fusion.parameters()) + list(head.parameters())
    if not frozen:
        params += list(experts.parameters())
    opt = torch.optim.AdamW(params, lr=c['ft_lr'] if frozen else c['ft_lr_e2e'], weight_decay=0.01)
    cache = {s: cache_beliefs(experts, s, mods) for s in ('train', 'valid')} if frozen else None
    y = {s: RT.DATA[s]['y'].to(RT.DEV) for s in ('train', 'valid')}
    vgen = torch.Generator(device=RT.DEV).manual_seed(3)

    def loss_over(split, av_fn, train):
        n = RT.N_SPLIT[split]
        bs = c['batch'] if train else 2048
        idx = torch.randperm(n, device=RT.DEV) if train else torch.arange(n, device=RT.DEV)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            if frozen:
                mus, lvs = cache[split][0][j], cache[split][1][j]
            else:
                b = get_batch(split, j.cpu(), mods, None if train else vgen)
                mus, lvs = expert_beliefs(experts, b)
            av = av_fn(len(j))
            if fusion.output_level and train:
                yield fusion.loss(y[split][j], mus, lvs, av, c['monig_reg'])
            else:
                m, v, _, _, _, _ = fused_output(fusion, head, mus, lvs, av)
                yield gauss_nll(y[split][j], m, v)

    best, best_state, bad, ep = float('inf'), None, 0, 0
    for ep in range(c['ft_epochs']):
        fusion.train(); head.train(); experts.train(not frozen)
        for loss in loss_over('train', lambda B: random_subsets(B, M), True):
            opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        fusion.eval(); head.eval(); experts.eval()
        with torch.no_grad():
            vals = []
            for code in all_subsets(M):
                av = torch.tensor(code, device=RT.DEV, dtype=torch.float32)
                set_seed(17)
                vals.append(torch.cat(list(loss_over('valid', lambda B: av.expand(B, M), False))).mean().item())
            v_nll = float(np.mean(vals))
        if v_nll < best - 1e-4:
            best, bad = v_nll, 0
            best_state = copy.deepcopy((fusion.state_dict(), head.state_dict(),
                                        None if frozen else experts.state_dict()))
        else:
            bad += 1
            if bad >= c['patience']:
                break
    fusion.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    if not frozen:
        experts.load_state_dict(best_state[2])
    experts.eval(); fusion.eval(); head.eval()
    return fusion, head, dict(stage2_valid_nll=best, stage2_epochs=ep + 1, alpha=head.log_alpha.exp().item(),
                              s_noise=head.log_s.exp().item(), **fusion.info())


def train_d0(experts, fusion, head, mods, c, seed):
    """D0: a point decoder trained with squared error on the fused latent mean (no uncertainty at all)."""
    set_seed(seed + 2)
    M = len(mods)
    cache = {s: cache_beliefs(experts, s, mods) for s in ('train', 'valid')}
    y = {s: RT.DATA[s]['y'].to(RT.DEV) for s in ('train', 'valid')}
    lin = nn.Linear(c['d_lat'], 1).to(RT.DEV)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=0.01)
    best, best_state = float('inf'), None
    for _ in range(c['d0_epochs']):
        perm = torch.randperm(RT.N_SPLIT['train'], device=RT.DEV)
        for i in range(0, RT.N_SPLIT['train'], c['batch']):
            j = perm[i:i + c['batch']]
            with torch.no_grad():
                fm = fused_output(fusion, head, cache['train'][0][j], cache['train'][1][j],
                                  random_subsets(len(j), M))[3]
            loss = F.mse_loss(lin(fm).squeeze(-1), y['train'][j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            mses = []
            for code in all_subsets(M):
                av = torch.tensor(code, device=RT.DEV, dtype=torch.float32).expand(RT.N_SPLIT['valid'], M)
                fm = fused_output(fusion, head, cache['valid'][0], cache['valid'][1], av)[3]
                mses.append(F.mse_loss(lin(fm).squeeze(-1), y['valid']).item())
        if np.mean(mses) < best:
            best, best_state = np.mean(mses), copy.deepcopy(lin.state_dict())
    lin.load_state_dict(best_state)
    return lin.eval()


def me_beliefs(experts, b, K, ratio, gen, masks=None, xs=None):
    """Masking ensemble (novel): K random maskings per expert; var = mean predicted var + var of predicted means."""
    MU, V = [], []
    for _ in range(K):
        mk = {m: random_token_mask(masks[m] if masks and m in masks else b[m + 'm'], ratio, gen) for m in experts}
        mu, lv = expert_beliefs(experts, b, mk, xs)
        MU.append(mu); V.append(lv.exp())
    MU, V = torch.stack(MU), torch.stack(V)
    epi = MU.var(0, unbiased=False)
    return MU.mean(0), torch.log(V.mean(0) + epi), epi
