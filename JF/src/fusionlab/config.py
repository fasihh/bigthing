"""Default configuration. Each notebook loads configs/<name>.yaml on top of this."""
import copy
import os

DEFAULTS = dict(
    # data
    pkl_path=None,                 # None = auto-search: mosei_compact.npz, then aligned_50.pkl / unaligned_50.pkl
    max_len=50, text_pca=128,
    # experts
    d_model=128, d_lat=64, n_layers=2, dropout=0.1,
    # stage 1: label-free expert pretraining
    pre_epochs=30, pre_lr=3e-4, batch=256, mask_ratio=0.4,
    beta_kl=1e-3, w_fused=1.0, w_vic=1.0, ema_start=0.996,
    beta_recon=0.1, kl_warmup=0.3, align_temp=0.1,
    sjepa_dz=16, sjepa_beta=0.01, sjepa_warmup=0.3, sjepa_samples=16,      # S-JEPA (Giakoumoglou et al. 2026)
    pe_heads=5,                                                            # predictor ensemble
    # stage 2: fusion + uncertainty-aware head
    ft_epochs=40, ft_lr=1e-3, ft_lr_e2e=3e-4, patience=6, d0_epochs=15,
    monig_reg=0.01,                                                        # MoNIG evidential regulariser
    # test-time uncertainty
    me_K=8, me_ratio=0.4,                                                  # masking ensemble
    # noisy TV
    noise_dim=32, dist_k=8, dist_rho=0.9,
    # evaluation
    text_drops=[0.25, 0.5, 0.75, 0.9],
    noise_lambdas=[0.25, 0.5, 0.75, 1.0],
    levels=[0.5, 0.8, 0.9, 0.95],
    ppc_reps=200,
    seeds=[0, 1, 2],
    out_dir='/kaggle/working' if os.path.isdir('/kaggle/working') else './out',
    quick=False,                   # True = tiny epochs, 1 seed: a crash check of the whole notebook only
)

QUICK = dict(pre_epochs=2, ft_epochs=2, patience=1, d0_epochs=2, seeds=[0], ppc_reps=20, me_K=2, sjepa_samples=4)


def load_config(path=None, **overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update(overrides)
    if os.environ.get('FUSIONLAB_TEST'):          # crash-check mode for CI / local tests (never for results)
        cfg.update(quick=True, out_dir=os.environ.get('FUSIONLAB_OUT', './test_out'),
                   pkl_path=os.environ.get('FUSIONLAB_PKL'), text_drops=[0.5], noise_lambdas=[1.0])
        keep = {'vjepa', 'sjepa', 'vjepa_vt', 'vjepa_pe', 'none', 'poe', 'moe', 'monig', 'dapoe', 'mpoe', 'concat'}
        for k, v in list(cfg.items()):
            if isinstance(v, list) and v and isinstance(v[0], str) and k not in ('dist_patterns',):
                cfg[k] = [x for x in v if x in keep] or v[:1]
        cfg['dist_patterns'] = cfg.get('dist_patterns', ['TAV_shared'])[-1:]
        cfg['dist_sigmas'] = [8.0]
    if cfg['quick']:
        cfg.update(QUICK)
    return cfg
