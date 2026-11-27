"""fusionlab: uncertainty-aware multimodal fusion with variational JEPA experts (CMU-MOSEI).

Typical notebook use:
    from fusionlab import config, data, batching, runner, report
    cfg = config.load_config('configs/03_mosei_missing.yaml')
    data.load_mosei(cfg)
    conds = batching.conditions(BASE, ['subset', 'textdrop'], cfg)
    runner.run_grid('missing', BASE, objectives, conds, frozen=[...], finetuned=[...], supervised=[...])
"""
from .runtime import BASE, NAMES, RT, set_seed  # noqa: F401

__version__ = '1.0.0'
