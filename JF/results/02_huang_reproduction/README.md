# 02 · Huang (2026) noisy TV: reproduction with new seeds

Code: the author's `exp1_multiseed_variance` notebook from github.com/YongchaoHuang/VJEPA (commit `88a90d2`),
run unchanged (3,000 epochs, linear models, 4-d signal + 4-d sticky distractor in 20-d observations).
Author's seeds: 42, 111, 123, 456, 789. New seeds: 0–9. CPU, about 70 s per seed.

## 1. The published table reproduces exactly

Rerunning the author's 5 seeds gives his printed table to 3 decimals (largest difference: **0.000**).

## 2. At the hardest noise level (scale 8), signal R² on test (higher is better)

| Model | Author seeds: median | New seeds: median | New seeds: mean ± std | New seeds: worst seed |
| --- | --- | --- | --- | --- |
| VAE (reconstruction) | 0.009 | −0.498 | −1.231 ± 2.003 | −6.334 |
| AR (next-step reconstruction) | 0.110 | −0.156 | −0.899 ± 1.763 | −5.610 |
| JEPA (deterministic) | **0.693** | **0.679** | **0.599 ± 0.312** | **−0.153** |
| VJEPA | 0.563 | 0.459 | −0.150 ± 2.106 | −6.435 |
| BJEPA | 0.176 | 0.458 | 0.194 ± 0.686 | −1.473 |

## 3. Seed-by-seed wins for VJEPA at scale 8 (10 new seeds)

| VJEPA vs | Signal R² wins | Distractor R² wins (lower) |
| --- | --- | --- |
| VAE | **9/10** | **10/10** |
| AR | **9/10** | **9/10** |
| JEPA | 3/10 | 3/10 |
| BJEPA | 6/10 | 3/10 |

## What the numbers say

- **Holds:** JEPA-family models beat reconstruction (VAE, AR) under heavy noise: VJEPA wins 9/10 seeds on signal
  and 10/10 on ignoring the distractor against the VAE.
- **Does not hold on new seeds:** VJEPA is not better than plain deterministic JEPA. JEPA wins 7/10 seeds at scale 8
  (and the author's own 5-seed table already shows JEPA ahead: 0.749 vs 0.566).
- **Instability:** one new seed (5) makes VAE, AR, VJEPA and BJEPA fail badly (R² below −1.4), while JEPA stays
  above −0.16 on every seed. Means and standard deviations over 5 seeds hide this; report medians and the worst
  seed.
- **Measurement note:** at scale 0 the distractor is multiplied by 0, so its probe target is all zeros and
  scikit-learn's R² reports 1.000 for a constant prediction. The "distractor R² = 1.000" rows at scale 0 in the
  author's table are an artefact, not a measurement.

Figures: `fig_02_huang_scale8.png` (medians, middle 50% of seeds), `fig_02_per_seed.png` (every seed).
