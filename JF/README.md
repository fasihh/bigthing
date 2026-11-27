# JF · Multimodal Fusion via Variational JEPA

Uncertainty-aware fusion of text, audio and video on **CMU-MOSEI**: each modality has a JEPA-style expert that
outputs a Gaussian belief, and the beliefs are combined by a **Product of Experts (PoE)**. Every experiment compares
PoE against the other fusion methods, on accuracy *and* on whether the uncertainty is honest when modalities are
missing, corrupted or noise.

## Layout

```
JF/
├── src/fusionlab/          shared library: every notebook imports it, so each method exists exactly once
│   ├── data.py             MOSEI loading + preprocessing (mosei_compact.npz cache)
│   ├── batching.py         batches, masking, test conditions, Noisy TV A–D corruptions
│   ├── models.py           experts (incl. S-JEPA and predictor ensemble), EMA target, uncertainty-aware head
│   ├── fusion.py           concat, late average, MoE, attention, PoE, prior-corrected PoE, MoNIG + novel fusions
│   ├── pretrain.py         stage 1 objectives: vjepa, sjepa, recon, align, vjepa_vt/vts/pe
│   ├── stage2.py           stage 2 (fusion + head), point decoder D0, masking ensemble
│   ├── evaluate.py         decoders D0–D3m, every metric, probes
│   ├── runner.py           resumable grid runner; reuses checkpoints across notebooks
│   ├── report.py           scoreboards (mean ± std over seeds) and bar charts
│   └── huang.py            reproduction harness for Huang (2026); downloads his code at run time
├── configs/                one YAML per notebook (seeds, methods, conditions)
├── notebooks/              run these on Kaggle (.ipynb); notebooks/src/*.py are the editable sources
├── results/                numbers and figures produced by the notebooks
├── scripts/py2nb.py        rebuilds the .ipynb files from notebooks/src
└── tests/smoke_test.py     crash check on placeholder arrays (never reported as results)
```

## Notebooks: one question each

| # | Notebook | Question | GPU time (P100, 3 seeds) |
| --- | --- | --- | --- |
| 00 | `00_data_prep` | Convert MOSEI once into a fast cache | CPU, ~15 min |
| 01 | `01_pretrain_experts` | Do the experts train without collapsing, under every objective? | ~1.5 h |
| 02 | `02_huang_reproduction` | Do Huang's noisy TV results hold on new seeds? | CPU, ~20 min |
| 03 | `03_mosei_missing_modalities` | Which fusion stays accurate and calibrated with missing modalities? | ~2.5 h |
| 04 | `04_mosei_corruption` | Does uncertainty notice corrupted or mismatched input? | ~1 h (reuses 03) |
| 05 | `05_multimodal_noisy_tv` | Does fusion ignore a noise channel / sticky distractor? | ~4 h |
| 06 | `06_downstream_calibration` | How much do the uncertainty and conformal calibration improve ranges and decisions? | ~1 h (reuses 03/04) |
| 07 | `07_novel_methods` | Do the new methods beat the best baselines? | ~4.5 h |
| 08 | `08_scoreboard` | Every number in one file (`final_numbers.md`) | CPU |

Run order: 00 → 01 → 03 → 04 → 06, and 02, 05, 07 in any order after 01. Every training notebook is resumable
(`RESUME_FROM`) and skips models it finds in attached inputs.

## Methods compared

**Fusion (all notebooks):** concatenation · late averaging · mixture of experts (MoE) · attention gate · **PoE** ·
prior-corrected PoE · MoNIG (Ma et al., NeurIPS 2021, evidential).
**Novel fusions (07):** disagreement-aware PoE · robust PoE · both · marginal-corrected PoE.

**Expert objectives:** **V-JEPA** (ours) · S-JEPA (Giakoumoglou et al., 2026) · reconstruction (MVAE) ·
contrastive alignment · supervised only. **Novel (07):** variational target (closed form, sampled) · predictor
ensemble · masking-ensemble uncertainty.

**Decoders:** point decoder (D0) · constant ranges (D1) · raw Gaussian (D2) · rescaled (D3a) · split conformal
(D3b) · per-modality-pattern (Mondrian) conformal (D3m) · per-condition conformal oracle (D3c).

## Running on Kaggle

1. Upload or link this repo (Kaggle: *Add Input → GitHub* or a dataset of the `JF/` folder). The first cell of every
   notebook finds `src/` automatically, or clones the repo when Internet is on.
2. Run `00_data_prep` once; save its output as a dataset; attach it to every other notebook.
3. Run `01_pretrain_experts`; attach its output to 03, 04, 06 and 07.
4. Each notebook ends with a **scoreboard** and **headline numbers** (`headline_XX.csv`); attach all outputs to
   `08_scoreboard` for the final file.

## Reproducibility

Seeds 0, 1, 2 everywhere (Huang reproduction: his 5 seeds + 10 new ones). Splits: train for training, validation
for early stopping and all calibration, test only for reporting. All test-time corruptions use fixed seeds, so every
model sees identical inputs.
