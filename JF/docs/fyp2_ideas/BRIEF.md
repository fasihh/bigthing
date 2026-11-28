# Brief for FYP-2 idea generation (read fully before working)

## The project so far (FYP-1)
Title: "Multimodal Fusion via Variational JEPA". Undergraduate final-year project, compute = free Kaggle GPU
(P100/T4, ~30 GPU-h/week, 12 h sessions). Dataset: CMU-MOSEI (MMSA aligned_50.pkl: BERT text 50x768 -> PCA 128,
COVAREP audio 50x74, FACET video 50x35; one label: sentiment in [-3, 3]; 16,326/1,871/4,659 train/valid/test).
OASIS (medical imaging) was descoped.

Architecture: one expert per modality (2-layer transformer -> 64-d latent) outputs a diagonal Gaussian belief
N(mu_m, sigma_m^2). Stage 1 (no labels): experts predict an EMA target latent built from the text encoder (V-JEPA
style; Gaussian NLL + small KL + PoE-of-random-subset NLL + VICReg on the text encoder). Baseline objectives:
reconstruction (MVAE), contrastive alignment (InfoNCE), supervised-only. Stage 2 (labels): fusion of the beliefs ->
linear head y ~ N(w.mu + b, alpha * sum w^2 sigma^2 + s^2), trained by Gaussian NLL with random modality subsets.
Fusions: Product of Experts (PoE, N(0,I) prior), prior-corrected PoE, MoE (moment-matched), attention gate; now also
concat, late averaging, MoNIG (Ma et al. 2021 evidential). Frozen experts vs fine-tuned vs supervised.
Already-implemented "novel" variants (results pending): variational EMA target (closed-form expected NLL; sampled),
predictor ensemble (5 heads, bootstrap), masking-ensemble test-time uncertainty, disagreement-aware PoE (variance +=
g * spread of expert means), robust PoE (precision down-weighted by distance to leave-one-out consensus),
marginal-corrected PoE (prior = target marginal). S-JEPA (Giakoumoglou et al. 2026) added as an objective.
Evaluation: 7 modality subsets, partial text dropout, Noisy TV A (test-time noise x'=sqrt(1-l)x+sqrt(l)eps),
B (extra pure-noise modality), C (modality swapped from another clip), D (Huang-style AR(1) sticky distractor mixed
into every modality, train+test, sigma 2 or 8, independent or shared across modalities); decoders: raw Gaussian,
post-hoc temperature, split conformal, Mondrian (per-mask) conformal; selective prediction, PIT, CRPS, AURC.

## Key results (3 seeds, real MOSEI, mean +/- std)
- Accuracy is decided by fine-tuning, not the objective: fine-tuned/supervised PoE MAE ~0.56 on TAV (Acc2 ~0.845);
  frozen experts MAE ~0.70-0.73. Supervised-only PoE (0.558) is as good as any pretrained+fine-tuned model.
  Pretraining objective matters only when frozen.
- PoE and attention beat MoE on accuracy and NLL in every regime (fine-tuned NLL 1.26 vs 1.31).
- Calibration of the raw Gaussian is decent on clean data (Cov90 0.87-0.89, CalErr ~0.02-0.04). Fine-tuned PoE is
  overconfident when text is missing (worst-subset Cov90 0.68-0.70); split conformal fixes average coverage
  (0.91) but per-condition conformal shows remaining shift. Frozen V-JEPA latent: Mahalanobis/d ~0.98 (well
  calibrated in latent space) when text present, but 1.74 without text.
- Tracking ratio (sigma rise / error rise when text lost): frozen V-JEPA PoE ~1.0 (ideal); fine-tuned ~0.96;
  under 90% text dropout ~0.80-0.95; under text replaced by pure noise 0.61-0.90 -> sigma under-reacts.
- THE CENTRAL FAILURE: learned sigma does not notice corrupted/out-of-distribution input. When the text stream is
  replaced by pure noise (lambda=1) the text expert's sigma stays the same or FALLS (frozen V-JEPA sd_t 0.068 ->
  0.061) while its fusion weight stays ~0.89; MAE rises 0.73 -> 0.85 and Cov90 falls to ~0.75-0.80. Swapped
  modality (C): sigma AUROC ~0.50 (chance); expert disagreement AUROC 0.56-0.87 (frozen) but ~0.50 after
  fine-tuning (fine-tuning destroys cross-modal agreement as a signal).
- Failure detection: fine-tuned models: Spearman(sigma, error) ~0.22, AUROC(err>1) ~0.65; frozen PoE ~0 (useless).
- Noise channel (B): PoE gives the pure-noise modality 13-38% of the precision (frozen), V-JEPA 20%; supervised PoE
  learns to give it 2%. Coverage barely changes.
- Sticky distractor (D, Huang-style, multimodal): every self-supervised objective (V-JEPA, recon, align) is
  hijacked: signal probe R2 falls from ~0.33 to ~0.07 at sigma=8 and distractor R2 ~0.18 (indep) / 0.52 (shared);
  supervised keeps signal R2 0.47. So V-JEPA's EMA target ABSORBS the distractor, especially when it is shared
  across modalities (the target encoder is text, and the distractor is in the text too).
- Huang (2026) reproduction, his own code, 15 seeds: his table reproduces exactly; VJEPA beats VAE/AR (9/10 new
  seeds) but deterministic JEPA beats VJEPA in 7/10 new seeds; one seed makes VJEPA collapse (R2 -6.4) while JEPA
  stays >= -0.15. Variance is huge; 5-seed means hide it.
- Representation: V-JEPA latent effective rank ~35/64 (no collapse), lower than recon (53) and align (45).

## Limitations of every component seen so far (to attack)
1. Aleatoric-only sigma: a Gaussian predictor head cannot know it is out of distribution (no epistemic term).
2. PoE assumes conditionally independent experts and trusts precision blindly -> a confident-but-wrong expert wins.
3. EMA/self-predictive targets are hijacked by predictable nuisance (shared sticky distractor) - the "noisy TV"
   problem is NOT solved by variational JEPA in the multimodal case.
4. Fine-tuning erases the self-supervised structure (cross-modal agreement, latent calibration).
5. Text dominates; audio/video experts are weak (MAE ~0.81 without text); fusion mostly = text.
6. Instability across seeds (Huang); variational JEPA is less stable than deterministic JEPA.
7. Calibration shifts across modality masks (needs Mondrian conformal); no guarantee under corruption.

## Constraints for FYP-2 proposal
- Must be implementable by one undergraduate in ~6 months, trainable on free Kaggle GPUs (small transformers on
  precomputed MOSEI features is the budget; optionally a second dataset with precomputed features, e.g. CMU-MOSI,
  CH-SIMS, MELD, UR-FUNNY, IEMOCAP, or a non-affective multimodal benchmark with features available).
- Should build on what exists (JEPA experts, PoE, the evaluation suite) and produce a clear, *measurable* win on
  the failure modes above, with numbers a jury can read (coverage, AUROC, MAE, R2).
- Must be genuinely novel: not just "add an ensemble" or "add conformal". Prefer an idea with a crisp principle
  (one sentence) and a name.
