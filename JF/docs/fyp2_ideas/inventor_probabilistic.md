# Probabilistic / Bayesian / information-theoretic method ideas for FYP-2

Author angle: probabilistic, Bayesian and information-theoretic ideas. Written after reading `BRIEF.md` and
`fusionlab/{models,fusion,pretrain}.py` (also `batching.py` for the distractor and `evaluate.py` for the probes).

## Code facts that shaped the ideas

- An expert outputs `mu, lv` (64-d diagonal Gaussian). `bound_lv` keeps sigma in [0.05, 7.4]. Nothing in the expert
  sees anything except its own input. So the variance is a regression output. It is aleatoric by construction, and
  on an off-manifold input it can take any value (this is limitation 1).
- `poe_core` sums the precisions `exp(-lv) * av`. Each expert's precision is self-reported, and the experts are
  assumed independent. `rpoe` already down-weights an expert that is far from the leave-one-out consensus, using a
  raw Mahalanobis distance. That fails after fine-tuning because the experts' means stop agreeing (brief: AUROC is
  about 0.50).
- The Noisy TV D distractor (`batching.ar1_distractor`) is resampled on every call, so it is independent of the clip
  and of the label. In the shared case it enters every modality through one fixed linear mix
  `(d @ DIST_MIX[m]) * FEAT_STD[m]` on word-aligned tokens. So under the shared distractor the modalities share a
  low-rank, linear, almost noise-free component that does not exist in clean MOSEI.
- Stage 1 already has the residuals `z - mu_m` against the EMA target, with no labels needed. They are a free
  estimate of how the experts' errors covary.

Notation: M experts, m = 1..M. Belief b_m(x) = N(mu_m, diag(s_m^2)), precision Lambda_m = diag(s_m^{-2}).
S is a subset of experts. PoE_S is the existing fused Gaussian over S. The existing `Head` gives (m_S, v_S) for the
label.

---

## 1. Liar's Posterior Fusion (LPF): exact Bayesian inference over which expert is corrupted

**Principle.** Do not trust an expert's self-reported precision. Infer the posterior probability that each expert is
corrupted from how jointly plausible all the beliefs are, then average the subset-PoEs under that posterior. A
corrupted expert cannot fake this quantity, because it depends on the other modalities' inputs.

**Math.**
- Use a low-dimensional summary of each belief, u_m = P_m mu_m in R^k (k about 8). P_m is a PCA of that expert's
  means on clean train data. Optionally add the scalar head projection w.mu_m. Stack them as u = [u_1, ..., u_M].
- Fit a clean joint model on unlabeled train data: u ~ N(m, C), with a full kM x kM covariance (24 x 24 for TAV).
  Ledoit-Wolf shrinkage keeps it stable.
- Introduce corruption indicators c_m in {0, 1} with prior pi (for example 0.05). Under c_m = 1, u_m comes from a
  broad outlier density N(m_m, kappa C_mm) with kappa about 10, independent of everything else.
- Each expert's own reported noise is added to its block: C_mm(x) = C_mm + P_m diag(s_m^2(x)) P_m^T. An expert that
  admits it is unsure is then not punished for being far away.
- With R = {m : c_m = 0}:

      p(u | c) = N(u_R ; m_R, C_RR(x)) * prod_{m not in R} N(u_m ; m_m, kappa C_mm)
      p(c | u) proportional to p(u | c) * prod_m pi^{c_m} (1 - pi)^{1 - c_m}      (2^M configurations: 8 for TAV, 16 with N)

- Predictive distribution (Bayesian model averaging over subsets):

      p(y | x) = sum_R p(R | u) N(y ; m_R, v_R),       (m_R, v_R) = Head(PoE_{R and available})
      reported mean  = sum_R p_R m_R
      reported var   = sum_R p_R (v_R + m_R^2) - mean^2

- The per-expert corruption score is r_m = P(c_m = 1 | u), which can be read directly as an OOD detector.
- "Frozen witnesses" variant: compute p(c | u) from the frozen stage-1 V-JEPA latents, where cross-modal agreement
  is still informative (brief: disagreement AUROC up to 0.87), and apply it to the fine-tuned predictors' subset
  outputs. This sidesteps limitation 4 without giving up fine-tuning accuracy.

**Why it fixes the brief's failures.**
- *Limitation 1, noise on text (A, lambda = 1):* the text belief from noise is not where N(u_t | u_a, u_v) puts it.
  So r_t goes up, the mixture moves to PoE_{AV} (which was trained under random subsets), and sigma rises to the AV
  level automatically. This is the epistemic term that the Gaussian head lacks.
- *Limitation 2, confident-but-wrong expert:* the expert's own precision only enters through C_mm(x). A small sigma
  makes the expert more surprising when it is inconsistent, not less.
- *Swapped modality (C):* a swapped clip is a real clip, so it is in-distribution on its own. Only the joint density
  can flag it.
- *Limitation 7:* the BMA variance includes the spread between subsets, so coverage under corruption is repaired
  without conformal.

**Measurable predictions (hypotheses to test).**

| Setting | Quantity | Now | Predicted |
|---|---|---|---|
| Text replaced by noise, frozen V-JEPA | text fusion share | about 0.89 | below 0.25 |
| same | MAE | 0.85 | about 0.80 (AV-only level is 0.81) |
| same | Cov90 | 0.75-0.80 | 0.86-0.89 |
| same | tracking ratio | 0.61-0.90 | 0.95-1.1 |
| Swap (C), fine-tuned | AUROC of r_m (frozen witnesses) | about 0.50 | about 0.80 |
| Swap (C), frozen | AUROC of r_m | 0.56-0.87 (raw disagreement) | about 0.85 |
| Clean data | MAE / NLL | baseline | unchanged within about 0.005, because the posterior puts about 0.95 on "all reliable" |

**Implementation.**
- New `fusion.py` kind `lpf`.
- Add `fit_joint(experts, mods)`, similar to `target_marginal` in `pretrain.py`: one pass over train, then PCA and a
  covariance. Store the result as buffers.
- In `Fusion.forward`, loop over the 2^M masks with `poe_core`, then mix. `stage2.fused_output` must return a
  mixture. The simplest route is to moment-match into (m, v) so that the evaluation suite works unchanged.
- The head is trained exactly as now, with random subsets, so every PoE_R already exists. pi and kappa are fixed or
  chosen on clean validation NLL, never on corrupted data. pi can also be learned by EM on unlabeled data.
- Cost: under one GPU-minute to fit and 8x a cheap PoE at test time. No retraining is needed: it runs on top of
  existing checkpoints.

**Risks.**
1. The noise-text belief could land near the conditional mean, because a/v predict t only weakly, so C_t|av is broad.
   Do this pilot first: compute the conditional Mahalanobis distance of noise-text u_t on existing checkpoints.
   That takes one afternoon and either validates or kills the idea.
2. With only 2 trusted experts, "which one lies" is not identifiable: a/v vs t is symmetric. The prior favours the
   explanation with the higher marginal density, and noise-text has a low marginal density as well.
3. Under the shared distractor every modality is shifted together, so this method cannot detect it. Idea 4 handles
   that case.

**Prior work and how this differs.**
- Classical fault detection and isolation in sensor fusion (chi-square innovation tests, multiple-model adaptive
  estimation) and outlier-indicator Bayesian sensor fusion. These use known physical models. LPF instead learns the
  joint belief density without labels and uses deep JEPA beliefs.
- gPoE (Generalized PoE, [arXiv 2211.03587](https://ar5iv.labs.arxiv.org/html/2211.03587)) learns per-modality
  weights.
- Reliability-Aware Probabilistic Fusion ([arXiv 2610.29621](https://arxiv.org/pdf/2610.29621)) multiplies the
  precision by a learned sigmoid r_m. The authors themselves say this is "not a calibrated quality probability". It
  is supervised and has no posterior over corruption.
- UNO (noisy-or fusion, [arXiv 1911.05611](https://arxiv.org/pdf/1911.05611)).
- None of these computes an exact posterior over corruption configurations from a label-free joint density, and none
  uses BMA over subset-PoEs. The existing `rpoe` is a heuristic special case: LOO distance with an identity
  covariance and no prior.

---

## 2. Bates-Granger PoE (BG-PoE): fuse with the experts' error covariance, not their independence

**Principle.** PoE double-counts whatever errors the experts share. Estimate the cross-expert error correlation from
the stage-1 residuals (no labels) and fuse by generalized least squares, so that redundant, shrink-to-the-mean, or
pure-noise experts lose their weight automatically.

**Math.**
- Standardized stage-1 residuals: e_m = (z - mu_m) / s_m. Per latent dimension j (or pooled over dimensions after
  whitening), estimate R_j in R^{MxM}, the correlation of (e_1j, ..., e_Mj) on train, with shrinkage
  R <- (1 - a) R + a I.
- Per sample: Sigma_j(x) = D_j(x) R_j D_j(x), with D_j(x) = diag(s_1j(x), ..., s_Mj(x)). Restrict it to the
  available experts A, and add the prior as a separate independent expert:

      Pfused_j = 1_A^T Sigma_j,A^{-1} 1_A + 1
      mu_j     = ( 1_A^T Sigma_j,A^{-1} mu_j,A ) / Pfused_j
      var_j    = 1 / Pfused_j

- With R = I this is exactly `poe_core`. The weights w = Sigma^{-1} 1 can be negative, which cancels the shared
  error. Use a non-negative variant (clip, or project R to keep w >= 0) as the safe default.

**Why it fixes the brief's failures.**
- *Noise channel (B):* an expert that predicts the marginal has residual e_n which is about (z - zbar)/s, and that is
  exactly the shrinkage part of every weak expert's error. So corr(e_n, e_a) is high, and GLS gives n almost no
  weight. Today PoE gives it 13-38%.
- *Overconfidence when text is missing:* audio and video both shrink toward the mean, so their errors are positively
  correlated. PoE treats them as two independent votes, which explains worst-subset Cov90 of 0.68-0.70.
- `mpoe` (divide out the duplicated prior) is the special case where the shared error is exactly the prior. BG-PoE
  generalizes it.
- *Shared distractor:* if the experts copy the same distractor, their residuals are correlated, so the fused
  variance is no longer falsely small.

**Measurable predictions.**

| Quantity | Now | Predicted |
|---|---|---|
| Precision share of the noise modality (B), frozen V-JEPA | 20% | below 5% |
| Worst-subset Cov90, raw Gaussian (no conformal) | 0.68-0.70 | 0.82-0.87 |
| Fine-tuned NLL | 1.26 | about 1.22 |
| Clean TAV MAE | baseline | about the same |

**Implementation.**
- New fusion kind `bgpoe`.
- R is estimated in a `target_residual_corr(experts, target)` pass, next to `target_marginal`. For the fine-tuned
  case, re-estimate R from the per-expert scalar prediction residuals on the validation split.
- A batched MxM solve per dimension (M is at most 4), which is trivial.
- Optional stage-1 variant: replace the `fused` term in `_loss_vjepa` with the BG-PoE NLL, so that the experts are
  rewarded for errors that are complementary rather than redundant. This is a diversity-by-likelihood objective.
- Cost: negligible.

**Risks.**
- Unconstrained GLS weights are unstable when R is nearly singular (text and the target are highly correlated).
  Shrinkage and the non-negative constraint are needed.
- The correlation can depend on the input (heteroscedastic correlation). A learned R(x) is a stretch goal.

**Prior work.**
- Bates and Granger (1969) optimal forecast combination.
- CoDE, "Aggregation of Dependent Expert Distributions" ([arXiv 2505.01134](https://arxiv.org/html/2505.01134v1)),
  is the closest. It generalizes PoE with an error covariance, but uses a single scalar rho chosen by
  cross-validation inside a multimodal VAE.
- BG-PoE differs in three ways:
  1. A full, per-dimension, mask-aware R, estimated label-free from JEPA target residuals.
  2. An explicit diagnosis that the noise-channel and missing-text overconfidence failures are shared-shrinkage
     errors.
  3. Use as a training signal for complementary experts.
- **The CoDE paper must be cited, and the contribution framed as an extension and diagnosis rather than a new
  operator.**

---

## 3. Explain-Yourself Precision (EYP): an expert keeps its precision only if its belief can regenerate its own input

**Principle.** Precision should be the evidence the input actually provides about the latent. Under a heavy-tailed
(Student-t) likelihood, that evidence shrinks with how badly the belief explains the observed tokens. The variational
Bayes weight of the scale mixture becomes a multiplicative precision factor.

**Math.**
- Add a small decoder g_m(mu_m, pos) that predicts the token features (or pooled second-order statistics: mean,
  token covariance diagonal, lag-1 autocorrelation) of modality m.
- Clean-data residual energy: Q_m(x) = sum over held-out tokens and features of ((x - g_m)/tau)^2.
- Fit a scale prior s_m^2 ~ InvGamma(a_m, b_m) by moment-matching the distribution of Q_m on clean train data. This
  calibrates the effective degrees of freedom D_eff, which accounts for correlated tokens.
- VB posterior weight:

      u_m(x) = E[s_m^{-2} | Q_m] = (a_m + D_eff/2) / (b_m + Q_m(x)/2)

  This is about 1 on clean data and well below 1 when the input is not explicable.
- Use Lambda_m^eff = u_m Lambda_m in `poe_core` and report s_m^2 / u_m as the expert's sigma. This is the standard
  Student-t VB fusion update, but driven by the input-reconstruction innovation rather than the target innovation.

**Why it fixes the brief's failures.**
- Pure-noise text cannot be regenerated from any 64-d belief: noise tokens have no lag correlation and the wrong
  covariance spectrum. So Q is far above its clean range, u_t is small, and sigma_t rises. That reverses the observed
  fall of sd_t (0.068 -> 0.061).
- Partial noise lambda gives a smoothly growing Q, so tracking becomes monotone.

**Measurable predictions.**

| Setting | Quantity | Now | Predicted |
|---|---|---|---|
| Noise text, lambda = 1 | text sigma (frozen V-JEPA) | 0.068 -> 0.061 | at least 0.15 |
| same | text share | 0.89 | below 0.4 |
| Noisy TV A curve | tracking ratio | 0.61-0.90 | 0.9-1.0 |
| Noisy TV A curve | Spearman(sigma, error) | about 0.22 | about 0.4 |
| Swap (C) | detection | about 0.50 | no change: the swapped clip regenerates fine |

The swap result is stated honestly so that idea 1 is the complementary piece.

**Implementation.**
- Reuse the `extra` decoder pattern from `_loss_recon`, but condition it on each expert's own mu_m and train it
  jointly in stage 1 as an auxiliary loss (weight about 0.1).
- After pretraining, run one pass to fit (a, b, D_eff).
- A new `Expert.evidence(x, mask)` method returns u_m. Then `lvs_use = lvs - log u`, the same hook that `rpoe` uses.
- Cost: plus about 10% in stage 1. Zero extra training if the decoder is fit post hoc on frozen beliefs.

**Risks.**
- On clean data the reconstruction is weak (BERT-PCA tokens are hard to predict from 64 dimensions), so the
  sensitivity may be low. Mitigation: use second-order statistics instead of raw tokens.
- A decoder that is too strong can "explain" noise too. Keep it a small MLP.

**Prior work.**
- Reconstruction error as an OOD score (autoencoders and VAEs; many papers).
- Outlier-robust Kalman filtering via VB Student-t (Agamennoni et al. 2012) and robust Student-t PoE.
- The new part is turning the self-explanation residual into a calibrated VB precision multiplier inside a JEPA PoE,
  with the degrees of freedom fitted on clean data and no corruption seen during fitting.

---

## 4. Too-Good-To-Be-True Target (TGT): information that different sensors share linearly and almost perfectly is an artefact, not semantics

**Principle.** BERT words, COVAREP prosody and FACET action units share meaning only nonlinearly and weakly. A
near-perfect *linear* canonical correlation between their token streams is therefore evidence of a common nuisance.
The EMA target is penalized for carrying the mutual information of those components.

**Math.**
- Token-level CCA (MOSEI aligned_50 is word-aligned) between text tokens and each other modality on unlabeled train
  data. Use multiset CCA for 3 or more modalities. It gives canonical pairs (a_i, b_i) with correlations rho_i.
- Gaussian mutual information per component: I_i = -1/2 log(1 - rho_i^2).
- Nuisance set N = {i : rho_i > tau}, with tau set by a "semantic ceiling". For example tau = max(0.8, the 99th
  percentile of rho on a clean held-out reference). On clean MOSEI, N is expected to be empty, so the method is a
  no-op there.
- Nuisance variates: n(x) = [a_i^T pooled(x_t)]_{i in N}.
- Added to the stage-1 loss, acting on the online text encoder that feeds the EMA target:

      L_TGT = lambda * || CrossCov_batch( h_t , n(x) ) ||_F^2

  (or HSIC with an RBF kernel for nonlinear leakage). By the EMA, the target then inherits the property that it
  contains (linearly) no information about N.
- Hard variant: project N out of every modality's raw tokens before both encoders (oblique projection). This is
  simpler but more destructive.
- Information view: maximize I(Z_target ; X_others) (JEPA) subject to I(Z_target ; N) of about 0, where N is the
  "cheaply shared" information. In V-information terms (Xu et al. 2020), only information that needs nonlinear
  computation to share is kept.

**Why it fixes the brief's failures.**
- In the shared Noisy TV D, the distractor enters every modality as `d @ DIST_MIX[m]`: the same k-dimensional AR(1)
  signal, linearly mixed, with sigma = 8 times the feature std. So k canonical correlations are about 1 and CCA finds
  exactly the distractor subspace. The target is then forbidden to encode it, so the predictors can no longer profit
  from predicting it (this is limitation 3).

**Measurable predictions.**

| Setting | Quantity | Now | Predicted |
|---|---|---|---|
| Shared distractor, sigma = 8 | signal probe R2 (V-JEPA) | about 0.07 | at least 0.25 |
| same | distractor R2 | 0.52 | below 0.10 |
| Shared distractor, sigma = 2 | similar direction | | |
| Clean data | signal R2 / MAE | baseline | unchanged within about 0.01 (N empty, verified by reporting the rho spectrum) |
| Independent distractor | | | **not** addressed (no cross-modal sharing); stated as a limitation |

For the independent case the analogous within-modality test would be lag CCA (first half vs second half of the
clip). But sentiment is clip-constant too, so that is riskier: optional.

**Implementation.**
- Fit CCA once per distractor setting. This needs `RT.ENV` active and a closed-form generalized eigenproblem on
  token covariances (128 + 74 + 35 dimensions), which takes seconds.
- The CCA must be fit on training data generated with the same corruption process as training. That is legitimate
  and label-free.
- Add the cross-covariance term inside `_loss_vjepa` next to `vicreg_terms(h_full)`.
- Cost: negligible.
- The same code also runs in Huang's setting with his own code (his "noisy TV" distractor) as an external
  replication.

**Risks.**
1. Real nuisances that are only moderately shared (for example speaker identity) are not caught.
2. If a distractor were nonlinearly mixed, linear CCA would miss it. KCCA or deep CCA is the extension.
3. This is a principled prior about *how* nuisance is shared, so it is not a universal solution. It must be
   presented as a falsifiable hypothesis ("artefacts share linearly, meaning shares nonlinearly").
4. Someone could object that the benchmark distractor is linear by design. Answer: report the rho spectrum on clean
   MOSEI, and add one nonlinear-mixed distractor condition as a stress test.

**Prior work.**
- BSS-CCA artefact removal in EEG (De Clercq et al. 2006) removes components by their *autocorrelation*. TGT is
  cross-modal.
- Information-theoretic disentanglement of shared and unique information
  ([arXiv 2410.23996](https://arxiv.org/html/2410.23996v1)) treats shared information as signal, which is the
  opposite stance.
- CF-JEPA ([arXiv 2610.00727](https://arxiv.org/pdf/2610.00727)) separates controllable from uncontrollable latents
  using actions. Affective data has no actions.
- VJEPA (Huang, [arXiv 2601.14354](https://arxiv.org/html/2601.14354v1)) identifies the noisy-TV issue but does not
  solve the multimodal shared case.
- I found no paper that uses the canonical correlation spectrum to define and remove a target-hijacking nuisance in
  JEPA.

---

## 5. Bits-Budget V-JEPA (B2-JEPA): constrain the predictive information in nats instead of weighting losses

**Principle.** The suspected way variational JEPA fails is that the predictor "gives up": it inflates s^2 toward
Var(z), so the 1/s^2-scaled gradient on mu dies and the run collapses. Turning "predict at least R nats" and "target
entropy at least H" into constraints with dual variables makes every seed operate at the same information point.

**Math.**
- Gaussian proxies on each batch:
  - predictive information: Ihat = 1/2 sum_j [ log Var_batch(z_j) - mean_batch log s_j^2 ]
  - target entropy: Hhat = 1/2 logdet( Cov_batch(z) + eps I )
- Objective:

      min  L_NLL + beta KL
      s.t. Ihat >= R_min,   Hhat >= H_min

- Lagrangian with GECO-style dual ascent on EMA-smoothed constraints:

      lambda_1 <- [ lambda_1 + eta (R_min - Ihat) ]_+
      lambda_2 <- [ lambda_2 + eta (H_min - Hhat) ]_+

- lambda_2 replaces the fixed `w_vic`.
- R_min and H_min are set once, as a fraction (about 0.7) of what deterministic JEPA reaches. They are not tuned per
  seed.
- Optional ablation: beta-NLL gradient reweighting (Seitzer et al. 2022), to separate the "dead gradient" mechanism
  from the "information level" mechanism.

**Why it fixes the brief's failures (limitation 6).**
- Huang reproduction: one seed of VJEPA collapses (R2 -6.4) and VJEPA loses to JEPA in 7 of 10 seeds. If the
  collapse mechanism is "variance up, information down", the constraint pushes back the moment Ihat falls, whatever
  the initialization.
- *Falsification check first (cheap):* on the collapsing seed's saved run, test whether mean log s^2 approaches
  log Var(z) before the collapse.

**Measurable predictions (Huang's code, 15 seeds, plus MOSEI with 5 seeds).**

| Quantity | Now | Predicted |
|---|---|---|
| Catastrophic seeds (R2 < -1) | 1/15 | 0/15 |
| Seed std of signal R2 | baseline | at least 2x smaller |
| VJEPA beats JEPA | 3/10 | at least 5/10 |
| MOSEI frozen probe R2, seed std | baseline | smaller |
| MOSEI frozen probe R2, mean | baseline | unchanged |

**Implementation.**
- In `_loss_vjepa`, add two scalars and their updates (the duals are not parameters of the optimizer).
- Log Ihat and Hhat every epoch. These logs are a deliverable in themselves: "information trajectories" per seed.
- Cost: zero extra forward passes. The cost is running 15+ seeds, which is cheap on Huang's toy and about 5 x 40 min
  on MOSEI.

**Risks.**
- The Gaussian proxies are crude, and log Var_batch is noisy at batch 256. Use EMA statistics.
- If the instability comes from the EMA schedule rather than the variance, the fix will not help. That would still be
  a clear negative result with a mechanism test.

**Prior work.**
- GECO / "Taming VAEs" (Rezende and Viola 2018), Lagrangian VAE (Zhao et al. 2018), beta-NLL (Seitzer et al. 2022),
  VICReg (a fixed-weight variance hinge).
- JEPA training dynamics: "Drive vs. Decay" (NeurIPS 2026) and Var-JEPA ([arXiv 2603.20111](https://arxiv.org/html/2603.20111)).
- I found no work that puts the variational-JEPA objective under constraints measured in nats, or that tests the
  "predictor gives up" collapse mechanism across seeds.

---

## 6. Posterior-as-Prior Fine-tuning (PaP-FT): keep the label-free beliefs as the Bayesian prior during fine-tuning

**Principle.** Fine-tuning should update the stage-1 posterior over latents, not overwrite it. Each fine-tuned
expert's belief pays a KL penalty to its frozen stage-1 belief, so the cross-modal agreement that ideas 1 and 3 rely
on survives.

**Math.**

    L_stage2 = NLL(y | Head(Fuse(b^ft))) + gamma * sum_m KL( b_m^ft(x) || sg(b_m^frozen(x)) )

- The KL is in closed form for diagonal Gaussians.
- gamma is chosen as the largest value with a validation MAE loss of at most 0.005. That turns the
  accuracy-versus-structure trade-off into a reported curve.

**Why it fixes the brief's failures.** It addresses limitation 4 directly. Fine-tuning currently drives disagreement
AUROC to about 0.50 and destroys latent calibration. With the KL anchor the latent stays close to a space where
"agreement means correct", which makes LPF (idea 1) usable without the frozen-witness trick.

**Measurable predictions.**

| Quantity | Now | Predicted |
|---|---|---|
| Disagreement AUROC under swap (C), fine-tuned | about 0.50 | about 0.75 |
| MAE | 0.56 | at most 0.565 |
| Mahalanobis/d without text | 1.74 | about 1.2 |

**Implementation.**
- In `stage2.stage2`, keep a frozen copy of the experts and add the KL term.
- Cost: one extra forward of the frozen experts.

**Prior work.** L2-SP (Li et al. 2018), Variational Continual Learning (posterior-as-prior, Nguyen et al. 2018) and
KL-regularized fine-tuning. The idea is not novel on its own; it is included as the enabling piece for idea 1, and
for the new measurement of how much cross-modal structure each gamma keeps.

---

## How the ideas combine (possible FYP-2 storyline)

"Trust must be earned from the other modalities, not self-reported."

- LPF (idea 1) is the fusion rule.
- BG-PoE (idea 2) is its correlated-error cousin and can serve as LPF's within-subset fusion.
- EYP (idea 3) is the single-expert evidence term. It plugs into C_mm(x) of LPF as a heavier outlier prior when u_m
  is low.
- TGT (idea 4) fixes the target.
- B2-JEPA (idea 5) fixes seed stability.
- Every number is an existing metric in the evaluation suite.

## Novelty search summary

- Corruption-aware PoE weighting exists in learned, supervised forms: gPoE, Reliability-Aware Probabilistic Fusion
  (2610.29621), PRA-PoE / CoPoE (missing modalities). None does exact posterior inference over corruption
  configurations from a label-free joint density.
- Correlated-expert PoE exists (CoDE, 2505.01134, with a scalar rho), so idea 2 must be framed as an extension.
- JEPA versus distractors: VJEPA (Huang) and CF-JEPA (action-based factorization). I found nothing that uses
  canonical-correlation-based nuisance nulling for the EMA target.
- Constrained (GECO-style) training is known for VAEs. I found no application to variational-JEPA seed stability.
- These searches were brief, a handful of queries. A proper literature check should be repeated before submission.

## Top pick

**Liar's Posterior Fusion (idea 1).**
- It has a one-sentence principle and an exact Bayesian derivation with only 2 hyperparameters.
- It runs on top of the existing checkpoints with no retraining, so an undergraduate can get first numbers in a week.
- It attacks the central failure (sigma that does not rise under noise or swap) and limitation 2 with a quantity a
  corrupted expert cannot fake.
- It gives directly readable numbers: text share 0.89 -> below 0.25, swap AUROC 0.50 -> about 0.8, Cov90 under noise
  about 0.77 -> about 0.87.
- It has a one-afternoon go/no-go pilot (conditional Mahalanobis of the noise-text belief).
- Natural pairings: TGT (idea 4) for the distractor result, which LPF cannot address, and B2-JEPA (idea 5) as a
  self-contained stability chapter.
