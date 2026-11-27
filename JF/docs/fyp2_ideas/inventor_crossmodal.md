# Cross-modal prediction, world-model and systems ideas for FYP-2

Angle: the project's unique asset is that every JEPA expert *predicts a latent* rather than just encoding
its input. Today, though, every expert predicts the **same text-derived EMA target**, and that is the root of
three failures in the brief:

* A corrupted text stream is still "predicted" by a confident text head. Nothing compares the text input with
  what the *other* senses expect, so sigma stays flat (limitations 1 and 2).
* The only "other view" is text, and text also carries the distractor. The target can therefore absorb any
  nuisance that text carries (limitation 3).
* Fine-tuning optimises only the label NLL. The latent stops being a prediction of anything, so agreement
  between experts stops meaning anything (limitation 4: swap AUROC 0.50).

The ideas below turn the single "everyone predicts text" arrow into a **graph of predictions over senses and
time**. They then use that graph for three jobs: auditing reliability without labels, inference that can
explain away a bad expert, and choosing targets that a nuisance cannot hijack.

Code facts the sketches rely on:

* `Expert = SeqEncoder backbone + Gaussian pred head`.
* `Target` is an EMA copy of the text backbone (`'det'`: LayerNorm point target).
* `poe_core` works on `(mus, lvs, av)`.
* `stage2` either caches frozen beliefs or fine-tunes end to end with the label NLL only.
* The distractor `ar1_distractor` is an AR(1) process with rho = 0.9 over the 50 frames. It is mixed linearly
  into each modality (`RT.DIST_MIX[m]`).
* The probe target is the clip-mean of the distractor.

Notation:

* M modalities, indexed i and j.
* h_i: the pooled backbone of modality i.
* T_j: an EMA target encoder for modality j. Its output e_j = T_j(x_j) is a 64-d LayerNormed latent.
* g_{i->j}: a Gaussian predictor head, (mu_ij, sigma^2_ij) = g_{i->j}(h_i).

---

## 1. CROSSWIRE: the cross-modal residual graph and earned-trust fusion

**Principle.** Every modality predicts every other modality's latent. A clip's pattern of prediction errors is a
label-free fingerprint that points to the modality that is lying. An expert keeps its fusion weight only by
predicting the others on *this* clip.

### Math

Stage 1 adds one EMA target per modality, T_j (EMA of `experts[j].backbone`, as `Target(kind='det')`), and
M x M predictor heads. The diagonal heads g_{i->i} are ordinary masked self-prediction (JEPA). The current
text-target heads become the column j = t, so the fusion latent space does not change.

```
L_xwire = sum_{i,j} NLL( sg e_j  |  g_{i->j}(h_i(ctx_i)) )           # graph edges, incl. masked self edge
        + beta_kl KL + w_fused * NLL(sg e_t | PoE_subset)             # existing V-JEPA terms (column t)
        + w_vic * sum_i VICReg(h_i)                                  # every backbone is now a target: guard all
```

Residual graph at test time, as a per-edge standardised surprise:

```
r_ij(x) = (1/d) * sum_k (e_jk - mu_ijk)^2 / sigma^2_ijk
z_ij    = (r_ij - median_ij) / MAD_ij      # median and MAD of r_ij on clean validation clips, label-free
```

The node corruption score contrasts the edges that touch modality k with the edges that don't. With M = 3 that
leaves exactly one pair of untouched edges:

```
s_k = mean_{j != k}( z_kj , z_jk )  -  mean_{i,j != k}( z_ij )  +  z_kk
```

* The out-edges (z_kj) catch a modality that cannot predict the others.
* The in-edges (z_jk) catch a modality that the others cannot predict.
* The self edge (z_kk) catches internal incoherence, for example pure noise.

**Earned-trust PoE** ("self-auditing" fusion). Precision is multiplied by a trust factor, and nothing else in
`poe_core` changes:

```
lambda_m' = lambda_m * pi_m,    pi_m = sigmoid(tau - kappa * s_m)
          ( equivalently  lv_m' = lv_m - log pi_m )
```

There are two ways to set (tau, kappa):

* **(a) Zero-shot.** tau = 3 and kappa = 1. The z-scores are unit-scaled by construction, so this works with
  no labels and no corruptions in training.
* **(b) Self-play.** Learn tau and kappa in stage 2 with *free* corruption augmentation: in 20% of training clips
  one non-text modality is swapped with another clip's, and the label is kept. Evaluate only on the corruption
  families never used in training (noise A, noise channel B, distractor D, text noise), so that the result
  shows generalisation and not memorisation.

### Why it fixes a failure (measurable predictions)

* **Swap (C).** Sigma AUROC is 0.50 and fine-tuned disagreement AUROC is about 0.50. A swapped audio track breaks
  the t<->a and v<->a edges, and the auditor is a frozen copy of the world model, so fine-tuning cannot erase it.
  * Predicted AUROC for swap[a] and swap[v]: **0.50 -> 0.70-0.80**.
  * Predicted AUROC for swap[av]: **-> 0.80-0.90**. Both edges into text break, and text is the strongest
    predictor.
* **Text replaced by noise (A, lambda = 1).**
  * Out-edges t->a and t->v and the self edge t->t explode, even though the text head's own sigma *falls*.
    Predicted detection AUROC: **about 0.5 (sigma) -> above 0.95**.
  * With earned-trust PoE, the text fusion weight should go **0.89 -> below 0.2**.
  * MAE should go **0.85 -> about 0.80** (the AV-only floor). Cov90 should go **0.75-0.80 -> about 0.88**.
* **Noise channel (B).** The pure-noise modality predicts nothing and nothing predicts it, so its z-scores are
  permanently high. Predicted precision share: **13-38% -> below 5%** (frozen). The supervised PoE learns 2%
  only by using labels.
* **Clean data cost.** Clean TAV MAE should change by at most 0.01.

### Implementation sketch

* `models.py`: add `class Graph(nn.Module)` holding `ModuleDict` predictors keyed `'t>a'` etc. (MLP 64->128->128,
  mu and lv split, `bound_lv`) and a `ModuleDict` of `Target`s, one per modality.
* `pretrain.py`: add objective `'xwire'`.
  * `_loss_xwire` reuses `_loss_vjepa` for column t and adds the other edges.
  * The EMA loop calls `targets[m].ema_from(experts[m].backbone, mom)` for each m.
  * The checkpoint stores the dict of target states.
* `evaluate.py`: add a `xwire_score(b)` that returns s (B, M). It plugs into the existing swap and noise AUROC
  code next to "sigma" and "disagreement".
* Fusion: build `lvs' = lvs - log pi` *before* calling `Fusion`, so the fusion API is unchanged and every
  fusion type (`poe`, `attn`, `mpoe`) can be "trusted".
* For fine-tuned models, keep a frozen copy of the stage-1 experts and graph as the auditor. This adds one
  extra forward pass.

**Compute.** Three EMA targets and nine small heads make stage 1 about 1.5-2x the current V-JEPA time. That is
still minutes per seed on a P100. Inference cost is negligible.

**Go/no-go check in week 1.** On validation, compare r_ij for matched clips with r_ij for clips shuffled within
the batch. If AUROC(matched vs shuffled) is below 0.6 for the a<->v edges, the audio-video edges are too weak.
In that case keep only the edges into and out of text and report that honestly.

### Risks

* COVAREP and FACET carry little mutual information on MOSEI, so the a<->v edges may be uninformative. Swap
  detection then depends on the edges through text (still informative, because text is the strong predictor).
* With M = 3, single-corruption localisation is identifiable but double corruption (swap[av]) is ambiguous.
  "Text alone is consistent with neither" looks the same as "text is the liar". The self edge and the clean
  calibration break the tie in practice. Report this as a known limit.
* Every backbone is now a target, so collapse is a risk. VICReg on every backbone, plus `collapse_diag` per
  target, guards against it.

### Closest prior work and difference

* Audio-visual forensics by sync and anomaly detection (for example Feng, Chen & Owens, CVPR 2023) detects
  *one* specific mismatch (lip-sync) with a dedicated model. CROSSWIRE is a general M x M *probabilistic*
  residual graph inside the fusion model, and it feeds the fusion weights.
* ACR ("Adaptive Confidence Regularization for Multimodal Failure Detection", arXiv 2603.02200) uses feature
  swapping and supervised confidence losses. It detects failures, not *which* modality, and it needs labels.
* "Adaptive Modality Reliability Diagnosis and Restoration" (Kumar et al., arXiv 2608.03475) learns
  reliability by regressing on synthetic corruption severities, using energy, ensemble disagreement, JS distance
  to consensus and embedding norm. It has no learned cross-modal *predictors* and no graph localisation.
* The project's own `rpoe` uses only leave-one-out distance between predicted *means*. A corrupted text expert
  still emits a plausible mean, which is why `rpoe` cannot see it. The out-edge t->a compares text's prediction
  with audio's *observed* latent.

---

## 2. EXPLAIN-AWAY INFERENCE (EAI): test-time predictive coding with lying switches

**Principle.** Fusion is inference in a small world model rather than a weighted average. The fused belief is the
latent that best explains *all observed modality latents* both bottom-up and top-down. Each modality has a
binary "this sense is lying" switch, and the switch posterior is solved exactly at test time.

### Generative model

There is a shared latent z ~ N(m0, P0^-1), where the prior is the target marginal already computed in
`target_marginal`. Each modality m has a switch S_m in {0, 1}, where 1 means reliable. If S_m = 1, modality m
supplies two observations of z:

* **Bottom-up:** mu_m ~ N(z, Sigma_m). This is the expert's belief, as in PoE.
* **Top-down:** e_m ~ N(A_m z + b_m, Psi_m). Here e_m = T_m(x_m) is the modality's *own* target latent from
  CROSSWIRE, and (A_m, b_m, Psi_m) is a *linear* decoder trained in stage 1 to predict e_m from the text
  target e_t. Its closed form is ridge regression on the cached training latents.

If S_m = 0, both observations come from a broad outlier density, with a constant cost c_m.

### Exact inference

Because the decoders are linear, each of the 2^M switch configurations S has a closed-form Gaussian posterior:

```
P_S   = P0 + sum_{m: S_m=1} ( Sigma_m^-1 + A_m^T Psi_m^-1 A_m )
mu_S  = P_S^-1 [ P0 m0 + sum_{m: S_m=1} ( Sigma_m^-1 mu_m + A_m^T Psi_m^-1 (e_m - b_m) ) ]
logp(S | x) = log evidence_S (Gaussian marginal likelihood, closed form)
              - sum_{m: S_m=0} c_m + log prior(S)
```

The fused belief is the moment-matched mixture sum_S p(S|x) N(mu_S, P_S^-1). It plugs straight into `Head`.
The per-modality corruption probability p(S_m = 0 | x) is a calibrated, label-free detector.

* With M = 3 there are 8 configurations: batched 64x64 solves, which is trivial.
* With a 4th noise channel there are 16.
* c_m is set label-free as the 99th percentile of modality m's clean validation energy.

There is also an iterative variant, true predictive coding. If the decoders are made small MLPs, z and pi are
updated by alternating gradient and sigmoid steps:

* z <- z - eta * grad_z E(z, pi)
* pi_m = sigmoid((c_m - E_m(z)) / T)

This runs 10 steps from the PoE initialisation. The linear exact version is the one to build; the iterative
version is an ablation.

### Why it fixes the failure

The central failure is a confident-but-wrong expert that wins PoE because of its precision. EAI decides which
modality to trust by **total explanatory consistency**, not precision. When text is noise:

* **Configuration "text on":** z is pinned near mu_t, and then both e_a and e_v are badly explained by A_a z and
  A_v z. That costs two large top-down terms.
* **Configuration "text off":** this pays a single c_t. Audio and video agree with each other and with a z
  inferred from them.

Majority-consistency emerges from Bayes and needs no hand-made rule. This is also the "explaining away" effect:
once the switch absorbs the text observation, the evidence for the latent is re-weighted.

**Predictions.**

* noise[T]@1.0 (frozen V-JEPA)
  * p(S_t = 0) AUROC: **about 0.50 -> above 0.95**
  * MAE: **0.85 -> about 0.80-0.82** (the AV-only floor)
  * Cov90: **0.75-0.80 -> 0.87-0.90**
* swap[a] and swap[v]: AUROC **-> 0.70-0.80**
* Noise channel B: its share drops below 5%.
* Clean TAV: MAE within 0.01 and Cov90 unchanged. The false-alarm rate of p(S_m = 0) > 0.5 on clean clips is
  set to about 1% by c_m.
* Tracking ratio under text noise: **0.61-0.90 -> about 1.0**. When the switch goes off, the variance becomes the
  AV-only variance, which matches the AV-only error.

### Implementation

* Write a new file `fusionlab/eai.py`:
  * `fit_decoders(cache_train, e_train)` does ridge regression per modality and stores A, b, diag Psi.
  * `eai_fuse(mus, lvs, e, av)` enumerates the configurations and returns `(mu, var, share)` with
    `share = p(S=1)`.
* Use diagonal Sigma and Psi, and keep P_S full: it is 64x64, so `torch.linalg.solve` in batch is fine.
* Register `'eai'` in `fusion.LATENT` as a `Fusion` kind with no trainable parameters. Optionally learn only
  `Head`.
* Missing modalities (av = 0) are switch-off with zero cost.

**Compute.** No training beyond CROSSWIRE targets and a closed-form ridge fit (seconds). Inference is 8 small
batched solves.

### Risks

* If A_a and A_v explain little variance (weak AV information), the top-down terms are flat. EAI then falls back
  to "drop whichever modality has the most self-edge surprise". That is still better than PoE, but swap detection
  will be weak.
* The setting of c_m trades false alarms against misses. Report the whole operating curve (AUROC), not one
  threshold.
* The top-down decoder from text space may itself be fooled by distractor-dominated latents. Evaluate under D
  as well.

### Closest prior work

* Robust Bayesian sensor fusion with outlier switches is classical. Examples are Kalman filters with
  switching/outlier models and the "Gaussian scale-mixture" robust PoE. Those use raw sensor models.
* Predictive coding networks perform iterative inference but are not used for multimodal reliability.
* Recent multimodal test-time adaptation work, such as "Calibrating Multimodal TTA under Reliability and
  Availability Shifts" (arXiv 2611.03232) and attention bootstrapping (AAAI 2025), *updates weights* with
  entropy objectives.
* **Novelty here:** exact switch-posterior inference in a *learned JEPA latent world model*, using each
  modality's *own* target latent as the top-down check. No weights are adapted and no labels are needed. I found
  no prior work combining these.

---

## 3. TIME AUDITOR: token-level temporal JEPA surprise as the missing epistemic signal

**Principle.** A Gaussian head can't tell that its input is alien, but a model of how a clip *evolves* can. Each
expert also predicts its own masked *frames'* latents from their neighbours. The clip's temporal surprise is an
input-level, label-free OOD score that a pooled sigma never sees.

### Math

Use the transformer's un-pooled frame states H_m in R^{L x d_model}. Add a light frame target T_m^frame, an EMA
of the per-frame states, and a predictor q_m that, given the context frames, predicts masked spans of frames
(I-JEPA/data2vec style, spans of 5 frames, 40% masked):

```
L_time = sum_m  mean_{t in masked}  NLL( sg T_m^frame(x_m)[t] | q_m(H_m(ctx))[t] )
S_m^time(x) = mean_t  (1/d) || (e_mt - mu_mt) / sigma_mt ||^2 ,  standardised by clean median/MAD
```

S_m^time is added to the CROSSWIRE node score (s_m <- s_m + S_m^time). EAI can also use it inside c_m.

### Why it fixes a failure

* Text replaced by noise has frames that are i.i.d. across time. Masked spans cannot be predicted from their
  neighbours, whereas real BERT, COVAREP and FACET frames are strongly autocorrelated. Surprise explodes even
  though the pooled head's sigma *falls* (0.068 -> 0.061).
* Predicted results:
  * AUROC for noise[T]@1: **about 0.50 -> above 0.97**.
  * AUROC at lambda = 0.5: **-> 0.85**.
  * Spearman(score, |error|) pooled over noise conditions: **about 0 (frozen sigma) -> 0.3-0.4**.
* **Complementarity is the jury figure.** The time auditor *cannot* see swaps: a swapped clip is a perfectly
  coherent clip. CROSSWIRE *can*. The noise case is the reverse. A 2x3 table
  {time, senses, both} x {noise, swap, noise channel} shows two orthogonal auditors and why both are needed.
  The predicted swap AUROC for the time auditor is about 0.5, and this is a *pre-registered* negative.

### Implementation

* `SeqEncoder.forward` currently pools. Add a `return_frames=True` path that returns `h` before pooling.
* Add `q_m`, a 1-layer transformer over [context states + mask tokens], and a frame-level EMA target, `Target`
  over the `SeqEncoder` with no pooling.
* Implement as objective `'xwire_time'` or as an extra loss term with weight w_time = 0.5.

**Compute.** Plus 30-50% stage-1 time, which is fine.

### Risks

* With aligned_50, many clips are heavily padded (short utterances). Use only valid frames, and down-weight
  clips with fewer than 10 valid frames.
* The AR(1) distractor is *more* temporally predictable than the signal. The time auditor will see distractor
  clips as "normal". That is correct behaviour (D is a train-and-test distribution, not a corruption) but must
  be stated.
* Partially overlaps with the Bayesian researcher's epistemic ideas. The difference is that this is a
  *dynamics-model surprise*, not a posterior over weights.

### Prior work

* Frame-level JEPA (V-JEPA video, data2vec) is used for representation learning, not as a per-sample
  reliability score in fusion.
* Reconstruction-error OOD detection is the closest idea. The difference is that the error here is in latent
  space, predicted from context, and standardised per modality.

---

## 4. PERSISTENCE TARGETS ("two senses, two times"): anti-hijacking by construction

**Principle.** A latent counts as a target only if a *different modality* can predict it from a *different, distant
time window* of the clip:

* Transient nuisance that is sticky but local in time cannot cross the time gap.
* Persistent nuisance confined to one sense cannot cross the modality gap.

Only content that is shared across senses and persistent across the clip, such as sentiment, survives.

### Math

Split each clip's *valid* frames into windows W1 (first 40%) and W2 (last 40%), with a gap of about 20% between
them. The gap matches the distractor's correlation time 1/(1 - rho) = 10 frames. Then:

```
L_persist = sum_{i} sum_{j != i or same-i-allowed?} sum_{(W,W') in {(1,2),(2,1)}}
              NLL( sg T_j(x_j restricted to W')  |  g_{i->j}( h_i(x_i restricted to W) ) )
            + w_vic * VICReg(h)  +  (existing PoE-subset term on the same cross-window targets)
```

* The fusion latent stays the text target space, but it is now T_t(text in the *other* window).
* No same-window edges are used.
* Two variants:
  * **P-time:** same modality allowed if cross-window.
  * **P-full:** cross-modal and cross-window only.

At test time the online encoder sees the full clip. During training, randomly pick (W, W') per batch, and also
feed the online encoder "full clip minus the target window and the gap" (I-JEPA context masking), so that the
test-time input is not out of distribution.

**Why the numbers favour it.** I simulated the project's own AR(1) process (rho = 0.9, L = 50, 20k clips). The
correlation between the distractor's means in two windows is:

| windows | corr of window means | R^2 of full-clip mean from one window |
|---|---|---|
| adjacent halves [0,25) / [25,50) | 0.25 | 0.62 |
| [0,20) / [30,50) (10-frame gap) | **0.10** | 0.51 |
| [0,15) / [35,50) (20-frame gap) | 0.04 | 0.40 |

With a 10-frame gap, at most about 1% of a window's distractor variance is predictable in the other window. The
target therefore *cannot* reward encoding the distractor, even when it is **shared** across modalities: shared
helps it cross senses but not time. Sentiment is a clip-level label, so it is constant across windows by
definition.

### Why it fixes the failure (predictions at D, sigma = 8)

| metric | current | predicted |
|---|---|---|
| signal R^2, shared distractor | 0.07 | **0.20-0.30** (supervised: 0.47) |
| distractor R^2, shared | 0.52 | **below 0.10** |
| distractor R^2, independent | 0.18 | **below 0.05** |

On clean data the frozen MAE should be unchanged or slightly worse: 0.70-0.73 -> 0.70-0.75. Windows see less
text. That cost is the price of robustness; report it.

**Extra evaluation, cheap and strong.** Port the idea to Huang's toy (`huang.py` already executes his code). His
VJEPA uses a next-step target. Replace it with a target from a gapped future window and run 15 seeds. The
prediction is that it beats VJEPA's distractor R^2 in at least 12/15 seeds. This directly answers "is the noisy-TV
problem solved?" on the field's own benchmark.

### Implementation

* `batching.py`: `window_masks(mask, frac=0.4, gap=0.2)` returns boolean masks for W1 and W2 over the valid
  frames. The existing `src_key_padding_mask` path handles it, so no model change is needed.
* `pretrain.py`: objective `'persist'`. The target is computed with `b['tm'] & W'`, and the context with
  `random_token_mask(b[m+'m'] & ~(W' or gap))`. Everything else reuses `_loss_vjepa` (column t) and the CROSSWIRE
  heads (other columns).
* Run in the existing D grid (`dist_patterns`, `dist_sigmas`). The probe code in `evaluate.py` (signal and
  distractor R^2) already exists.

**Compute.** The same as V-JEPA plus CROSSWIRE: two windows per batch is two target passes.

### Risks

* Real MOSEI features within 20-frame windows may carry less sentiment, so the clean probe R^2 could drop.
  Mitigate with 3 random window pairs per clip.
* A distractor with *clip-constant* shared nuisance (rho -> 1) defeats this, and no label-free method can tell
  it apart from content. State this as a theorem-style limit: without labels, persistent AND shared AND
  predictable is indistinguishable from signal. It makes the claim honest and sharp.
* Speaker identity is persistent and shared, so it will also be kept.

### Prior work

* Slow feature analysis and temporal-contrastive learning (CPC, TCN) reward persistence, but within one stream,
  and they *do* encode sticky nuisance.
* "Contrast encodes inductive bias: separating slow noise from dynamics" (arXiv 2606.07770) shows that JEPA and
  InfoNCE organise around per-trajectory slow nuisance. Its fix is within-trajectory negatives, for dynamics
  learning. That is the mirror image of our goal (keep the clip-level content, drop transient nuisance).
* Huang (2026, VJEPA, arXiv 2601.14354) motivates the noisy-TV test but uses a same-stream next-step target.
* Novelty: a JEPA *target* defined by a cross-modal **and** cross-time-gap constraint, with the gap set by the
  nuisance's correlation time, plus an explicit identifiability limit. I found no prior multimodal JEPA doing
  this.

---

## 5. KEEP-ALIVE FINE-TUNING: the world model stays on while the label is learned

**Principle.** Fine-tuning destroys cross-modal agreement because nothing asks the latent to keep *predicting*
anything. Keep the label-free predictive losses running during stage 2, so the fine-tuned latent remains a
calibrated world model whose disagreements still mean something.

### Math

In stage 2 with `frozen=False`:

```
L = NLL_label(y | Head(Fusion(beliefs)))
  + lambda_wm * [ L_xwire (or L_persist) on the same batch, EMA targets keep updating at mom 0.999 ]
  + lambda_cal * ( mean_k (e_t - mu_m)^2 / sigma_m^2  - 1 )^2          # keep latent Mahalanobis/d ~ 1
```

* The last term is a cheap "latent calibration anchor". The brief notes that frozen V-JEPA has
  Mahalanobis/d of 0.98, and this keeps it there.
* lambda_wm is chosen on validation NLL from {0.1, 0.3, 1}.

### Why it fixes a failure (predictions)

| metric | current | predicted |
|---|---|---|
| fine-tuned disagreement AUROC on swaps | about 0.50 | **0.75-0.85** (the frozen range of 0.56-0.87, with the frozen auditor no longer needed) |
| fine-tuned MAE (TAV) | 0.56 | 0.56-0.57 (no accuracy loss) |
| worst-subset Cov90 when text is missing | 0.68-0.70 | **0.78-0.85** |
| AV-only MAE | 0.81 | about 0.78-0.79 (speculative) |

* The worst-subset Cov90 improves because the a and v experts are still held to be calibrated predictors of the
  text latent.
* The AV-only MAE is speculative. The cross-modal term keeps distilling the text latent into the weak experts
  during fine-tuning (limitation 5).

### Implementation

* `stage2.py`: pass `targets` and `graph` into `stage2`. In the non-frozen branch, compute the stage-1 loss on
  the same `b` and add it.
* After each `opt.step()`, call EMA. That is about 15 lines.

**Compute.** About 2x the fine-tuning time, which is minutes.

### Risks

* This is the least novel idea. Auxiliary SSL during fine-tuning is known, for example multi-task SSL
  (Hendrycks et al. 2019) and test-time training (Sun et al. 2020).
* The new angle is preserving *cross-modal predictive agreement* specifically as a reliability signal for
  fusion, with a measured before/after on swap AUROC. Treat it as an enabling component, not the headline.

---

## 6. (Bonus, systems) AUDITOR/WORKER SPLIT

This is not a separate method but a design rule that makes ideas 1-3 robust. The **worker** (fine-tuned experts
and head) maximises accuracy. The **auditor** (frozen stage-1 world model: CROSSWIRE graph, time auditor, EAI
decoders) never sees labels. It only sets trust weights pi_m and the switch posterior. It costs one extra
forward pass.

This turns "fine-tuning destroys agreement" (limitation 4) from a problem into an ablation: fine-tuned
auditor vs frozen auditor vs keep-alive auditor (idea 5).

---

## Novelty check (brief web search, Nov 2026)

* [ACR, arXiv 2603.02200](https://arxiv.org/html/2603.02200): failure detection with feature swapping, but
  supervised and confidence-based. It is not a cross-modal predictor graph and does not localise the corrupted
  modality.
* [Adaptive Modality Reliability Diagnosis and Restoration, arXiv 2608.03475](https://arxiv.org/pdf/2608.03475):
  trained on synthetic corruptions, with cues from energy, ensembles, JS-to-consensus and embedding norm, plus
  cross-modal *restoration*. It is the nearest competitor to CROSSWIRE and EAI. Differences: it is not
  label-free, has no learned M x M latent predictors, and no exact switch posterior. Cite it and compare to it.
* [Reliability-Aware Probabilistic Fusion with Incomplete Evidence, arXiv 2610.29621](https://arxiv.org/pdf/2610.29621)
  and [RaGCD, arXiv 2611.13827](https://arxiv.org/pdf/2611.13827): reliability-weighted fusion. Neither is built
  on predictive cross-modal residuals.
* [Calibrating Multimodal TTA under Reliability/Availability Shifts, arXiv 2611.03232](https://arxiv.org/pdf/2611.03232)
  and [Attention Bootstrapping for Multimodal TTA](https://arxiv.org/html/2503.02221v1): these adapt weights at
  test time. EAI adapts *beliefs and switches* only, with exact inference.
* [Contrast encodes inductive bias: slow noise vs dynamics, arXiv 2606.07770](https://arxiv.org/html/2606.07770):
  confirms that predictive SSL is hijacked by slow nuisance. Its fix is opposite in spirit to Persistence
  Targets.
* [VJEPA, Huang, arXiv 2601.14354](https://arxiv.org/html/2601.14354): the noisy-TV baseline.
* [Defending multimodal fusion against single-source adversaries, arXiv 2206.12714](https://ar5iv.labs.arxiv.org/html/2206.12714):
  odd-one-out detection with a *supervised* network, under adversarial perturbations. It is the conceptual
  ancestor of graph localisation. Differences: it is not predictive, not probabilistic, and not label-free.

No search hit combined: label-free M x M **JEPA latent prediction residuals** -> modality localisation ->
precision re-weighting; or exact switch inference with own-modality top-down checks; or cross-modal plus
cross-time-gap JEPA targets.

## Suggested FYP-2 package and evaluation table

* **Headline:** CROSSWIRE + EAI ("a world model that audits its own senses").
* **Second contribution:** Persistence Targets (anti-hijacking).
* **Enabling components:** Time Auditor and Keep-Alive.

All of these fit in 6 months. Stage-1 runs are minutes per seed, so 5 seeds x 3 objectives x D-grid fits within
a couple of Kaggle weeks.

| Failure (brief) | Metric | Now | Predicted | Idea |
|---|---|---|---|---|
| text -> noise, sigma blind | detection AUROC | about 0.50 | above 0.95 | 1, 2, 3 |
| text -> noise | MAE / Cov90 | 0.85 / 0.75-0.80 | about 0.81 / about 0.88 | 2 (1) |
| swap (C), fine-tuned | AUROC | about 0.50 | 0.70-0.90 | 1, 2, 5 |
| noise channel (B) | precision share | 13-38% | below 5% | 1, 2 |
| shared distractor, sigma = 8 | signal / distractor R^2 | 0.07 / 0.52 | 0.20-0.30 / below 0.10 | 4 |
| text missing, fine-tuned | worst Cov90 | 0.68-0.70 | 0.78-0.85 | 5 |
| clean | MAE | 0.56 (FT) | within 0.01 | all (no-harm check) |
