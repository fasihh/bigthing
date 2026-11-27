# Literature scout: trends and gaps for FYP-2 (as of Nov 2026)

Scope: the five threads named in the brief, focused on 2025-2026 work. I opened every paper listed here (arXiv abstract or HTML page, or the proceedings page) during this survey. "L#" refers to the numbered limitations in `BRIEF.md`:

- L1: the predicted sigma is aleatoric only.
- L2: PoE trusts precision blindly.
- L3: the EMA target gets hijacked by the distractor (noisy TV).
- L4: fine-tuning erases the self-supervised structure.
- L5: text dominates the fusion.
- L6: results are unstable across seeds.
- L7: calibration shifts across modality masks and under corruption.

Caveat: many of these are fresh arXiv preprints with no peer review yet. The summaries come from the abstracts, plus the HTML text where I note it. Treat any claimed numbers as the authors' own.

---

## 1. Probabilistic / uncertainty-aware JEPA and latent world models

| Paper | Year / id | Finding (1-2 sentences) | L# |
|---|---|---|---|
| [VJEPA: Variational JEPA as Probabilistic World Models](https://arxiv.org/abs/2601.14354), Y. Huang | 2026, arXiv 2601.14354 (ICML 2026 poster) | Learns a predictive distribution over future latents with a variational objective. BJEPA fuses a dynamics expert with a prior expert through a PoE. The paper claims VJEPA filters high-variance nuisance distractors that collapse generative baselines. This is the paper the brief reproduces; FYP-1 found the claim fails for shared or sticky multimodal distractors. | L1, L3, L6 |
| [MDN-VJEPA: Multimodal Probabilistic Joint-Embedding World Models](https://arxiv.org/abs/2610.13956), Y. Huang | 2026, arXiv 2610.13956 | Swaps the Gaussian latent predictor for a mixture density network and adds trajectory-coherence and Chapman-Kolmogorov consistency terms. It recovers branching predictive laws, and keeping the full mixture improves planning. The abstract reports no distractor results. | L1 |
| [Var-JEPA](https://arxiv.org/abs/2603.20111), M. Gögl, C. Yau | 2026, ICML 2026 (PMLR 306), arXiv 2603.20111 | Recasts JEPA as a coupled latent-variable model trained with a single ELBO (no ad-hoc anti-collapse terms). Its native latent uncertainty beats MC-dropout and ensembles on Spearman/AUC for ambiguity detection, but loses on AURC. There is **no OOD evaluation**; the tests use only simulated ambiguity. | L1 |
| [S-JEPA: A Stochastic Latent Variable Generalization of JEPA](https://arxiv.org/abs/2611.09302), N. Giakoumoglou et al. | 2026, arXiv 2611.09302 (code says WACV 2027) | The predictor's conditioning variable z becomes stochastic: a posterior q(z \| ctx, tgt) during training and a prior p(z \| ctx) at test time. This gives error-correlated uncertainty maps and diverse completions without hurting the representation. It targets ambiguity (aleatoric multimodality), not OOD. | L1 |
| [Flow-JEPA](https://arxiv.org/abs/2608.29029), Y. Huo et al. | 2026, arXiv 2608.29029 | Finds that existing JEPA world models degrade badly under localized OOD visual noise. A conditional flow-matching latent dynamics model (Gaussian source to clean future latents) raises success under noise from 67% to 86%. | L1, L3 |
| [JEDI: Joint Embedding Diffusion World Model](https://arxiv.org/abs/2605.13013), J. Y. Lim et al. | 2026, arXiv 2605.13013 | End-to-end latent diffusion world model trained JEPA-style. It argues that JEPA objectives induce a predictive information bottleneck and that diffusion denoising has a matching decomposition. | L1 |
| [N-JEPA: Improving JEPA with Diffusion Noise](https://arxiv.org/abs/2507.15216), Y. Qiu et al. | 2025, arXiv 2507.15216 | Treats diffusion noise as a masking state and applies a multi-level noise schedule as augmentation for robustness. | L3 |
| [LeJEPA](https://arxiv.org/abs/2511.08544), R. Balestriero, Y. LeCun | 2025, arXiv 2511.08544 | SIGReg pushes embeddings toward an isotropic Gaussian, with one trade-off hyperparameter and no EMA teacher or stop-gradient. Claims stability across architectures and domains. | L6 (and L3, since it removes the EMA target) |
| [AV-JEPA: Extending LeJEPA to Audio-Visual SSL](https://arxiv.org/abs/2607.15295), B. Robson et al. | 2026, arXiv 2607.15295 (ICML 2026) | Multimodal LeJEPA with modality dropout as the masking strategy. It drops the EMA teacher, the decoder and the negatives. | L3, L6 |
| [Le MuMo JEPA](https://arxiv.org/abs/2603.24327), C. Cornelissen et al. | 2026, arXiv 2603.24327 | Multimodal LeJEPA with learnable fusion tokens as a cross-modal bottleneck (RGB+LiDAR, RGB+thermal). | L5 |
| [Biased Dreams: Limits to Epistemic UQ in Latent Dynamics Models](https://arxiv.org/abs/2604.25416), J. Berger et al. | 2026, arXiv 2604.25416 | **Key negative result.** In Dreamer's RSSM, ensemble disagreement captures local uncertainty but misses compounding error. An "attractor effect" pulls rollouts into well-supported latent regions where uncertainty looks low. The authors say this "cannot be resolved through improved uncertainty estimation alone". | L1 |
| [Epistemic Uncertainty Is Not the Reducible Kind](https://arxiv.org/abs/2606.12646), R. Young | 2026, arXiv 2606.12646 | Theory: ensemble disagreement mostly reflects the training procedure (it goes to 0 under consistent training and reflects initialization noise under interpolation). | L1, L6 |
| [Position: Epistemic UQ methods are fundamentally incomplete](https://arxiv.org/abs/2505.23506), S. Jiménez et al. | 2025, arXiv 2505.23506 | Second-order methods inflate the aleatoric estimate and deflate the epistemic one, because of unaccounted bias. Each method captures only part of the epistemic variance. | L1 |
| [Uncertainty-aware Latent Safety Filters](https://arxiv.org/abs/2505.00779), J. Seo et al. | 2025, CoRL 2025 | Uses world-model epistemic uncertainty as an OOD proxy and calibrates the OOD threshold with conformal prediction. | L1, L7 |
| [Detection Is Free, Diagnosis Is Not](https://neurips.cc/virtual/2026/166027), P. Wasinger et al. | NeurIPS 2026 | In JEPA-style world models, label-free SSL metrics detect transfer failures but cannot localize them (encoder vs predictor). | L1, L4 |
| [Zero-Label Driving Scenario Complexity via JEPA](https://arxiv.org/abs/2606.28383), S. Jaiswal | 2026, arXiv 2606.28383 | JEPA temporal prediction error used as a label-free complexity/anomaly score (AP 0.512 vs 0.436 chance). Shows that latent prediction residual is a weak but real signal. | L1 |

## 2. The noisy-TV / distractor problem in self-predictive learning

| Paper | Year / id | Finding | L# |
|---|---|---|---|
| [Contrast encodes inductive bias: separating slow noise from dynamics](https://arxiv.org/abs/2606.07770), P. Gulati, I. Nemenman | 2026, arXiv 2606.07770 | **Most relevant to L3.** Noise that is constant within a trajectory gets encoded when negatives come from *other* trajectories, because it separates positives from negatives. Taking negatives from *within* the same trajectory removes the shortcut. Limits (from the HTML): there is still an error peak when the noise correlation time τ ≈ trajectory length T, and the analysis covers InfoNCE only. The authors state explicitly that whether the fix transfers to non-contrastive/EMA JEPA is **an open question**. | L3 |
| [How JEPA Avoids Noisy Features (deep linear self-distillation)](https://arxiv.org/abs/2407.03475), E. Littwin et al. | 2024, NeurIPS 2024 | JEPA prioritizes high-regression-coefficient ("influential") features, whereas MAE prioritizes high-covariance features. JEPA learns noisy features more slowly. This explains why JEPA ignores *unpredictable* noise. A *predictable* (sticky) distractor has a high regression coefficient, so the theory predicts the hijack FYP-1 observes. | L3 |
| [Is the reconstruction loss culprit? An attempt to outperform JEPA](https://arxiv.org/abs/2603.14131), A. Potapov et al. | 2026, arXiv 2603.14131 | On Huang's "TV-series" linear testbed, JEPA's apparent edge over autoencoders comes from bottleneck and component-selection asymmetries. Gated predictive autoencoders that pick predictable components match or beat JEPA. | L3, L6 |
| [JEPA-Bisim](https://arxiv.org/abs/2602.18639), L. Toso et al. | 2026, arXiv 2602.18639 | Adds a bisimulation encoder to a DINO-WM-style JEPA so that states with similar control dynamics are grouped. It is more robust to background and distractor changes. This needs a reward/control signal. | L3 |
| [Dueling World Models: Advantage-Style Action Channels for Common-Mode Distractor Rejection](https://arxiv.org/abs/2608.06706), J. Li et al. | 2026, arXiv 2608.06706 | Subtracts the mean prediction across actions, so any action-independent (common-mode) variation, including distractors, cancels exactly. It cannot reject distractors that track the action. **This is a direct analogue of the shared-across-modalities distractor (D-shared), but nobody has done the cross-modal version.** | L3 |
| [Predictive Objectives Discard Exogenous Control-Relevant Features](https://arxiv.org/abs/2606.30068), A. Pendharkar | 2026, arXiv 2606.30068 | All reward-free predictive objectives lose exogenous but relevant features. Adding 2% reward-labelled transitions restores them. Lesson: a small amount of labels re-anchors what is "relevant". | L3, L4 |
| [StarWM: Self-Supervised Attention Routing for Robust World Models](https://arxiv.org/abs/2609.30667), Z. Zhang et al. | 2026, arXiv 2609.30667 | Uses self-supervised dynamics to learn *where* reconstruction should apply, with stop-gradients between the two objectives. Robust to random-frame and sequential-video distractors. | L3 |
| [CsDreamer: Leveraging Conditional Dependence for World Model Denoising](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6f01b2a921c0a4dc34609613f4f74b8e-Abstract-Conference.html), S. Zhang et al. | NeurIPS 2025 | Models the collider dependence between task-relevant and irrelevant latents during inference, instead of assuming they are independent. Better under heavy noise. | L3 |
| [Beyond Noisy-TVs: Learning Progress Monitoring](https://arxiv.org/abs/2509.25438), Z. Hou et al. | 2025, arXiv 2509.25438 | Exploration angle: rewarding the *improvement* of the dynamics model's error, rather than the error itself, is noise-robust and linked to information gain. | L1, L3 |
| [CIG: Exploration via Conditional Information Gain](https://arxiv.org/abs/2605.20878), T. Joseph et al. | 2026, arXiv 2605.20878 | A log-det of an ensemble disagreement kernel, robust to stochastic distractors in model-based RL (MBRL). | L1, L3 |
| [Do-JEPA: From Masking to Intervention](https://arxiv.org/abs/2609.37378), H. Resani, J. Shi | 2026, arXiv 2609.37378 | Predicts the *difference* between latent futures with and without an action, which separates causal effect from co-occurrence. | L3 |

**What actually solves noisy TV (as of 2026):** it depends on the kind of distractor.

- **Unpredictable i.i.d. noise:** predictive or JEPA losses already ignore it (Littwin).
- **Predictable or slow nuisance:** solved only by injecting an *invariance signal* that the nuisance does not have. Examples:
  - within-trajectory contrast (Gulati);
  - action or control structure (Dueling WM, Bisim, Do-JEPA);
  - a small amount of reward or labels (Pendharkar).
- **Variational, stochastic, flow or diffusion predictors (VJEPA, S-JEPA, Flow-JEPA):** they change *how* the target is predicted, not *what* the target encodes. None of them claims to solve a shared predictable nuisance.

## 3. Uncertainty-aware multimodal fusion under corrupted / conflicting modalities

| Paper | Year / id | Finding | L# |
|---|---|---|---|
| [Reliability-Aware Probabilistic Fusion for Multimodal Learning with Incomplete Evidence](https://arxiv.org/abs/2610.29621), P. Palo et al. (Clifton lab) | 2026, arXiv 2610.29621 | **Gaussian PoE with an input-dependent reliability weight that tempers each modality's precision**, plus missing-aware training (clinical, HAIM-MIMIC-MM). Under unseen test-time corruptions, reliability weights and latent uncertainty "respond differently" and **the method fails under severe degradation**. The authors say dependable quality signals are an open challenge. | L1, L2 |
| [Multimodal Learning with UQ based on Discounted Belief Fusion](https://proceedings.mlr.press/v258/bezirganyan25a.html), G. Bezirganyan et al. | AISTATS 2025 (PMLR 258) | Order-invariant evidential fusion with **conflict-based discounting**, which redistributes belief away from a modality that looks unreliable. It detects conflicting samples better than averaging or accumulation. | L2 |
| [PRA-PoE](https://arxiv.org/abs/2605.13081), G. Yang et al. | 2026, arXiv 2605.13081 | Uncertainty-aware Gaussian PoE plus prototype-anchored alignment across modality subsets. Evaluated on every modality subset (ADNI, OASIS-3). | L2, L7 |
| [Context-specific Credibility-aware Multimodal Fusion (C²MF)](https://arxiv.org/abs/2603.26629), P. Tenali et al. | 2026, arXiv 2603.26629 | Per-instance source credibility via a KL-based score in a conditional probabilistic circuit. Introduces a "Conflict" benchmark with class-specific corruptions and reports up to +29% under heavy noise. | L2 |
| [TIER-MoE](https://arxiv.org/abs/2607.27289), Y. Chang et al. | 2026, arXiv 2607.27289 | Reliability score = the expected unimodal loss, estimated from out-of-fold predictions (needs labels). It is combined with sparse routing and an always-on shared path. | L2, L5 |
| [GIML: General Incomplete Multimodal Learning via Dynamic Quality Perception](https://arxiv.org/abs/2607.06943), X. Meng, S. Wei | 2026, arXiv 2607.06943 | Treats missing and corrupted modalities as degrees of information loss. A noise-aware quality estimator is **trained on injected noise** and paired with noise/semantic decoupling. | L1, L2 |
| [UDML: Unbiased Dynamic Multimodal Fusion](https://arxiv.org/abs/2603.19681), S. Wei et al. | 2026, arXiv 2603.19681 | Same family: predicts the intensity of injected noise from features. Also corrects for modality-reliance bias, measured by modality dropout, so a hard-to-learn modality is not penalized. | L2, L5 |
| [Before Fusion, Ask What to Keep](https://arxiv.org/abs/2606.02679), J. Liu et al. | 2026, arXiv 2606.02679 | A pre-fusion plug-in that compares modality summaries to detect support or contradiction, then scales features per instance and per dimension. Includes sentiment benchmarks and corrupted-input tests. | L2 |
| [TAME: Text-Anchored Modality Experts with Uncertainty-Guided Routing](https://arxiv.org/abs/2610.27571), X. Huang et al. | 2026, arXiv 2610.27571 | **On MOSI, MOSEI and CH-SIMS.** Text is a fixed, always-on shared expert; A-T and V-T experts are routed by per-sample uncertainty from "Prophet Heads". The abstract gives no corruption experiments. | L5, L2 |
| [Enhance-then-Balance Modality Collaboration (EBMC)](https://arxiv.org/abs/2604.12518), K. He et al. | 2026, arXiv 2604.12518 | MSA: strengthens weak non-verbal modalities, uses energy-guided gradient rebalancing, and adds instance-level trust weights. Reports robustness to missing modalities. | L5 |
| [Conflict-aware Penalty + Statistical Loss for MSA](https://arxiv.org/abs/2605.28575), J. Dai et al. | 2026, arXiv 2605.28575 | Text from a strong pre-trained encoder dominates and causes gradient conflicts. Penalizes gradient-norm conflicts (MOSI). | L5 |
| [MCAN: Multi-level Conflict-Aware Network for MSA](https://arxiv.org/abs/2502.09675), Y. Gao et al. | 2025, arXiv 2502.09675 | Separates agreeing from conflicting unimodal and bimodal components, with discrepancy constraints (MOSI, MOSEI). | L2 |
| [RCE: Reliability-aware Cross-sample Enhancement for Robust MSA](https://arxiv.org/abs/2609.30470), M. Jiang et al. | 2026, arXiv 2609.30470 | Per-modality uncertainty from an adaptive variational information bottleneck (VIB), plus retrieval of similar confident samples to fill missing modalities. Evaluated in noisy and missing settings. | L1, L2 |
| [TMDC: Two-stage Modality Denoising and Complementation](https://arxiv.org/abs/2511.10325), Y. Zhuang et al. | 2025, arXiv 2511.10325 | Handles missing and noisy MSA jointly (MOSI, MOSEI, IEMOCAP). | L2 |
| [PRLF: Progressive Representation Learning for MSA with Incomplete Modalities](https://arxiv.org/abs/2603.09111), J. Bao et al. | 2026, arXiv 2603.09111 | A reliability estimator from recognition confidence plus Fisher information picks the dominant modality, then the others are progressively aligned to it (MOSI, MOSEI, SIMS). | L2, L5 |
| [FedUAF](https://arxiv.org/abs/2603.13291), X. Zhu et al. | 2026, arXiv 2603.13291 | Per-modality uncertainty plus reliability-guided aggregation for federated MSA (MOSI, MOSEI). | L2 |
| [SURE: multimodal pretraining with missing modalities via uncertainty](https://arxiv.org/abs/2504.13465), D. A. Nguyen et al. | 2025, arXiv 2504.13465 | Reconstructs missing modalities in latent space and propagates their uncertainty to the prediction, using a Pearson-correlation loss. Includes sentiment analysis. | L1, L7 |
| [Hidden in the Multiplicative Interaction (Gated Symile)](https://arxiv.org/abs/2604.05834), T. Rheude et al. | 2026, arXiv 2604.05834 | In multiplicative (multilinear) fusion, one weak or misaligned modality distorts everything. Gating pulls unreliable embeddings toward a learned neutral direction. **This is the same pathology as PoE.** | L2 |
| [UAF: Uncertainty-Aware Crossmodal Fusion (animal behaviour)](https://arxiv.org/abs/2608.02104), E. Yaghoubi, F. Haselbeck | 2026, arXiv 2608.02104 | Gaussian per-stream uncertainty weighting without reliability labels. Shows the "precision-weighted fusion" recipe is now commodity. | L2 |
| [SuMi: Stable TTA under Complex Multimodal Noises](https://arxiv.org/abs/2503.02616), Z. Guo, T. Jin | 2025, arXiv 2503.02616 | "Multimodal wild TTA": interquartile-range (IQR) smoothing, unimodal-guided low-entropy sample selection, and mutual-information sharing. | L1, L4 |
| [BalanceBenchmark (survey)](https://arxiv.org/abs/2502.10816), S. Xu et al. | 2025, arXiv 2502.10816 | Taxonomy and toolkit for multimodal imbalance methods. | L5 |

## 4. Label-free epistemic / OOD / corruption detection

| Paper | Year / id | Finding | L# |
|---|---|---|---|
| [MultiOOD + A2D](https://arxiv.org/abs/2405.17419), H. Dong et al. | 2024, arXiv 2405.17419 (NeurIPS 2024) | The **discrepancy between modality-specific predictions is much larger on OOD than on in-distribution (ID) data**. The A2D training algorithm deliberately enlarges it. | L1, L4 |
| [Modality-Aware OOD Detection for Multi-Modal Action Recognition](https://arxiv.org/abs/2606.24404), L. Doorenbos et al. | 2026, arXiv 2606.24404 | Post-hoc detector that combines the multimodal-vs-unimodal prediction relationship with a feature-manifold score. New SOTA on MultiOOD. | L1 |
| [Teacher-Guided Training (Domain-Sensitivity Collapse)](https://arxiv.org/abs/2603.11269), H. Yang et al. | 2026, arXiv 2603.11269 | Supervised training collapses features into the low-rank class subspace and throws away the directions that carry domain shift, so distance-based OOD scores go blind. Distilling residual features from a frozen DINOv2 teacher fixes it at no inference cost. **This is the mechanism behind L4** (sentiment is a 1-D target, the extreme case). | L4, L1 |
| [Probabilistic Skip Connections](https://arxiv.org/abs/2501.04816), F. Jimenez, M. Katzfuss | 2025, arXiv 2501.04816 | Single-pass epistemic UQ: find an intermediate layer that has not neurally collapsed and fit a density to its features, with no retraining. | L1, L4 |
| [Calibrated Decomposition of Aleatoric/Epistemic Uncertainty in Deep Features](https://arxiv.org/abs/2511.12389), D. Kumar et al. | 2025, arXiv 2511.12389 | Epistemic score = local support + spectral collapse + cross-layer inconsistency. Fed into conformal calibration for tighter intervals. | L1, L7 |
| [Split-Point Self-Consistency (SDS)](https://arxiv.org/abs/2509.13262), Z. Zhao, K. Chen | 2025, arXiv 2509.13262 | Post-hoc residual-based epistemic score for regression from split-point residual identities. Single pass. | L1 |
| [CLEAR: Calibrated Learning for Epistemic and Aleatoric Risk](https://arxiv.org/abs/2507.08150), I. Azizi et al. | 2025, arXiv 2507.08150 | Two-parameter calibration that combines any aleatoric and any epistemic estimator into conformal intervals. 28% narrower intervals on 17 datasets. | L1, L7 |

## 5. Conformal prediction for multimodal / missing-modality settings

| Paper | Year / id | Finding | L# |
|---|---|---|---|
| [Conformal Calibration for Multi-Modal Regression with Missing Modalities](https://arxiv.org/abs/2608.07795), I. Azizi | 2026, COPA 2026 (PMLR 329) | One predictor per modality produces a **cross-modal disagreement score** that scales split-conformal intervals. A **Mondrian variant stratifies by disagreement or by modality availability**. Mask-matched recalibration recovers up to 19.5 pp of coverage. | L7, L2 |
| [Conformal Fusion Under Missing Modalities (MCCF)](https://arxiv.org/abs/2608.07183), A. Moayedikia | 2026, arXiv 2608.07183 | Evidential per-modality heads with Dempster-Shafer combination (absent modality = vacuous evidence), plus **Mondrian conformal per modality-presence mask**. | L7 |
| [Mask-Conditional Conformal Prediction](https://arxiv.org/abs/2512.14221), J. Fan et al. | 2025-26, arXiv 2512.14221 | Preimpute-mask-then-correct reweighted CP. Mask-conditional validity for general missingness mechanisms, with narrower intervals than plain Mondrian. | L7 |
| [Missingness-Aware CP Under Cross-Hospital Shift](https://arxiv.org/abs/2609.30781), L. You et al. | 2026, arXiv 2609.30781 | Mondrian by missingness reduces the worst-group gap, but the gain shrinks or reverses once per-site shift is considered. **Pooled gains can hide within-site failures.** | L7 |
| [Adapting Prediction Sets to Distribution Shifts Without Labels (ECP/EACP)](https://arxiv.org/abs/2406.01416), K. Kasa et al. | 2024-25, PMLR 286 | Uses the model's own uncertainty on *unlabelled* shifted data to rescale conformal scores. Classification only. | L7, L1 |
| [Conformal Prediction Adaptive to Unknown Subpopulation Shifts](https://arxiv.org/abs/2506.05583), N.-S. Wang et al. | 2025, arXiv 2506.05583 | Recovers coverage when the group labels behind the shift are unknown and must be inferred. | L7 |
| [Conformal Shapley Intervals](https://arxiv.org/abs/2602.00171), M. Chandy et al. | 2026, arXiv 2602.00171 | Conformal intervals on per-modality Shapley importance to select modality subsets. | L5, L7 |

---

## Synthesis

### (a) Trends (2025-2026)

1. **Probabilistic JEPA has become mainstream, but every version models aleatoric ambiguity, not OOD.** VJEPA, Var-JEPA (ICML'26), S-JEPA (WACV'27), MDN-VJEPA and Flow-JEPA all make the predictor stochastic. Their uncertainty experiments cover ambiguity, branching futures or error correlation. Only Flow-JEPA tests OOD noise, and it does so for *task success*, not uncertainty quality. Var-JEPA reports no OOD evaluation at all.
2. **Doubts about ensembles and second-order epistemic UQ** (Biased Dreams, Young 2026, Jiménez 2025). In latent models, encoder and transition dynamics pull states into "well-supported" regions, so disagreement stays low exactly when it should rise. This matches FYP-1's observation that a pure-noise text input yields a smaller sigma.
3. **Distractor robustness now comes from invariance structure, not from better density modelling.** Examples: within-trajectory negatives, action common-mode cancellation, bisimulation, intervention differences, or a few labels. Making the predictor variational does not remove a *predictable* nuisance.
4. **EMA teachers are being dropped.** LeJEPA, SIGReg variants, AV-JEPA and Le MuMo JEPA all remove the EMA target, citing stability. This is relevant to L3 (the EMA target absorbs the distractor) and L6 (seed variance).
5. **"Reliability-weighted PoE" has become a standard design.** Examples: Palo et al. 2026 (PoE precision × learned reliability), PRA-PoE, UAF, GIML/UDML, EBMC and TAME on MOSEI. Reliability is learned either from labels (TIER-MoE: out-of-fold loss) or from *synthetic injected noise* (GIML, UDML). It is almost never learned from self-supervised cross-modal structure.
6. **Conflict-based discounting is the evidential-fusion answer to "confident-but-wrong"** (Bezirganyan AISTATS'25, C²MF, MCAN). It is evaluated almost only on classification; Bezirganyan's work is evidential classification, and MoNIG's regression line has seen little 2025-26 follow-up.
7. **Cross-modal discrepancy is a well-established label-free OOD signal for classification** (MultiOOD/A2D; Doorenbos 2026). Supervised fine-tuning collapses the feature directions that carry shift (TGT 2026). The usual remedy is to distil from, or keep, a frozen self-supervised teacher.
8. **Mask-conditional (Mondrian) conformal prediction is now routine** (Azizi COPA'26, MCCF, Fan et al., You et al.). Disagreement-scaled conformal scores also exist (Azizi). The frontier is shift *within* a mask (corruption, site), where pooled results can mislead.
9. **In multimodal sentiment analysis (MSA), robustness work focuses on "noisy + missing" and modality imbalance (text dominance)**, reported as accuracy, F1 and MAE (TMDC, RCE, EBMC, TAME, PRLF). Calibration, coverage and failure-detection AUROC are almost never reported on MOSI or MOSEI.

### (b) Open gaps that no paper I found closes

- **G1. Label-free epistemic precision for PoE in regression (L1+L2).**
  - Palo et al. (2026) make PoE precision reliability-weighted, but they state that the method *fails under severe degradation* and that dependable quality signals remain open.
  - GIML and UDML learn quality from *injected synthetic noise*, so they are trained on the corruption family they are tested on. That is not label-free OOD.
  - No paper derives each expert's *epistemic* precision from a **self-supervised cross-modal predictive residual**: how badly modality m's latent is predicted by the other modalities (or by its own EMA target), relative to the training residual distribution. Nor does any paper use that residual inside a Gaussian PoE for continuous regression, evaluated on pure-noise, swapped and extra-noise-channel modalities.
  - MultiOOD and Doorenbos use prediction discrepancy for *classification OOD detection*, not to *reweight fusion*, and not after fine-tuning.
- **G2. The noisy-TV fix for non-contrastive EMA JEPA, and for a distractor *shared* across modalities (L3).**
  - Gulati & Nemenman explicitly leave the non-contrastive/EMA case open. They also report a residual failure exactly when the noise correlation time is about the clip length, which is the regime of the AR(1) sticky distractor.
  - Dueling WM cancels *common-mode across actions*. No paper does common-mode cancellation *across modalities or across clips of the same video*: a target that keeps only what differs between in-video segments, or what is predictable cross-modally but not intra-modally.
  - MOSEI has natural structure for this: multiple segments per video, and the same speaker within a video.
- **G3. Keeping self-supervised uncertainty after fine-tuning (L4).**
  - TGT shows that supervised training collapses shift-sensitive directions, and fixes it for image OOD by distilling a frozen teacher.
  - Nobody has shown this for **multimodal regression with a 1-D target**, where the collapse should be worst. No paper measures whether *cross-modal agreement* survives fine-tuning (FYP-1: AUROC 0.56-0.87 frozen vs ~0.50 fine-tuned), or proposes keeping a frozen JEPA branch purely as the uncertainty and reliability channel while a fine-tuned branch does the accuracy.
- **G4. Conformal coverage under *corruption* (not missingness) without test labels, for multimodal regression (L7).**
  - Mondrian-by-mask is solved (Azizi, MCCF, Fan).
  - Corruption is not observable as a mask. Kasa et al.'s label-free adaptation is classification only, and Azizi's disagreement score comes from *supervised* unimodal predictors.
  - Still open: a conformal method whose Mondrian bins or weights come from a **label-free self-supervised reliability score**, and that keeps per-corruption coverage (e.g. Cov90 on text→noise) on MOSEI.
- **G5. Uncertainty evaluation standards in MSA (L1, L5, L7).** No MOSI or MOSEI paper I found reports coverage, PIT, CRPS, or failure-detection AUROC under swapped or pure-noise modalities. FYP-1's evaluation suite is itself a contribution.
- **G6. Seed stability of variational JEPA (L6).** Huang (2601.14354, 2610.13956) and Var-JEPA report means. No paper reports per-seed collapse rates or compares EMA-free objectives (LeJEPA/SIGReg) against EMA-based variational JEPA under distractors. Potapov et al. (2026) already question the JEPA-vs-AE gap on Huang's testbed.
- **G7. Weak-modality experts as trustworthy fallbacks (L5).** TAME, EBMC and the Dai et al. conflict-penalty work rebalance *accuracy*. None asks whether the A/V experts' *uncertainty* is honest when text is lost. FYP-1's tracking-ratio metric is not used anywhere in the literature I found.

### (c) Work that already overlaps the brief's "novel" variants (so these are NOT novel)

| FYP-1 variant | Prior work that already does it (or very close) |
|---|---|
| Variational EMA target / stochastic predictor | VJEPA (Huang 2026), Var-JEPA (Gögl & Yau, ICML'26), S-JEPA (Giakoumoglou 2026, stochastic z with posterior and prior) |
| Mixture / multimodal predictor (if considered) | MDN-VJEPA (Huang 2026) |
| Predictor ensemble (bootstrap heads) | Standard deep-ensemble disagreement (e.g. CIG 2026 uses ensemble disagreement). *Biased Dreams* and Young (2026) show its limits in latent models. Not novel, and likely to fail for the same attractor reason. |
| Robust PoE (precision down-weighted by distance to leave-one-out consensus) | Conflict-based discounting (Bezirganyan, AISTATS'25); per-instance credibility via KL (C²MF 2026); reliability-weighted PoE precision (Palo et al. 2026); gating toward neutral (Gated Symile 2026). **The LOO-consensus distance in a Gaussian PoE for regression is a small twist, not a headline contribution.** |
| Disagreement-aware PoE (variance += g·spread) | Azizi (COPA 2026) disagreement-scaled conformal intervals in multimodal regression; MoE moment matching already adds the spread term. |
| Mondrian (per-mask) conformal | Azizi 2026; MCCF (Moayedikia 2026); Fan et al. 2025-26 mask-conditional CP; You et al. 2026. Not novel. |
| Masking-ensemble test-time uncertainty | Closely related to modality-dropout or subset ensembles. MultiOOD and Doorenbos use multimodal-vs-unimodal prediction relations for OOD scores. Not novel on its own. |
| Text-anchored fusion with uncertainty routing (if considered for L5) | TAME (Huang X. et al., Oct 2026, on MOSI/MOSEI/CH-SIMS) |
| Noise-intensity / quality estimator trained on injected noise (if considered for Noisy TV A) | GIML (2026), UDML (2026) |

What still looks open and fits a Kaggle budget is the combination of three pieces:

- G1: use a self-supervised cross-modal predictive residual as the epistemic term inside PoE.
- G3: keep a frozen JEPA "uncertainty branch" alongside the fine-tuned accuracy branch.
- G4: Mondrian or weighted conformal prediction keyed on that label-free residual.

The FYP-1 corruption suite (A/B/C/D) is the right test-bed, and the jury-readable metrics are already in place: Cov90 on text→noise, AUROC for swapped modalities, and the tracking ratio. A separate, cheaper but riskier line is G2: within-video or cross-modal common-mode targets for EMA JEPA under the shared sticky distractor, measured by signal-probe R2 and distractor R2.
