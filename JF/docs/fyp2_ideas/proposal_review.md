# FYP-2 proposal review and merged proposal

Reviewer stance: senior area chair, tough but fair. I read `BRIEF.md`, `trends.md`, `inventor_crossmodal.md` and
`inventor_probabilistic.md`, the code in `JF/src/fusionlab` (`fusion.py`, `stage2.py`, `pretrain.py`,
`evaluate.py`, `runner.py`), and the FYP-1 result tables in `JF/results/run1/tables.txt`. All FYP-1 numbers below
come from that file (3 seeds, mean). Every paper cited in sections 1 and 4 was opened during this review (arXiv
abstract page, proceedings page, or PDF). Papers marked "(via trends.md)" were opened by the literature scout and
not re-opened here. Papers marked "(inventor citation, not re-verified)" are named only to score the inventors'
ideas and are not used as evidence.

---

## 0. Executive summary

- **Merged proposal: CROSS-EXAMINED FUSION (CXF).** Thesis: *a modality's weight in the product of experts should
  be earned by being corroborated by the other senses, not self-reported. CXF computes an exact, label-free
  posterior over which senses are lying, using a clean joint density of JEPA beliefs and target latents, and then
  fuses by Bayesian model averaging over trusted subsets.*
- **Novelty verdict, stated honestly.** The core principle is old. Classical fault detection and isolation (FDI)
  already does this:
  - Analytical redundancy with generalized voting (Chow 1980).
  - Banks of fault hypotheses with posterior probabilities (MMAE; Roumeliotis et al.).
  - Bernoulli outlier indicators in robust Kalman fusion (Wang et al. 2018, 2022).
  - Bayesian fusion of inconsistent sensors (Kumar et al. 2007).

  Deep learning has close cousins too: odd-one-out fusion over leave-one-out subsets (Yang et al., CVPR 2021),
  self-supervised detect-reject-correct (Lee et al. 2020), cross-modal reconstruction fault detection (Jensen et
  al. 2026), and a MOSEI reliability model trained on injected corruptions (Wu et al., *Electronics* 2026).

  What is new is narrower, but real and testable:
  - The FDI "system model" is a **learned self-supervised JEPA latent world model**, and its evidence is
    cross-modal latent prediction.
  - Inference is an **exact switch posterior fit on clean, unlabeled data only**. No corruption is ever seen.
  - Corruption is converted into **inferred missingness**. The subset heads and the per-mask (Mondrian) conformal
    quantiles already trained for missing modalities then transfer to corruption.
  - An **auditor/worker split** survives the collapse of agreement caused by fine-tuning.
  - The work is evaluated as **calibrated multimodal regression** (coverage, CRPS, tracking ratio). No MSA paper I
    found reports these under corruption.

  Frame CXF as "classical FDI, rebuilt inside a learned latent world model". Do not frame it as a new principle.
- **Secondary contributions (a coherent "senses and time" story):**
  - S1, Persistence Targets: cross-modal plus cross-time-gap JEPA targets, aimed at the shared sticky distractor.
    CXF cannot detect that distractor, and it actually *inflates* cross-modal agreement.
  - S2, the Time Auditor: temporal redundancy, the other half of classical analytical redundancy. It is the
    within-modality evidence that breaks CXF's identifiability limit when M = 3. S2 is a stretch goal.
- **Week-1 pilot.** It needs no training: it runs on the FYP-1 checkpoints. There are three go criteria: G1 is noise
  detection, G2 is swap localisation beating raw disagreement, and G3 is coverage and MAE repair on a fine-tuned
  worker (section 4.7).

---

## 1. Novelty audit

### 1.1 Core idea: infer, per clip and without labels, which modality is corrupted from cross-modal consistency, and weight the PoE by that inference

#### (a) Classical sensor fusion and FDI: how much of the core idea is already known

| Prior work (opened) | What it does | Overlap with CROSSWIRE / EAI / LPF |
|---|---|---|
| E. Y. Chow, *A Failure Detection System Design Methodology*, MIT LIDS-TH-1055, 1980 ([NTRS PDF](https://ntrs.nasa.gov/api/citations/19810003928/downloads/19810003928.pdf)) | **Analytical redundancy.** Parity relations are combinations of sensor outputs that equal zero when nothing has failed. There are two kinds: *direct* redundancy (across sensors at the same instant) and *temporal* redundancy (across histories). The **generalized voting scheme** uses a set of relations in which each component is included in at least one relation and excluded from at least one. The failed component is the one shared by every violated relation. | **This is exactly the CROSSWIRE node score** (in-edges and out-edges versus edges that don't touch k). The Time Auditor is temporal redundancy. Very high overlap in *principle*. |
| S. I. Roumeliotis, G. S. Sukhatme, G. A. Bekey, "Sensor Fault Detection and Identification in a Mobile Robot" (the authors' page labels it IROS'98) ([page](https://www-users.cse.umn.edu/~stergios/papers/abstracts/abstract_iros98.html)) | **Multiple-model adaptive estimation (MMAE).** A bank of Kalman filters, one per failure hypothesis plus a nominal filter, whose residuals give a probability for each hypothesis. | **This is LPF's and EAI's enumeration of 2^M configurations with per-configuration evidence.** High overlap in principle. |
| H. Wang, H. Li, J. Fang, H. Wang, "Robust Gaussian Kalman Filter With Outlier Detection", IEEE SPL 25(8), 2018 ([PDF](https://personal.stevens.edu/~hli/papers/WangLiFang18.pdf)) | A Bernoulli indicator per measurement (Beta prior), inferred by mean-field VB. Its posterior depends on the residual between the measurement and the state. An outlier's noise covariance is scaled to R/z̄, so the outlier is ignored. | **This is EAI's "lying switch" and the "earned-trust" precision multiplier** (precision × posterior that the measurement is nominal). |
| H. Wang et al., "Outlier-Detection Based Robust Information Fusion for Networked Systems", arXiv 1909.10006 (DOI 10.1109/JSEN.2022.3212908) | One binary indicator per sensor in multi-sensor fusion, residual-based, inferred by VB. Centralized and decentralized variants. | Per-*sensor* switches in multi-sensor fusion. Same structure as per-*modality* switches. |
| M. Kumar, D. P. Garg, R. A. Zachery, "A method for judicious fusion of inconsistent multiple sensor data", IEEE Sensors J. 7(5), 2007 ([Duke record](https://scholars.duke.edu/publication/971371)) | A modified Bayesian fusion with a term for "probability that a measurement is valid given the data and the true state". When sensors disagree, the posterior variance widens. Spurious sensors are removed automatically. | The "inconsistency widens variance" idea behind LPF's BMA variance and FYP-1's `dapoe`. |
| H. Zhang et al., "Integrated navigation system based on fault detection using double state Chi-square test", *Acta Aeronautica et Astronautica Sinica* 41(S2), 2020 ([abstract](https://hkxb.buaa.edu.cn/EN/abstract/article/1000-6893/17982)) | Residual and state χ² tests for isolating a faulty subsystem in federated filtering. | The χ² gate: "drop expert m if its standardized residual exceeds a χ² quantile". This is a **mandatory baseline**. |

**What classical FDI gives for free, and what it does not.** The *principle* is about 40 years old: localize a fault
by which consistency relations fail, then infer a posterior over fault hypotheses and fuse under it. Any jury
member from control or signal processing will recognize it, so the proposal must cite it up front.

Classical FDI assumes a *known physical model* (state-space dynamics, measurement matrices, noise covariances), and
its sensors measure the *same* physical state. In multimodal affect data:

1. No model links BERT, COVAREP and FACET features. It has to be *learned without labels*, and the JEPA latent
   predictors are exactly that learned model. The "parity relation" becomes the cross-modal JEPA prediction
   residual.
2. The modalities are *not* redundant measurements of one state. They share only a weak, nonlinear slice of
   information: AV-only MAE is 0.81 versus 0.56 for TAV. Whether corroboration carries enough signal is therefore an
   *empirical* question with no classical answer. The pilot tests it.
3. Fine-tuning destroys the learned parity relations: disagreement AUROC is about 0.50 after fine-tuning. FDI has
   no analogue for this.
4. The target is calibrated *predictive* uncertainty for a downstream regression (coverage, CRPS). FDI targets
   state-estimation accuracy.

#### (b) Machine learning prior work

| Prior work (opened) | Mechanism | Label-free? | Localizes? | Reweights fusion? | Regression and calibration? |
|---|---|---|---|---|---|
| K. Yang, W.-Y. Lin, M. Barman, F. Condessa, Z. Kolter, "Defending Multimodal Fusion Models against Single-Source Adversaries", CVPR 2021 ([arXiv 2206.12714](https://ar5iv.labs.arxiv.org/html/2206.12714)) | An **odd-one-out network** outputs P(modality i perturbed) and P(none). The fusion is a **weighted sum of k+1 leave-one-out fusions, weighted by those probabilities.** | **No.** It is trained with cross-entropy on *synthetically perturbed* inputs. | Yes | Yes (a mixture over subsets) | No (classification and detection, adversarial setting) |
| M. A. Lee, M. Tan, Y. Zhu, J. Bohg, "Detect, Reject, Correct: Crossmodal Compensation of Corrupted Sensors", arXiv 2012.00201 | Self-supervised. Detects a corrupted sensor from *unimodal reconstruction* loss, rejects it, and compensates from the other modalities. | Yes (self-supervised) | Yes | Yes (hard reject) | No (robotics) |
| M. M. Jensen, D. Hammershøi, R. Wiśniewski, O. Fink, "Robust Fault Detection in Mechanical Multimodal Time Series via Self-Supervised Cross-Modal Reconstruction", arXiv 2609.16314 (Sep 2026) | Reconstructs each modality from the others. The **cross-modal reconstruction error** is the anomaly score, with an adaptive test-time threshold. | Yes | Not stated in the abstract | Not stated | No (detection) |
| S. Kumar et al., PRIME, "Adaptive Modality Reliability Diagnosis and Restoration…", arXiv 2608.03475 (Aug 2026) | A per-modality weakness score (a log-variance) built from confidence, disagreement, cross-modal consensus and degeneracy. It is **trained on controlled corruptions with known severity**, then restores cross-modally and fuses by inverse variance. | No (synthetic severity supervision) | Yes (per modality) | Yes | Partly (heteroscedastic) |
| Y. Wu, X. Zhu, H. Liu, "Reliability-Aware Multi-Modal Sentiment Analysis Under Missing and Corrupted Modalities", *Electronics* 15(16):3624, 2026 ([MDPI](https://www.mdpi.com/2079-9292/15/16/3624)) | **On MOSI and MOSEI.** An evidential head plus a quality head regressed to "1 − injected severity". JS disagreement down-weights modalities. Reports ECE, NLL, Brier, selective risk, and **clean-vs-corrupted AUROC**, with some unseen test corruptions. | No (injected corruptions) | Yes (quality per stream) | Yes | Partly (classification-style ECE) |
| C. Meng et al., MRCF, arXiv 2608.03611 (Aug 2026) | MOSI, MOSEI and CH-SIMS. Reliability from intra-modal quality plus **cross-modal semantic consistency** (cosine of learned projections). Trained on a model-derived proxy from complete-vs-incomplete prediction agreement. | Partly (no injected noise, but trained through the task) | Per modality | Yes | No calibration or coverage; erasure only, no noise corruption |
| P. Palo et al. (Clifton lab), "Reliability-Aware Probabilistic Fusion…", arXiv 2610.29621 (Oct 2026) | A Gaussian PoE whose precisions are tempered by an input-dependent reliability weight. **Fails under severe degradation.** The authors call dependable quality signals an open problem. | Not stated | Implicit | Yes | Partly |
| G. Bezirganyan et al., "Discounted Belief Fusion", AISTATS 2025 ([PMLR](https://proceedings.mlr.press/v258/bezirganyan25a.html), [arXiv 2412.18024](https://arxiv.org/html/2412.18024v2)) | A pairwise **conflict matrix**, turned into an agreement matrix. Modality v's discount is the product of its row. Fixed (not learned), on Dirichlet opinions. | Yes (the discount rule itself) | Soft | Yes | No (classification, Dirichlet) |
| H. Ma et al., MoNIG, "Trustworthy Multimodal Regression with Mixture of Normal-inverse Gamma Distributions", NeurIPS 2021 ([arXiv 2111.08456](https://arxiv.org/abs/2111.08456v1)) | Evidential NIG per modality, fused by NIG summation. Claims robustness to corrupted modalities through uncertainty. Includes MSA. | Uses labels | No explicit localization | Yes | Yes (regression). Already in FYP-1. |
| Y. Cao, D. J. Fleet, gPoE, arXiv 1410.7827 | Expert weight = entropy change from prior to posterior. No outlier step. | Yes | No | Yes | Gaussian |
| H. Dong, Y. Zhao, E. Chatzi, O. Fink, MultiOOD/A2D, arXiv 2405.17419 | **Modality prediction discrepancy** is much larger on OOD data, and A2D enlarges it during training. | Uses class labels | No (clip-level OOD) | No | No (classification OOD) |
| I. Azizi, "Conformal Calibration for Multi-Modal Regression with Missing Modalities", arXiv 2608.07795 | A disagreement score from per-modality (supervised) predictors. Intervals are disagreement-scaled, or Mondrian by disagreement or availability. | Predictors are supervised | No | No (intervals only) | **Yes (regression, conformal)**. Missingness, not corruption. |
| P. Tenali et al., C²MF, arXiv 2603.26629 | Per-instance credibility via a KL measure computed exactly in a conditional probabilistic circuit. A "Conflict" benchmark. | Not stated | Yes | Yes | Not regression (likely classification) |
| J. Li et al., AdaPGC, arXiv 2611.03232 (Nov 2026) | Test-time adaptation for corrupted ("reliability-shifted") modalities using class-conditional Gaussians, with unlabeled test data. | Yes (test-time) | Partly | Yes | Classification |
| S. Eduardo, A. Nazábal, C. K. I. Williams, C. Sutton, "Robust VAEs for Outlier Detection and Repair of Mixed-Type Data", AISTATS 2020 ([PMLR 108](https://proceedings.mlr.press/v108/eduardo20a.html)) | Unsupervised. **Learns the probability that each cell is an outlier** inside a VAE. | Yes | Per cell | Repairs | No |

I also searched for "JEPA prediction error used to localize a corrupted modality and reweight fusion". The hits were
single-modality JEPA anomaly detectors: MADJEPA (arXiv 2611.25194, medical images) and SC-JEPA (arXiv 2602.04643,
time-series anomaly prediction). Neither localizes a modality or touches fusion.

#### (c) Verdict for the core idea

| Component | Status |
|---|---|
| "Detect and localize the inconsistent sensor from cross-sensor consistency, then exclude or down-weight it" | **Known** (Chow 1980; MMAE; Wang 2018/2022; Kumar 2007). In deep learning, **known with supervision** (Yang CVPR'21, PRIME, Wu 2026) and **known self-supervised for detection** (Lee 2020, Jensen 2026). |
| "Weighted mixture over leave-one-out or subset fusions, weighted by P(modality corrupted)" | **Known** (Yang CVPR'21, supervised odd-one-out; MMAE for filters). LPF's BMA over subset-PoEs is the probabilistic, label-free version of Yang's robust fusion layer. Cite it prominently. |
| Evidence = **JEPA latent predictive residuals** inside a self-supervised world model (beliefs plus own-target latents) | **No prior found.** |
| **Exact** posterior over 2^M configurations from a **clean-only, unlabeled joint density** (no corruption seen) | **No prior found in deep multimodal fusion.** It is standard in classical MMAE with known models. |
| Corruption → inferred missingness → reuse of the **subset heads and per-mask conformal quantiles** already trained for missing modalities | **No prior found.** Azizi does Mondrian by availability or disagreement, not by inferred corruption. |
| Auditor/worker split: a frozen self-supervised auditor survives fine-tuning collapse | Related to teacher distillation for OOD (TGT, arXiv 2603.11269, via trends.md). New for multimodal reliability. |
| Calibrated regression under corruption (Cov90, CRPS, tracking ratio) on MOSEI | Wu et al. 2026 report ECE and clean-vs-corrupted AUROC on MOSEI, but use *injected-corruption supervision* and no coverage or tracking. **The evaluation is a genuine contribution.** |

**Bottom line.** The originality score is about 3/5. The contribution is defensible if it is pitched as *"label-free
FDI in a learned JEPA latent world model, and the first corruption-calibrated multimodal regression on MOSEI"*, and
if the classical lineage is cited in the first paragraph. Three papers must be cited and compared against:
- Yang et al., CVPR 2021 (the closest deep method; a supervised odd-one-out mixture over leave-one-out fusions).
- Wu et al., 2026 (the closest in domain: MOSEI, corruption, reliability, calibration metrics).
- Chow 1980 / MMAE (the closest in principle).

### 1.2 Secondary idea A: Persistence Targets (cross-modal plus time-gap JEPA targets)

| Prior work (opened) | Relation |
|---|---|
| C. Feichtenhofer, H. Fan, B. Xiong, R. Girshick, K. He, "A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning", CVPR 2021 ([arXiv 2104.14558](https://arxiv.org/abs/2104.14558)) | The objective rewards features that are **persistent across time within a video**, and persistence helps even with 60-second gaps. **The "persistence across a gap" principle is known** (single modality, contrastive or BYOL style). |
| P. Sermanet et al., Time-Contrastive Networks, arXiv 1704.06888 | Positives are simultaneous **multiple views**; negatives are temporal neighbours. The learned representation ignores nuisance such as occlusion, blur, lighting and background. This is the cross-view half of the principle, without a time gap. |
| P. Gulati, I. Nemenman, "Contrast encodes inductive bias", arXiv 2606.07770 | Slow, trajectory-constant noise hijacks predictive and contrastive objectives. The fix is within-trajectory negatives. Its goal is the opposite (keep the dynamics, drop clip-constant noise). |
| S. Jenni, A. Black, J. Collomosse, "Audio-Visual Contrastive Learning with Temporal Self-Supervision", arXiv 2302.07702 (AAAI 2023) | Cross-modal contrast between *temporally corresponding* audio and video, plus temporal pretext tasks. No gap is designed against nuisance. |
| Huang, VJEPA, arXiv 2601.14354 (via trends.md) | The noisy-TV claim being tested. Uses a same-stream next-step target. |

**Verdict: about 3/5.** Persistence (Feichtenhofer), cross-view invariance (TCN) and the hijacking diagnosis (Gulati)
are all known. Three things are new:
1. The *conjunction* of cross-modal and cross-time-gap constraints on the *EMA target of a JEPA*.
2. Choosing the gap from the nuisance's correlation time.
3. An explicit identifiability limit: "persistent ∧ shared ∧ predictable nuisance cannot be told apart from content
   without labels".

These are evaluated on the field's own noisy-TV benchmark (Huang's code) and on a multimodal shared distractor. A
clean, falsifiable contribution of moderate size.

### 1.3 Secondary idea B: Time Auditor (temporal JEPA surprise as within-modality evidence)

This is Chow's *temporal redundancy* (above) implemented as masked-frame latent prediction. Reconstruction-error OOD
scores and single-modality JEPA anomaly detectors (SC-JEPA, MADJEPA, and JEPA prediction error as a complexity
score, arXiv 2606.28383, via trends.md) are close. **Verdict: about 2/5 as a standalone idea.** Its value here is
*structural*: it supplies the within-modality block that resolves CXF's two-of-three ambiguity, and it gives the
"two orthogonal auditors" figure (noise vs swap).

### 1.4 Other secondary ideas (brief)

- **TGT (CCA nulling).** I found no JEPA work that uses the canonical-correlation spectrum to null a nuisance. But
  the FYP-1 distractor is *linear by construction* (`d @ DIST_MIX[m]`), so a reviewer will say the method was
  designed for the benchmark. Novelty is about 3, credibility about 2.
- **BG-PoE.** Bates-Granger (1969) forecast combination and CoDE (arXiv 2505.01134, "consensus of dependent
  experts", opened: it learns subset contributions) already cover it. An extension, not a contribution.
- **B2-JEPA.** GECO-style constrained objectives (inventor citation, not re-verified) applied to VJEPA seed
  stability. A useful diagnostic, small novelty.
- **Keep-alive and PaP-FT.** Auxiliary SSL and KL-to-pretrained during fine-tuning are standard (the inventors say
  so themselves). Use them as ablations only.

---

## 2. The merged proposal: CROSS-EXAMINED FUSION (CXF)

**One-sentence thesis.** *Trust is earned by corroboration: CXF replaces each expert's self-reported precision with
an exact, label-free posterior over which senses are lying, computed from how well each sense's JEPA belief and
observed latent agree with the other senses under a clean joint density. It then fuses by Bayesian model averaging
over trusted subsets, so corruption is handled as inferred missingness.*

The design takes:
- the exact enumeration and the BMA over subset-PoEs from **LPF**;
- the own-modality observed latent as a top-down check (e_m) from **EAI**;
- the cross-modal edges between M x M targets as the trained evidence of a later phase, from **CROSSWIRE**;
- the frozen auditor from the **auditor/worker split**.

### 2.1 Model

Notation:
- Modalities m = 1..M (TAV gives M = 3; adding the noise channel N gives M = 4).
- **Auditor** = frozen stage-1 experts and EMA target(s). It never sees labels.
- **Worker** = the stage-2 model being protected (frozen or fine-tuned experts, fusion, and `Head`).

**Evidence per modality.** For clip x:

```
mu_m^A(x), s_m^A(x)    auditor expert m's belief about the text-target latent (existing V-JEPA experts)
e_m(x) = T_m(x_m)       modality m's own target latent
                          Phase A: only e_t (existing text EMA target)
                          Phase B: per-modality EMA targets, as in CROSSWIRE
o_m(x) = [ P_m mu_m^A(x) ;  Q_m e_m(x) ]  in R^{k_m}
          P_m, Q_m: PCA projections (k = 8 each), fit on clean unlabeled train latents
o = [o_1, ..., o_M]     roughly 24-48 dimensions
```

**Clean joint density.** Fit on clean, *unlabeled* validation half A with Ledoit-Wolf shrinkage:
`o ~ N(m, C)`.

**Switches.** Each modality has a switch c_m in {0, 1} (1 = "lying": corrupted, swapped or alien), with prior
`P(c_m = 1) = pi` (default 0.05). Write R = {m : c_m = 0} for the reliable set. Then:

```
p(o | c) = N(o_R ; m_R, C_RR)  *  prod_{m not in R}  N(o_m ; m_m, kappa * C_mm)        (kappa = 10)
```

Under the outlier model a lying block is *decoupled* from the others and broadened.

**Exact posterior with a temperature T.** T absorbs misspecification: the joint is not truly Gaussian, and the PCA
blocks are correlated.

```
log p(c | o) = (1/T) * log p(o | c) + sum_m [ c_m log pi + (1 - c_m) log(1 - pi) ] - log Z      (2^M terms)
```

**Why this *is* cross-examination.** Hold the other switches at "reliable". The log Bayes factor for "m is lying"
is:

```
log BF_m = (1/T) [ log N(o_m ; m_{m|-m}, C_{m|-m})  -  log N(o_m ; m_m, kappa C_mm) ]^-
         = (1/2T) [ d2_cond(m)  -  d2_marg(m)/kappa ]  +  const(log-dets)
```

Here `d2_cond(m)` is the Mahalanobis distance of block m from its *linear cross-modal prediction* by the other
blocks. These are the CROSSWIRE in-edges and out-edges, and the EAI top-down decoders A_m, both calibrated by C.
`d2_marg(m)` is its marginal (single-modality) surprise. So:

- A swapped modality is ordinary marginally but not conditionally, so d2_cond is large.
- A pure-noise modality is large on both.
- An expert that is honestly unsure is penalized less once the heteroscedastic option below is used.

These are Chow's parity relations and generalized voting with learned, calibrated relations. Chow's "each component
excluded from at least one relation" is automatic here: every subset R is scored.

**Optional heteroscedastic block (ablation H-on).** Use C_mm(x) = C_mm + P_m diag(s_m^A(x)^2 - mean s^2) P_m^T,
clipped to be PSD. A confident expert is then *more* surprising when it is inconsistent (the LPF argument). The
default is off, because FYP-1 shows that sigma is uninformative under corruption.

**Predictive distribution: BMA over subset-PoEs of the worker.**

```
For each R (intersected with the available modalities):
    (mu_R, var_R) = PoE_R(worker beliefs)              # poe_core with av * 1_R
    (m_R, v_R)    = Head(mu_R, var_R)                  # existing head; trained on random subsets, so every PoE_R is in-distribution
p(y | x)   = sum_R  p(R | o)  N(y ; m_R, v_R)
mean       = sum_R p_R m_R
var        = sum_R p_R (v_R + m_R^2)  -  mean^2                         # includes the between-subset spread
share_m    = sum_R p_R  share_m(R)                                      # reported fusion weight
r_m        = P(c_m = 1 | o)                                             # per-modality, label-free corruption probability
```

R = ∅ falls back to the prior predictive `Head(prior)`.

**Conformal variant CXF-C** (corruption becomes inferred missingness). FYP-1 already computes per-mask conformal
quantiles `qn_mask[R]` on validation (decoder D3m). The CXF-C interval is the posterior mixture of per-subset
intervals:

```
[ m_R - q_R sd_R , m_R + q_R sd_R ]   mixed by p(R | o)
```

Implementation: take the 90% quantile of the mixture by bisection, or use the MAP R̂ when p(R̂ | o) > 0.9.

If the switch posterior is right, a corrupted clip is calibrated *as if* that modality were missing. Missing
modalities are the condition FYP-1 can already calibrate. No corrupted calibration data is used.

### 2.2 Exact posterior vs a learned residual-to-precision map: the decision

**Decision: the exact posterior is the method. The learned map (CROSSWIRE "earned trust",
`pi_m = sigmoid(tau - kappa s_m)`) is an ablation and, in its self-play form, a baseline.**

1. **Label-free and corruption-agnostic by construction.** The exact posterior needs only a clean joint density.
   A learned map needs either labels or injected corruptions. With injected corruptions it joins the crowded family
   of Yang (CVPR'21), PRIME, Wu (2026), GIML and UDML, and is trained on the very corruption family it is tested
   on. The project's distinctive claim ("no corruption ever seen") survives only with the exact posterior.
2. **Semantics you can test.** r_m is a probability. Its clean false-alarm rate, its detection AUROC and its
   localization accuracy (argmax_m r_m vs the truly corrupted m) are all measurable. A sigmoid score has no such
   meaning: Palo et al. themselves call their weight "not a calibrated quality probability" (as quoted by the
   probabilistic inventor).
3. **Coverage comes for free.** BMA variance includes the spread between subsets. Mixing at the subset level also
   enables CXF-C, whereas a precision multiplier produces a single PoE and loses both.
4. **Cost is trivial.** 2^M = 8 (16 with N). If the heteroscedastic option is off, C is fixed, so every marginal
   Cholesky factor is *precomputed once*. Inference is 8 Mahalanobis distances on vectors of at most 48 dimensions
   plus 8 cheap PoEs, against about 1 ms of GPU time per batch.
5. **Honest weakness.** The Gaussian-joint likelihood is misspecified: there are heavy tails and the PCA
   summarization loses information. Mitigations: the temperature T; a Student-t ablation (replace N with a
   multivariate t with ν = 5 in both hypotheses); and the learned map as an *upper-bound* comparison (if self-play
   beats CXF by more than 0.05 AUROC on *unseen* corruption families, report that honestly).

### 2.3 Training procedure

- **Phase A (no new training).**
  - Stage 1 = existing V-JEPA, or align, checkpoints (`v2_experts_{group}_{obj}_s{seed}.pt`).
  - Stage 2 = existing stage-2 checkpoints. They were trained with `random_subsets` (`stage2.py`, line 73), so every
    subset head exists.
  - Fit CXF on the clean validation split:
    1. PCA (P_m, Q_m) on train latents.
    2. (m, C) with Ledoit-Wolf on valid-A (the first half of the 1,871 validation clips).
    3. Fix pi = 0.05 and kappa = 10 *a priori*.
    4. Choose T on valid-B so that the **clean false-alarm rate** `mean_m 1[r_m > 0.5]` equals 1%.

  No labels and no corrupted data are used anywhere in the fit. The sensitivity grid pi ∈ {0.01, 0.05, 0.2},
  kappa ∈ {3, 10, 30} is reported, not tuned.
- **Phase B ("CROSSWIRE witnesses", trained evidence).** Add per-modality EMA targets T_a and T_v and M x M Gaussian
  predictor heads g_{i→j}. The stage-1 loss is L_xwire from `inventor_crossmodal.md` §1, with VICReg on every
  backbone. The text column stays the fusion space, so the workers are unchanged. o_m then includes Q_m e_m for
  every m, plus the out-predictions P_j g_{m→j}(h_m), so the a↔v relations become learnable and nonlinear. The CXF
  fit is the same as in Phase A.
- **S1 (Persistence witnesses).** Train the stage-1 auditor with Persistence Targets (section 3). The CXF fit is the
  same.

### 2.4 Inference procedure (per batch)

1. Run the auditor on the (possibly corrupted) inputs and compute o (one extra forward pass when the worker is
   fine-tuned; zero when frozen).
2. Compute the 2^M log-joints, add the prior, normalize with temperature T, giving p(R | o) and r_m.
3. Run the worker beliefs, then for each R compute PoE_R → Head → (m_R, v_R), and mix (section 2.1).
4. Return (mean, var, shares, r_m). Optionally return CXF-C intervals.

### 2.5 How it plugs into `JF/src/fusionlab`

| File | Change |
|---|---|
| **new** `cxf.py` | <ul><li>`fit_witness(auditor, target, mods, split='valid')`: one pass, PCA, Ledoit-Wolf, valid-A/B split, T search. Returns a `Witness` `nn.Module` with buffers (m, C, the precomputed Cholesky factor for each R, P, Q, T, pi, kappa).</li><li>`Witness.evidence(b, masks, xs)`: calls `models.expert_beliefs(auditor, b, masks, xs)` and `target(xs.get('t', b['t']), mask)` on the *same corrupted inputs*.</li><li>`Witness.posterior(o) -> logp_R (B, 2^M), r (B, M)`.</li><li>`cxf_output(fusion, head, mus, lvs, av, logp_R)`: loops over `batching.all_subsets(M)`, calls `fusion.poe_core(mus, lvs, av * R)` and `head`, then mixes. It also returns a moment-matched latent (fm, fv) so that `D0`, the probes and `latent_maha` keep working.</li></ul> |
| `stage2.py` | `fused_output(..., witness_logp=None)`: if given, dispatch to `cxf_output`. Stage-2 *training* is unchanged (CXF is inference-only). |
| `evaluate.py` | In `predict`, after `mus, lvs = expert_beliefs(...)`, compute `o = model['witness'].evidence(b, masks, xs)` when present and pass `logp_R`. Log `r_m` next to `esd`, `dis` and `pe`. Add `det_AUROC_cxf` and the localization accuracy where `det_AUROC_sd`/`_dis` are computed (around line 273). Add decoder `D3x` (CXF-C) next to `D3m`. |
| `runner.py` | Load the auditor with `pretrain_or_load` (frozen copy) next to the worker. Call `fit_witness` once per (objective, seed). Cache it to `ckpt/` as `cxf_{group}_{obj}_s{seed}.pt`. |
| `fusion.py` | Add the baseline kinds `chi2gate`, `vbind` (VB Bernoulli indicator vs the fused estimate, Wang 2018 style), `gpoe` (entropy-change weights), `conflict` (Bezirganyan-style agreement product on Gaussian SKL), and `relpoe` (Palo-style learned sigmoid reliability). Each is under 30 lines and reuses `poe_core`. |
| `pretrain.py` (Phase B, S1) | Objectives `xwire` and `persist`. The `Target` dict per modality, the EMA loop over m, and the window masks in `batching.py` follow the crossmodal inventor's sketch. |

---

## 3. Secondary contributions: one story, "corroboration across senses and time"

CXF trusts whatever the senses *agree on*. That gives two natural follow-up questions, and each secondary
contribution answers one.

### S1 (required): Persistence Targets, so that what is corroborated is content and not shared nuisance

- **Why it belongs.** Under the shared sticky distractor (D-shared), every modality carries the same predictable
  nuisance. Cross-modal agreement is then *inflated by the nuisance*. CXF sees "consistent senses", and the V-JEPA
  target absorbs the distractor (FYP-1, σ = 8: signal R² 0.068, distractor R² 0.517). CXF cannot fix this, because
  the shared distractor is not a lying sense. The *target* has to change.
- **Method.** As in `inventor_crossmodal.md` §4:
  - Split each clip into windows W1 and W2 (40% each) with a gap of about 20%, matched to the AR(1) correlation time
    1/(1 − ρ) = 10 frames.
  - The JEPA target for a prediction from modality i, window W, is T_j(x_j restricted to W'), with j ≠ i
    (P-full), or j = i allowed only across the gap (P-time).
  - The text column remains the fusion space.
- **The same machinery serves both contributions.** The M x M heads of Phase B become cross-window heads. S1 is
  therefore "CROSSWIRE with a time gap", and the auditor trained this way becomes the "Persistence witness" for CXF.
- **Stated limit.** Nuisance that is persistent (ρ → 1), shared and predictable cannot be told apart from content
  without labels. Speaker identity will survive.

### S2 (stretch): Time Auditor, temporal redundancy as within-modality evidence

- **Why it belongs.** With M = 3, CXF prefers "one liar" over "two liars", so swap[AV] (two corrupted) can be
  mis-explained as "text is lying". A within-modality block breaks the tie, because it doesn't depend on the other
  senses. That block is masked-frame latent surprise S_m^time, standardized on clean data.
- **How it enters CXF.** Append S_m^time to o_m. The marginal-vs-conditional decomposition then automatically uses
  it as a self-edge.
- **The figure it produces.** A 2 x 3 table {time, senses, both} x {noise, swap, noise channel}.
- **Why it is only a stretch goal.** It needs a frame-level target and a padding-aware loss, and the noise-detection
  part may already be easy without it (see G1).

**Explicitly dropped from the headline:**
- EYP (overlaps S2, and is weaker on BERT-PCA tokens).
- TGT (benchmark-specific linearity).
- BG-PoE (CoDE overlap).
- B2-JEPA (a separate story, which could be an appendix on Huang's toy if time remains).
- Keep-alive and PaP-FT (ablations: "frozen auditor vs keep-alive auditor").

---

## 4. Experimental plan for FYP-2

### 4.1 Hypotheses and success criteria

All values are from FYP-1 `tables.txt` (3 seeds). Targets are for 5 seeds, and a criterion counts as met when the
5-seed mean reaches the target with the 95% bootstrap CI excluding the FYP-1 value. "FT" = fine-tuned PoE worker
(V-JEPA objective) with a frozen V-JEPA auditor; "FR" = frozen V-JEPA PoE.

| # | Hypothesis | Metric | FYP-1 value | Target |
|---|---|---|---|---|
| H1 | CXF detects alien input that sigma cannot see | AUROC of r_t, text pure noise (λ = 1), clean vs corrupt | sigma about 0.50 (brief); sd_t *falls* 0.068 → 0.061 (FR) | **≥ 0.90**. Also ≥ 0.75 at λ = 0.5. |
| H2 | CXF localizes swaps *better than raw disagreement* (the real test of cross-examination) | AUROC of r_a (swap A) and r_v (swap V); localization accuracy argmax r on swapped clips | disagreement AUROC: FR V-JEPA 0.565 / 0.562; FR align 0.841 / 0.827; FT about 0.50 | **≥ 0.70 (V-JEPA auditor), ≥ 0.85 (align auditor)**, and **≥ raw-disagreement AUROC + 0.03 for the same auditor**. Localization ≥ 0.6 (chance 0.33). |
| H3 | Corruption repair: accuracy | MAE, text noise λ = 1 | FR 0.848, FT 0.870; AV-only floor 0.817 / 0.809 | **FR ≤ 0.83, FT ≤ 0.84** |
| H4 | Corruption repair: calibration without corrupted calibration data | Cov90 (D3a), text noise λ = 1; Cov90 for the "pure noise" condition kind | FR 0.799, FT 0.746; pure-noise kind: FT D3a 0.817, D2 0.789 | **≥ 0.86 (raw CXF), ≥ 0.88 (CXF-C)** |
| H5 | Fusion weight and uncertainty track corruption | text share at λ = 1; tracking ratio (text pure noise) | share 0.892 (FR); tracking 0.855 (FR), 0.627 (FT) | **share ≤ 0.25; tracking 0.95–1.10 (FR), ≥ 0.90 (FT)** |
| H6 | Noise channel (B) gets no weight, label-free | precision share of N | V-JEPA 0.199, align 0.128, recon 0.381 (supervised PoE learns 0.024) | **≤ 0.05** with Cov90 unchanged (±0.01) |
| H7 | No harm on clean data | TAV MAE / NLL / Cov90; clean false-alarm rate | FT 0.563 / 1.261 / 0.883 (D2) | MAE ≤ +0.005, NLL ≤ +0.01, Cov90 ±0.01, FAR ≤ 2% |
| H8 (S1) | Persistence Targets resist the shared distractor | probe signal R² / distractor R², D-shared σ = 8, V-JEPA | 0.068 / 0.517 (indep: 0.071 / 0.181) | **signal ≥ 0.20, distractor ≤ 0.15 (indep ≤ 0.08)**; clean frozen MAE within +0.02 of 0.727 |
| H9 (S1) | The result transfers to Huang's own noisy-TV toy | distractor R², gapped target vs VJEPA, 15 seeds | VJEPA reference: Huang code, FYP-1 reproduction | gapped target better in **≥ 12/15 seeds** (one-sided sign test p ≈ 0.018) |
| H10 (story link) | Persistence witnesses make CXF robust under D-shared | swap-A/V AUROC of r under D-shared σ = 8 | to be measured in the pilot / Phase A | Persistence auditor ≥ V-JEPA auditor + 0.05 |
| H11 (S2, stretch) | Temporal and cross-modal evidence are complementary | AUROC table {time, senses, both} x {noise, swap, B} | n/a | time ≥ 0.9 on noise and about 0.5 on swap (pre-registered); "both" ≥ max of the two on every column |
| H12 (2nd dataset) | Direction replicates on CMU-MOSI | H1, H3, H4, H7 metrics | not measured | Same direction, with H1 ≥ 0.85 and H4 ≥ 0.85 (MOSI is small: 5 seeds, wide CIs reported) |

### 4.2 Baselines (all on the same workers, subsets and corruption suite)

1. **Existing FYP-1 fusions:** PoE, `poe_pc`, `mpoe`, `rpoe` (LOO consensus), `dapoe`, attention, MoE, concat,
   late average, and **MoNIG** (Ma et al., NeurIPS 2021).
2. **Classical FDI:**
   - (a) **χ² gate**: drop m if its LOO-standardized residual exceeds the 99% χ² quantile fit on clean validation.
   - (b) **VB Bernoulli-indicator fusion** in the style of Wang et al. 2018 / 2022: a mean-field indicator on the
     residual between each expert and the fused estimate, 3 iterations.
   - (c) **MMAE-style hypothesis bank** with identity covariance. This is CXF without the learned joint covariance,
     which isolates the value of *learning* the parity relations.
3. **Deep reliability methods:**
   - (a) **gPoE** (Cao & Fleet; entropy-change weights).
   - (b) **Conflict discounting** (Bezirganyan et al., AISTATS 2025), adapted to Gaussian SKL.
   - (c) **Reliability-weighted PoE** (Palo et al. 2026 style): a learned sigmoid r_m(x) trained by the task loss.
   - (d) **Odd-one-out robust fusion** (Yang et al., CVPR 2021): supervised on injected swaps and noise, *trained on
     the swap and audio/video noise families and tested on text noise and B*, which are unseen.
   - (e) **Quality-head reliability** (Wu et al. 2026 / PRIME style): a quality score regressed to the injected
     severity.
   - (f) **CROSSWIRE learned map with self-play** (the ablation in section 2.2).
4. **Conformal:**
   - Split conformal (D3b).
   - **Mondrian per mask** (D3m; Azizi 2026 / MCCF).
   - **Disagreement-scaled conformal** and **Mondrian by disagreement** (Azizi 2026).
   - CXF-C must beat all of these on *worst-condition* Cov90 without using corrupted calibration data.
5. **S1 baselines:** V-JEPA, deterministic JEPA, recon, align, supervised, and **within-clip-negatives InfoNCE**
   (the Gulati & Nemenman fix, adapted).

Baselines 3(d) and 3(e) see corruption during training. They are reported in a separate block labeled
"corruption-supervised (privileged)".

### 4.3 Datasets (availability checked)

- **CMU-MOSEI** (primary): the MMSA `aligned_50.pkl` already in use.
- **CMU-MOSI** (secondary, zero engineering): `MOSI/Processed/aligned_50.pkl` is listed in the official MMSA repo
  ([github.com/thuiar/MMSA](https://github.com/thuiar/MMSA), Google Drive / BaiduYun). It has the same feature
  families as MOSEI, so `data.py` only needs a path change. Its 2,199 clips mean wide CIs.
- **CH-SIMS v2(s)** (stretch, different language and features, *with unimodal labels*):
  - Official downloads (Google Drive / Baidu) are linked in the [ch-sims-v2 README](https://raw.githubusercontent.com/thuiar/ch-sims-v2/main/README.md).
  - An unofficial Hugging Face mirror holds a single `ch-simsv2s.pkl` (3.62 GB) ([AZYoung/SIMSV2_processed](https://huggingface.co/datasets/AZYoung/SIMSV2_processed/tree/main)).
  - The paper (Liu et al., ICMI 2022, [arXiv 2209.02604](https://arxiv.org/abs/2209.02604v1)) states that the added
    segments carry **unimodal and multimodal annotations**, with regression labels in [−1, 1] (dataset page).
  - **Unique value:** a *natural* test of CXF. Correlate r_m with the |unimodal − multimodal| label disagreement (an
    exploratory hypothesis, with no target fixed in advance).
  - Cost: the MMSA SIMS file is *unaligned* (`unaligned_39`), so `batching.py` needs per-modality lengths
    (`RT.LENS` already exists).
- MELD and IEMOCAP: not verified in this review, and not needed.

### 4.4 Ablations

1. **Evidence blocks:** beliefs only (= LPF); own-target only; both (CXF); plus time block (S2); plus Phase-B
   witnesses.
2. **Covariance:** learned C vs identity (= MMAE / `rpoe`-like) vs diagonal.
3. **Inference:** exact BMA vs soft precision scaling `λ_m · (1 − r_m)` vs MAP subset vs learned map.
4. **Likelihood:** Gaussian vs Student-t (ν = 5); heteroscedastic block on or off.
5. **Hyperparameters:** a sensitivity grid over pi, kappa, k (4, 8, 16) and T (clean-FAR targets of 0.5, 1 and 5%).
6. **Auditor:** frozen V-JEPA vs frozen align vs fine-tuned (no split) vs keep-alive (`inventor_crossmodal.md` §5)
   vs PaP-FT.
7. **S1:** P-time vs P-full; gap ∈ {0, 10, 20 frames}; σ ∈ {2, 8}; independent vs shared distractor.

### 4.5 Compute estimate (free Kaggle T4, about 30 GPU-h per week, 12-hour sessions)

The per-run times are *estimates* scaled from the FYP-1 config (d_model 128, 2 layers, 30 pre-epochs, batch 256,
16k clips). **Week 1 must replace them with measured times.**

| Block | Runs | GPU-h / run (est.) | GPU-h |
|---|---|---|---|
| Pilot (CXF fit + evaluation on existing checkpoints) | 2 objectives × 3 seeds × {FR, FT} | 0.3 | about 4 |
| Phase A full grid (CXF + 12 baselines, about 40 conditions, 5 seeds; seeds 3-4 need new stage-1 and stage-2 runs) | 2 objectives × 2 new seeds × (stage 1 + 2 regimes) + evaluation of 3 objectives × 5 seeds × 2 regimes × 13 fusions | 0.5 (train) / 0.1 (eval) | about 45 |
| Corruption-supervised baselines (odd-one-out, quality head) | 2 × 5 seeds × 2 regimes | 0.4 | 8 |
| MOSI replication | 3 objectives × 5 seeds, full pipeline | 0.2 | 6 |
| Phase B CROSSWIRE witnesses | 5 seeds × (stage 1 at about 1.5–2x + evaluation) | 1.0 | 10 |
| S1 Persistence (MOSEI D grid: 2 variants × 2 σ × 2 patterns × 3 seeds, plus baselines) | about 40 | 0.6 | 24 |
| S1 on Huang's toy (15 seeds × 3 methods) | 45 | 0.1 | 5 |
| S2 Time Auditor (stretch) | 5 seeds | 1.0 | 5 |
| CH-SIMS v2 (stretch) | 3 objectives × 3 seeds | 0.5 | 5 |
| Re-runs and debugging margin (×2) | | | about 110 |
| **Total** | | | **about 220 GPU-h over 26 weeks**, against about 780 available. About 3.5x headroom. |

### 4.6 Timeline (6 months, 26 weeks)

| Weeks | Work | Milestone / decision |
|---|---|---|
| 1 | **Go/no-go pilot** (section 4.7) on FYP-1 checkpoints. Measure per-run times. | GO / GO-narrow / NO-GO |
| 2–4 | `cxf.py`, the evaluation hooks, and the classical-FDI and deep baselines. Phase A on MOSEI (3 seeds). | H1–H7 first numbers |
| 5–6 | CXF-C conformal; ablations 1–5; seeds 3–4. | **Mid-term table** (H1–H7, 5 seeds) |
| 7–8 | MOSI port (H12); corruption-supervised baselines. | |
| 9–12 | Phase B CROSSWIRE witnesses (`xwire`), retrain, re-evaluate; matched-vs-shuffled edge AUROC. | Decide whether Phase B enters the headline |
| 13–17 | S1 Persistence Targets: MOSEI D grid (H8), Huang's toy over 15 seeds (H9), Persistence witnesses inside CXF (H10). | S1 result |
| 18–20 | S2 Time Auditor (H11) **or** CH-SIMS v2 natural-conflict study, whichever is more promising. | Stretch |
| 21–22 | Freeze experiments; final 5-seed tables; failure analysis (which clips does CXF get wrong?). | Results freeze |
| 23–26 | Writing, figures, viva preparation; a 2-week buffer. | Submission |

### 4.7 Week-1 go/no-go pilot (no training, existing checkpoints)

**Setup.**
- Auditors: frozen `v2_experts_*_vjepa_s{0,1,2}` and `*_align_s{0,1,2}` with their text EMA target. Note the asymmetry
  in FYP-1: raw disagreement AUROC is 0.84 for align but 0.56 for V-JEPA. Both auditors must be tested.
- Workers: the saved stage-2 FR-PoE and FT-PoE models (`runner.py` saves them).
- Evidence: Phase A o = [P_t mu_t, Q_t e_t, P_a mu_a, P_v mu_v].
- Fit on clean validation halves only.
- Conditions: clean; noise T, A and V at λ ∈ {0.5, 1}; swap A, V and AV; the B channel (+N checkpoints).
- Scoring is test-split only.

**Measured quantities.**
- The inventor's "afternoon" diagnostic: the distribution of d2_cond(t) and d2_marg(t) on noise-text vs clean.
- AUROC and localization of r_m.
- CXF-BMA MAE, Cov90, share and tracking for both workers.
- Baselines on the same clips: sigma, raw disagreement (`pairwise_skl`), `rpoe` weight, χ² gate.

**Go criteria. All must hold, as a mean over 3 seeds.**

| Criterion | Pass if |
|---|---|
| **G1 (alien input)** | AUROC(r_t) ≥ 0.90 on text noise λ = 1 with the V-JEPA auditor, and the clean FAR is at most 2%. |
| **G2 (cross-examination adds something)** | For at least one auditor, AUROC(r_a) on swap A **and** AUROC(r_v) on swap V are ≥ 0.70, **and** ≥ that auditor's raw disagreement AUROC + 0.03. |
| **G3 (repair a real failure)** | On the FT worker under text noise λ = 1: Cov90 (D3a) ≥ 0.83 (from 0.746) **and** MAE ≤ 0.85 (from 0.870). Clean TAV MAE increases by at most 0.005. |

**Decision.**
- **GO**: all three hold. Run the full plan.
- **GO-narrow**: G1 and G3 hold but G2 fails. Continue, but Phase B (trained witnesses with e_a and e_v) moves to
  weeks 2–5 and becomes mandatory. Swap claims are conditional on it. First run the CROSSWIRE edge check (AUROC of
  matched vs within-batch-shuffled residuals for a↔v below 0.6 means the AV edges are uninformative), and restrict
  the claims to "text vs rest" if it fails.
- **NO-GO**: G1 fails. Noise-text beliefs then land inside the clean joint, which is the attractor effect of
  "Biased Dreams" (arXiv 2604.25416, via trends.md). The latent evidence is blind, so switch to the fallback
  (section 5.2).

**Caveat a reviewer will raise.** G1 alone could be met by any marginal OOD score: d2_marg of the text block. It is
*necessary* but not *sufficient*. The scientific claim of CXF rests on **G2/H2** (swaps are marginally
in-distribution) and on **H4** (calibrated repair). The write-up must say so, and report the "marginal-only"
ablation next to CXF everywhere.

---

## 5. Risks and fallback

### 5.1 Risks

| Risk | Likelihood | Effect | Mitigation |
|---|---|---|---|
| Weak cross-modal information on MOSEI (AV-only MAE 0.81). The conditional densities are broad, so swaps stay undetectable. | High for the V-JEPA auditor (disagreement AUROC 0.56), medium for align (0.84) | H2 fails | Test both auditors in the pilot; Phase B witnesses; report the honest operating curves. Swap of A/V barely changes MAE (0.727 → 0.727 FR V-JEPA), so H3/H4 do not depend on it. |
| Attractor effect: corrupted inputs map *onto* the clean manifold in the latent | Medium | G1 fails | Own-target block e_t (input-sensitive); S2 temporal block; fallback F1. |
| Gaussian misspecification makes the posterior overconfident, so the clean FAR is too high | Medium | H7 fails | Temperature T set by the clean FAR; Student-t ablation; report the full grid. |
| Two-of-three ambiguity (swap AV) | Certain in principle | Mislocalization | S2 time block; report it as a stated identifiability limit. |
| A shared distractor inflates agreement, so CXF trusts the nuisance | Certain under D-shared | CXF does nothing there | That is exactly S1's job (H10). |
| A reviewer says "this is just MMAE / odd-one-out" | High | Novelty criticized | Cite Chow, MMAE, Wang, Yang and Wu up front. Claim the *learned-latent, clean-only, inferred-missingness and calibrated-regression* deltas. Include the identity-covariance MMAE baseline to show the learned parity relations matter. |
| Persistence windows lose sentiment content | Medium | H8 clean-cost criterion fails | 3 window pairs per clip; report the cost; gap sweep. |
| Seed variance (Huang: 1/15 catastrophic) | Medium | Noisy conclusions | 5 seeds on MOSEI, 15 on the toy; report per-seed results, not just means. |

### 5.2 Fallback if the pilot says NO-GO (G1 fails), or G2 *and* Phase B fail

- **F1: keep the inference engine, change the evidence.** The switch posterior and the BMA/CXF-C machinery are
  evidence-agnostic. Replace the latent blocks with *input-level* evidence:
  - the Time Auditor (S2 promoted to core);
  - or the EYP input-regeneration energy.
  
  The thesis becomes "trust earned by temporal self-consistency", which is Chow's temporal redundancy. It is less
  novel but likely to work on pure noise. Swap claims are dropped.
- **F2: promote S1 to the headline.** Persistence Targets plus Huang's-toy replication plus the identifiability
  limit form a self-contained noisy-TV paper, and the main cost (MOSEI D grid plus toy) is already budgeted.
  CXF becomes a carefully documented negative result ("cross-modal corroboration is too weak on MOSEI features to
  localize swaps"), which a jury will respect if it is quantified against the classical baselines.
- **F3: minimum viable FYP** (if everything is weak). The evaluation contribution (G5 in `trends.md`): the first
  coverage / CRPS / tracking-ratio benchmark of 15+ reliability-fusion methods under corruption on MOSEI and MOSI,
  including classical FDI baselines. Low novelty, but solid and publishable as a benchmark/workshop paper.

---

## 6. Scores of every original idea (both inventor files)

Scale 1–5. Novelty: 5 = no close prior. Impact: on the brief's failures and to a jury. Feasibility: 5 = easy for one
undergraduate on Kaggle. **Risk: 5 = most likely to fail.** The novelty scores take the audit in section 1 into
account.

| Idea (source) | Novelty | Impact | Feasibility | Risk | Note |
|---|---|---|---|---|---|
| CROSSWIRE: M x M residual graph + earned-trust PoE (crossmodal §1) | 2 | 4 | 4 | 3 | Generalized voting over parity relations (Chow 1980); cross-modal reconstruction fault detection (Jensen 2026). Kept as Phase-B witnesses. |
| Explain-Away Inference: exact switch posterior with top-down own-target check (crossmodal §2) | 3 | 4 | 4 | 3 | MMAE / Bernoulli-indicator fusion in a learned latent. Its own-target block is merged into CXF. |
| Time Auditor (crossmodal §3) | 2 | 3 | 3 | 3 | Temporal redundancy. S2 (stretch). |
| Persistence Targets (crossmodal §4) | 3 | 4 | 4 | 3 | Feichtenhofer / TCN / Gulati are adjacent; the conjunction for a JEPA target is new. **S1.** |
| Keep-alive fine-tuning (crossmodal §5) | 1 | 3 | 5 | 2 | Auxiliary SSL during fine-tuning is standard. Ablation. |
| Auditor/worker split (crossmodal §6) | 2 | 3 | 5 | 1 | Design rule, adopted. |
| Liar's Posterior Fusion (probabilistic §1) | 3 | 4 | 5 | 3 | MMAE + Yang-style subset mixture, but label-free and clean-only. **The core of CXF.** |
| Bates-Granger PoE (probabilistic §2) | 2 | 3 | 5 | 3 | Bates-Granger (1969, inventor citation, not re-verified) and CoDE (2505.01134). Possible extension of the within-subset PoE; not in the headline. |
| Explain-Yourself Precision (probabilistic §3) | 2 | 3 | 3 | 4 | VB Student-t fusion with reconstruction energy. Weak on BERT-PCA tokens. Fallback F1 evidence. |
| Too-Good-To-Be-True CCA target nulling (probabilistic §4) | 3 | 3 | 5 | 3 | Strong on the linear benchmark, but invites a "designed for the benchmark" critique. Possible S1 baseline/ablation. |
| Bits-Budget JEPA (probabilistic §5) | 2 | 2 | 4 | 4 | GECO-style constraint (inventor citation). Diagnostic value for L6, a separate story. |
| Posterior-as-Prior fine-tuning (probabilistic §6) | 1 | 3 | 5 | 2 | L2-SP / KL anchoring. Ablation. |

---

## 7. Reference list (opened during this review)

Classical FDI and robust fusion:
- Chow 1980, MIT LIDS-TH-1055 (NTRS 19810003928).
- Roumeliotis, Sukhatme, Bekey, sensor fault detection with MMAE (IROS'98 per the authors' page).
- Wang, Li, Fang, Wang, IEEE SPL 25(8), 2018.
- Wang et al., arXiv 1909.10006 / IEEE JSEN 2022.
- Kumar, Garg, Zachery, IEEE Sensors J. 7(5), 2007.
- Zhang et al., *Acta Aeronautica et Astronautica Sinica* 41(S2), 2020.

Deep reliability and robust fusion:
- Yang et al., CVPR 2021 / arXiv 2206.12714.
- Lee et al., arXiv 2012.00201.
- Jensen et al., arXiv 2609.16314.
- PRIME (Kumar et al.), arXiv 2608.03475.
- Wu, Zhu, Liu, *Electronics* 15(16):3624, 2026.
- MRCF (Meng et al.), arXiv 2608.03611.
- Palo et al., arXiv 2610.29621.
- Bezirganyan et al., AISTATS 2025 / arXiv 2412.18024.
- MoNIG (Ma et al.), arXiv 2111.08456 (NeurIPS 2021).
- gPoE (Cao & Fleet), arXiv 1410.7827.
- CoDE (Mancisidor et al.), arXiv 2505.01134.
- MultiOOD (Dong et al.), arXiv 2405.17419.
- C²MF (Tenali et al.), arXiv 2603.26629.
- AdaPGC (Li et al.), arXiv 2611.03232.
- RVAE (Eduardo et al.), AISTATS 2020 (PMLR 108).
- MADJEPA, arXiv 2611.25194.
- SC-JEPA, arXiv 2602.04643.

Conformal:
- Azizi, arXiv 2608.07795.

Persistence and distractors:
- Feichtenhofer et al., CVPR 2021 / arXiv 2104.14558.
- Sermanet et al., arXiv 1704.06888.
- Gulati & Nemenman, arXiv 2606.07770.
- Jenni et al., arXiv 2302.07702.

Data:
- MMSA repository (github.com/thuiar/MMSA).
- CH-SIMS v2 (Liu et al., ICMI 2022, arXiv 2209.02604); README and HF mirror AZYoung/SIMSV2_processed.
