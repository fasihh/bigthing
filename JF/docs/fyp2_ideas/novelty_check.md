# Novelty check for Cross-Examined Fusion (CXF)

Date of check: 27 Nov 2026 (submission 4 Dec 2026). Stance: adversarial. My job was to find overlap, not to reassure.

Inputs read: `JF/proposal/proposal.tex` (all of it) and `ideas/proposal_review.md` sections 1 and 2.

Method of checking: every paper below was opened through its arXiv abstract page, arXiv HTML or PDF full text,
or its publisher page, via WebFetch. "Full text" means that I read the method section, not only the abstract.
Pages I could not open are listed in section 5 and are not used as evidence.

---

## 1. The seven named papers

The columns in each table are the questions the task asked: inputs, supervision, whether the paper localises the
corrupted modality, whether it re-weights fusion, whether it is probabilistic or exact, and calibration.

### (a) Lee, Tan, Zhu, Bohg, "Detect, Reject, Correct: Crossmodal Compensation of Corrupted Sensors", arXiv 2012.00201 (Dec 2020)

Read in full text (PDF).

| Aspect | What the paper does |
|---|---|
| Inputs | RGB, depth, force/torque, proprioception. Simulated peg insertion (RoboSuite), with an SAC policy on top. |
| Model | Each modality encoder outputs a Gaussian. The Gaussians are fused by a **product of experts**, and the model is trained with an ELBO. |
| Training | **Modality dropout**: one of the three modalities is dropped at each step, plus a latent distance loss to the full-modality latent. Training data is clean. |
| Detection | Per-modality **unimodal reconstruction error** (L2). Thresholds are chosen per modality "to maximize AUROC for detecting corruption in the validation dataset". So corrupted data is used to set the threshold. If several modalities are flagged, the one with the largest z-score is chosen. |
| Localises? | Yes. |
| Fusion change | **Hard reject**: the flagged expert is removed from the PoE, and the remaining subset compensates. |
| Probabilistic / exact? | Probabilistic latent, but the decision is a deterministic threshold. There is no posterior over configurations. |
| Calibration | None. |

**Overlap with CXF.** This is the closest *architecture*. A PoE of Gaussian experts trained with modality dropout,
self-supervised detection of which modality is corrupted, and removal of that expert so that the subset prediction is
used. That is **"corruption treated as missingness" in hard form, in 2020**.

**What CXF adds.**
- The evidence is cross-modal (a conditional residual), not unimodal reconstruction. Lee's score cannot detect a
  swapped but realistic modality.
- The threshold uses no corrupted data. Lee tunes it on corrupted validation data.
- An exact soft posterior and BMA, instead of a hard reject.
- Calibrated regression and conformal intervals.

**The proposal does not cite this paper. It must.**

### (b) Jensen, Hammershøi, Wiśniewski, Fink, "Robust Fault Detection in Mechanical Multimodal Time Series via Self-Supervised Cross-Modal Reconstruction", arXiv 2609.16314 (Sep 2026)

Read in full text (HTML, about 200k of 258k characters).

| Aspect | What the paper does |
|---|---|
| Inputs | Industrial multimodal time series: AMPERE, KAIST and IMAD-DS (vibration, current, voltage, acoustic and similar). |
| Model | One shared model with per-modality encoders, self-attention fusion and per-modality decoders. Losses: self-reconstruction, shared-latent reconstruction, and **leave-one-out cross-modal reconstruction**. |
| Supervision | "trained exclusively on healthy samples". **Clean only.** |
| Score | Per-modality cross-modal reconstruction error s_m, used as a *system* anomaly score. |
| Threshold | Adaptive: an EWMA mean + kappa sd per operating condition. |
| Localises? | Not claimed and not evaluated. Per-modality scores exist, but they are not used to isolate a sensor. |
| Fusion change | None. Detection only. |
| Probabilistic / exact? | No. The authors state that there is no probabilistic model. |
| Calibration | None, beyond thresholding. |

**Overlap with CXF.** It is clean-only and uses cross-modal predictive evidence. This is the closest prior art for
"the system model is a learned cross-modal predictor fitted on clean data".

**What CXF adds.** Localisation, fusion and prediction under corruption, and a posterior with calibration.

**The proposal does not cite this paper. It must.**

### (c) Yang, Lin, Barman, Condessa, Kolter, "Defending Multimodal Fusion Models against Single-Source Adversaries", arXiv 2206.12714 (CVPR 2021 per the review; the arXiv page lists no venue)

Read in full text (PDF).

| Aspect | What the paper does |
|---|---|
| Odd-one-out network | Takes the features z and outputs k+1 probabilities: one for "modality i is perturbed" for each i, and one for "none is perturbed". |
| Supervision | Trained with cross-entropy on **adversarially perturbed** features (label i) and clean features (label k+1). Training and test use the same PGD family. |
| Fusion | z_out = sum_{i=1}^{k+1} o(z)_i * e_i(z), where e_i = NN(z_{-i}) is the leave-one-out fusion and e_{k+1} uses all modalities. |
| Tasks | EPIC-Kitchens, KITTI, **CMU-MOSI** (sentiment). |
| Calibration | None. The odd-one-out quality is reported only as detection rates. |

**Overlap with CXF.** This is the closest *inference structure*: a probability-weighted mixture over leave-one-out
subset predictors, where the weights are P(which modality is corrupted).

**Differences.**
- Yang is supervised on the attack family it is tested on.
- Yang considers only single-source corruption (k+1 hypotheses, not 2^M).
- Yang mixes features, not predictive distributions.
- Yang has no calibration.

**CXF is, structurally, "Yang's robust layer, made generative and label-free, mixing predictive densities".** Say
so.

### (d) Wu, Zhu, Liu, "Reliability-Aware Multi-Modal Sentiment Analysis Under Missing and Corrupted Modalities", Electronics 15(16):3624, 2026

Read in full text (MDPI HTML, about 200k of 219k characters).

| Aspect | What the paper does |
|---|---|
| Datasets | MOSI and **MOSEI**. |
| Model | Per-modality Dirichlet evidential head, plus a sigmoid quality head. |
| Supervision | The quality head is regressed to "1 − injected severity": "During training, the quality head receives continuous targets derived from known synthetic corruption severity." |
| Fusion | Reliability = availability × quality × uncertainty penalty, then a JS-disagreement penalty, normalised over the available modalities. |
| Corruption protocol | Training corruptions at severity 0.1–0.5. **Unseen test operators** too (phrase deletion, token permutation, impulse noise, channel dropout, block dropout, temporal shift). |
| Main task | Binary classification. MAE and correlation come from a separate MSE head. |
| Metrics | NLL, Brier, **ECE** (e.g. 4.2 pp on MOSEI under mixed stress), Risk@80%, and **clean-vs-corrupted AUROC** (full model 89.0). |
| Coverage / CRPS | None. |
| Limitation they state | The quality head "is supervised by controlled severity and may rank unseen degradation imperfectly". |

**Overlap with CXF.** Same domain (MOSEI), the same per-modality corruption probability, the same AUROC metric, a
calibration metric, and an unseen-corruption test.

**Differences.** Wu is supervised by injected corruption. It reports no regression coverage and no CRPS, and it has
no swap/mismatch corruption.

### (e) PRIME, arXiv 2608.03475 (Aug 2026)

Read in full text (HTML).

| Aspect | What the paper does |
|---|---|
| Score | A per-modality log-variance s_m from a Transformer over a 4-dim descriptor: energy confidence, ensemble disagreement, **cross-modal JS consensus**, and embedding norm. |
| Supervision | "Because modality-reliability annotations are unavailable", the weakness sigma(s_m) is **regressed onto synthetic injected severity**. Class labels are also used. |
| Fusion | Cross-modal residual restoration from high-precision sources, then inverse-variance weighted fusion. |
| Task | Intent **classification** (MIntRec, MIntRec2.0). |
| Calibration | None. No unseen-corruption protocol in the body. |

**Overlap with CXF.** Cross-modal consensus as evidence for per-modality reliability, and precision-weighted fusion.

**Differences.** PRIME is corruption-supervised and classification-only, and it has no posterior or calibration.

### (f) Palo, Tiwari, Carletti, Thakur, Gu, Clifton, "Reliability-Aware Probabilistic Fusion for Multimodal Learning with Incomplete Evidence", arXiv 2610.29621 (Oct 2026)

Read in full text (HTML, about 100k of 108k characters).

| Aspect | What the paper does |
|---|---|
| Model | A Gaussian PoE. Each expert's precision is tempered by r_m = sigmoid(MLP([mu_m; log sigma_m^2])). |
| Supervision | "learned end-to-end through supervised fusion, without corruption-ranking supervision". So it uses **task labels and no corruption**. Corruption is used only at evaluation. |
| Localises? | No, and the weights respond **inconsistently**. ECG reliability *rises* under severe ECG corruption. |
| Task | Binary classification (Symile-MIMIC, HAIM-MIMIC-MM). Metrics: AUROC and AUPRC; Brier is supplementary, "not interpreted as calibration". |
| Open problem quoted | "learning dependable quality signals remains an open challenge". **The paper does not say "label-free".** |

**Overlap with CXF.** It is the reliability-weighted PoE baseline. Its failure is exactly CXF's motivation:
self-derived weights do not follow corruption.

**Fix needed in the proposal.** The proposal says Palo's "authors list label-free reliability signals as an open
problem". That is a mis-attribution. Their words are "dependable quality signals". Also, their weight is learned from
*task labels only*, not from corruptions. Line 75 lumps Palo with the corruption-trained methods only implicitly, so
check the sentence.

### (g) Azizi, "Conformal Calibration for Multi-Modal Regression with Missing Modalities", arXiv 2608.07795 (PMLR v329, 2026)

Read in full text (HTML, about 200k of 441k characters).

| Aspect | What the paper does |
|---|---|
| Methods | Split CQR; disagreement-scaled CQR with a_gamma(x) = sqrt(1 + gamma d(x)^2); Mondrian by disagreement bins or by availability/mask. |
| Disagreement score | RMS deviation of the per-source predictions, from **supervised** per-source predictors. |
| Missing vs corrupted | **Missingness only.** Corruption appears only in related work ("sources absent or corrupted"). The masks are fixed and known. Nothing infers the mask. |
| Datasets and metrics | Regression (SRED, Mercari, Pawpularity, IMDB-WIKI): coverage, width, **interval CRPS**. |
| Fusion change | None. It does not identify the unreliable source. |

**Overlap with CXF.** Per-mask conformal for multimodal regression, and CRPS and coverage evaluation. This is
CXF-C's building block.

**What remains CXF's.** Inferring the mask under corruption and mixing the per-mask intervals by a posterior.

---

## 2. New papers found in the sweep, ordered by closeness

I ran 27 WebSearch queries, most in "extended" mode. They covered:
- label-free and clean-only corrupted-modality detection;
- modality inconsistency in MSA;
- BMA and mixtures over modality subsets;
- conformal prediction under corruption;
- robust and mixture multimodal VAEs and outlier PoE;
- multi-view anomaly detection;
- sensor fault isolation with deep learning;
- test-time reliability;
- multimodal OOD;
- missing vs corrupted calibration;
- JEPA reliability;
- caption mismatch;
- one-class detector → missing token;
- mixtures of mask-conditional conformal quantiles;
- learned parity relations;
- world-model surprise.

### NEW-1. Mohapatra, Sui, Yang, Lundell, Xia, Zhu, "Source-Learned Reliance for Selective Test-Time Adaptation of Multimodal Time Series" (CARAT), arXiv 2610.07499 (Oct 2026)

Read in full text (HTML, first 100k of 196k characters). **This is the most dangerous paper for delta 3.**

- **Detector.** Per-modality one-class detector on reconstruction-error and log-spectral features, using a squared
  **Mahalanobis** distance. It is **fit on clean source data only**, with the threshold at "the 95th percentile of
  clean-train scores". It gives a per-sample *suspect set*, so it localises.
- **What happens to a flagged stream.** It is **replaced by "the missingness symbol employed by the modality-dropout
  curriculum"**, provided the remaining streams keep enough reliance mass. Otherwise it is attenuated by a learned
  gate. That is **corruption-as-missingness, clean-only, localised, published one month ago.**
- **Decisions.** Hard routing (beta_i >= 0.8). There is no posterior and no mixing over subsets.
- **Other.** Detection is unimodal (per-stream features, not cross-modal), so it cannot see a swapped but realistic
  stream. It does test-time entropy adaptation, classification only (DSADS, PAMAP2, DaliaHAR, WESAD). It has **no
  calibration, conformal or coverage**. The authors list uncertainty and abstention as future work.

### NEW-2. Zollicoffer, Chopra, Yan, Ma, Eaton, Riedl, "World Model Robustness via Surprise Recognition" (WISER), arXiv 2512.01119 (Dec 2025)

Read in full text (PDF).

- **Model and score.** A Dreamer-style **latent world model**. Per-sensor Bayesian surprise is KL[q(z|h,x) || p(z|h)],
  computed with each sensor isolated.
- **Rejection.** It rejects high-surprise sensors and "chooses the input subset minimizing the Bayesian surprise". The
  rejection is greedy, and the authors compare it with an **exhaustive 2^N subset search**.
- **Training.** "Representation dropout" during world-model training. The rejection mechanism is not trained on
  corruptions.
- **Threshold and calibration.** The threshold is a heuristic 5 sd. No calibration. RL control tasks (CARLA, Safety
  Gym).

**Overlap with CXF.** "A learned latent world model is the system model, its surprise rejects sensors, and it picks
among modality subsets". This kills any broad claim that using a *learned latent world model* for FDI is new.

**What remains CXF's.** The evidence is conditional and cross-modal (WISER's per-sensor surprise is essentially
marginal), an exact posterior instead of a min-surprise argmin, BMA, and calibration. Also, JEPA specifically
(non-generative) rather than an RSSM.

### NEW-3. Wang and Lan, "Towards a Hierarchical Bayesian Model of Multi-View Anomaly Detection", IJCAI 2020, pp. 2420–2426

Read in full text (PDF).

- **Model.** A shared latent z with view-specific projections, using a Student-t prior and noise. It is a robust
  probabilistic CCA, i.e. a **joint density across views**, trained **only on normal data** ("inductive
  semi-supervised").
- **What it detects.** "Class-outliers" = **inconsistency across views**. The threshold is calibrated on a
  **view-swapped** mixture set.
- **What it does not do.** It gives an instance-level score only: **no localisation of the view**, no fusion, no
  prediction.

**Overlap with CXF.** The clean joint density over views and detecting swapped views from cross-view inconsistency.
This is CXF's evidence model (Gaussian joint ≈ linear pCCA), minus localisation and fusion.

### NEW-4. Mišić and Horii, "Sensory Precision Inference for Multimodal Arbitration under Uncertainty", arXiv 2609.15065 (Sep 2026)

Read in full text (HTML).

- Per-modality precision is a latent inferred at test time by free-energy minimisation. Training is **unsupervised**
  and corruption appears only at evaluation.
- "the inferred precision of the corrupted modality decreases with increasing noise". So it gives label-free
  re-weighting of fusion by prediction error.
- There is no discrete corruption variable, no posterior over configurations, and no calibration. The data is toy
  multimodal MNIST.

**Overlap with CXF.** Label-free, prediction-error-driven down-weighting of a corrupted modality. A
predictive-coding cousin.

### NEW-5. Tenali, Sidheekh, Mathur, Blasch, Kersting, Natarajan, C²MF, arXiv 2603.26629 (v2, Jun 2026)

Read in full text (HTML).

- Credibility = KL(full posterior || posterior with modality i marginalised out), **computed exactly** in a
  conditional probabilistic circuit. Fusion is credibility-weighted.
- It is **trained with labels on corrupted data** (conflict ratio 0.7). There is no posterior over which modality is
  corrupted. Classification only, with no calibration metrics.

**Overlap with CXF.** "Exact" tractable inference over modality subsets for reliability. Cite it to pre-empt
"exact inference over subsets is not new". CXF's exactness is over *corruption configurations under a clean density*,
which C²MF does not do.

Predecessor, opened at abstract level: Sidheekh et al., "Credibility-Aware Multi-Modal Fusion Using Probabilistic
Circuits", arXiv 2403.03281 (2024).

### NEW-6. Conformal per missing-modality pattern is now crowded (all missingness only, none for corruption)

- Moayedikia, "Conformal Fusion Under Missing Modalities" (MCCF), arXiv 2608.07183 (Aug 2026). Abstract only.
  Mondrian conformal keyed on the presence mask, with a modality-dropout backbone.
- Ajit, "Guarantees That Survive a Missing Scan: Modality-Conditional Conformal Prediction for Multimodal Medical
  Diagnosis", Research Square rs-10336064 (Jul 2026). Abstract page. Per-pattern Mondrian. Corruption is not
  addressed.
- GitHub `SilentEyes-0/modality-conditional-conformal-meld` (README, no paper). Per-pattern Mondrian conformal on
  MELD (classification) under missing **and corrupted** inputs. It finds that text corruption breaks coverage unless
  the calibration *matches the corruption*. It does not detect or infer corruption. This is useful *supporting*
  evidence for CXF's motivation, and an unpublished artefact.

### Weaker overlaps (opened; cite at most one line or none)

| Paper | Opened | What it does | Why it is weaker |
|---|---|---|---|
| Jiang et al., CPSC, arXiv 2605.03820 | full text | Conformal reliability scores on noisy-modality *classification*, used during training. | Label-dependent; no localisation or posterior. |
| Iwata and Yamada, arXiv 1411.3413 (published at NeurIPS 2016; I could not open that PDF) | abstract | Multi-view anomaly = inconsistent views; DP over latent vectors; instance-level. | Ancestor of Wang and Lan. |
| gPoE for noisy environments, arXiv 2211.03587 | abstract and body summary | Learned softmax precision weights. | Trained in the noisy setting; no calibration. |
| Tian et al., UNO, arXiv 1911.05611 | abstract | Uncertainty-scaled noisy-or fusion for segmentation. | — |
| Park et al., MoME, arXiv 2503.19776 | abstract | Per-query routing among camera, LiDAR and fused experts. | Router quality signal unspecified. |
| Patel et al. (Sandia), arXiv 2511.16816 | full text | Dirichlet trust weights as tempered likelihood powers inside one Bayesian model; LOO discordance; near-nominal coverage. Physical forward models; BMA used as a *baseline*. | Not deep, not per-instance. |
| Altinses and Schwung, arXiv 2603.25103 | abstract | Self-supervised fault detection and correction in a multimodal AE pre-trained on clean data. | No posterior or calibration. |
| Li et al., AdaPGC, arXiv 2611.03232 | abstract | Test-time adaptation with class-conditional Gaussians under modality reliability shift. | Classification. |
| AlMarri et al., RobustA, arXiv 2511.07276 | abstract | Corruption-level-weighted AV anomaly detection. | — |
| Wilkinghoff et al., arXiv 2604.13252 | abstract | Perspective piece on context-conditional multimodal anomaly detection. | No method. |

---

## 3. Ranked list of substantial overlaps

| Rank | Paper | Link | What it pre-empts |
|---|---|---|---|
| 1 | Lee et al., Detect-Reject-Correct, arXiv 2012.00201, 2020 | https://arxiv.org/abs/2012.00201 | PoE + modality dropout + self-supervised per-modality detection + drop expert → subset prediction. |
| 2 | Mohapatra et al., CARAT, arXiv 2610.07499, 2026 | https://arxiv.org/abs/2610.07499 | **Clean-only** per-modality Mahalanobis detector → replace with the **missingness token from modality-dropout training**. |
| 3 | Yang et al., odd-one-out robust fusion, arXiv 2206.12714 (CVPR 2021) | https://arxiv.org/abs/2206.12714 | P(modality i corrupted)-weighted mixture of leave-one-out fusions; includes MOSI. |
| 4 | Zollicoffer et al., WISER, arXiv 2512.01119, 2025 | https://arxiv.org/abs/2512.01119 | Latent world model surprise selects the trusted sensor subset (vs exhaustive 2^N); trained with representation dropout. |
| 5 | Wu, Zhu, Liu, Electronics 15(16):3624, 2026 | https://www.mdpi.com/2079-9292/15/16/3624 | MOSEI, per-modality quality, clean-vs-corrupted AUROC, ECE and NLL, unseen corruptions (supervised). |
| 6 | Wang and Lan, IJCAI 2020 | https://www.ijcai.org/proceedings/2020/0335.pdf | Clean-only joint (Student-t pCCA) density over views detects view-swaps (no localisation). |
| 7 | Jensen et al., arXiv 2609.16314, 2026 | https://arxiv.org/abs/2609.16314 | Clean-only cross-modal reconstruction evidence (no localisation or fusion). |
| 8 | Mišić and Horii, arXiv 2609.15065, 2026 | https://arxiv.org/abs/2609.15065 | Label-free precision inference down-weights the corrupted modality. |
| 9 | Tenali et al., C²MF, arXiv 2603.26629, 2026 | https://arxiv.org/abs/2603.26629 | Exact marginalisation over modalities for credibility (supervised). |
| 10 | Azizi, arXiv 2608.07795, 2026; MCCF, arXiv 2608.07183 | https://arxiv.org/abs/2608.07795 | Per-mask Mondrian conformal for multimodal regression and CRPS. |
| 11 | PRIME, arXiv 2608.03475; Palo et al., arXiv 2610.29621 | https://arxiv.org/abs/2608.03475 | Reliability-weighted fusion; already cited. |

---

## 4. Verdict on each claimed delta

**Delta 1: the system model is a self-supervised JEPA latent world model, and the evidence is cross-modal latent prediction.**
- **Partly holds.** I found no work that uses a *JEPA* (non-generative, latent-predictive) model as the FDI model for
  modality corruption.
- The broader claim "a learned latent world model as the system model, with its surprise used to reject sensors" is
  **not new**: WISER (2025), Lee (2020, a VAE-PoE), and Jensen (2026, cross-modal reconstruction).
- Keep only the narrow version, and justify *why JEPA matters*. For example: evidence in a non-reconstructive latent
  ignores pixel or nuisance detail. Without an ablation against a reconstruction-based witness, this delta is cosmetic.

**Delta 2: an exact posterior fitted on clean, unlabelled data only.**
- **"Clean-only" alone is not novel.** CARAT, Jensen, Wang and Lan, WISER and Mišić are all clean-only or
  corruption-free. Lee is clean-trained, though its threshold is tuned on corrupted validation data.
- **What still holds:** an *exact posterior over all 2^M corruption configurations*, built from a *conditional
  (cross-modal) clean density* with a decoupled and broadened outlier block, and used *inside deep multimodal fusion*.
  I found no deep-learning paper that does this. WISER does an argmin over subsets, Yang uses supervised k+1 classes,
  and C²MF does exact marginalisation for credibility, not a corruption posterior.
- It is classical MMAE, and it is close to the robust-PCA/pCCA FDI literature. Keep the "classical, rebuilt" framing.

**Delta 3: corruption as inferred missingness, with per-subset conformal transfer.**
- **The hard version is pre-empted twice.** Lee 2020 drops the flagged expert from a modality-dropout PoE. CARAT 2026
  replaces flagged streams with the modality-dropout missingness token, using a clean-only detector.
- Per-mask conformal itself is published: Azizi, MCCF, MC².
- **What still holds** is narrower. It is (i) the *soft* version, BMA of subset predictive densities weighted by the
  posterior, and (ii) *mixing per-mask conformal intervals by that posterior* so that coverage under silent
  corruption is restored without corrupted calibration data. I found no prior for (ii).
- **Caveat a reviewer will raise.** A posterior mixture of per-mask conformal intervals has **no finite-sample
  guarantee** unless the posterior is right. The guarantee holds exactly only if the MAP mask is the true
  missingness-equivalent mask with probability 1. State that CXF-C is "conformal-inspired" and that its coverage is
  measured empirically, or derive a bound in terms of posterior error (coverage ≥ 1 − alpha − P(wrong mask)).

**Delta 4: auditor/worker split.**
- **Holds weakly.** CARAT and WISER also separate the detector from the predictor. CARAT's detector works on input
  features and is independent of adaptation.
- The *finding* that fine-tuning erases cross-modal agreement (from Part I), and that a frozen self-supervised
  auditor preserves it, is new as far as I found. Present it as an empirical design finding, not a method novelty.

**Delta 5: calibrated regression evaluation on MOSEI and COCO.**
- **Holds, with a qualifier.**
  - Wu 2026 already reports ECE, NLL and clean-vs-corrupted AUROC on MOSEI under corruption, including unseen
    corruptions (as classification).
  - Azizi reports coverage and CRPS for multimodal regression, under missingness only.
  - No paper found reports regression coverage, CRPS or tracking under *silent* corruption, *including
    swaps/mismatches*, for MOSEI. COCO with image and two captions, used as a calibrated regression benchmark, is
    also unreported.

**Overall.** The novelty is real but narrower than the current paragraph implies. The defensible core is:
- an exact, clean-only posterior over corruption configurations from *cross-modal conditional* evidence (which
  catches swaps that unimodal clean-only detectors such as CARAT, Lee and WISER cannot);
- soft BMA over subset predictors;
- posterior-mixed per-mask conformal;
- calibration evaluation under silent corruption.

I estimate originality at about 2.5–3/5. That is unchanged from the review, but the reason has changed: CARAT and
WISER are new threats.

---

## 5. Pages I could not open (not used as evidence)

- The Iwata and Yamada NeurIPS 2016 PDF (server error). I used the arXiv 1411.3413 abstract.
- The Roberts et al. Oxford "Multi-Sensor Fault Recovery in the Presence of Known and Unknown Fault Types" (FUSION
  2009) PDF, blocked by robots.txt. Its title suggests a switching model with an unknown-fault (broad) class, which is
  CXF's kappa-broadened outlier block. Try to obtain and cite it.
- "Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities" (PMC13119957), behind a
  captcha.
- Eduardo et al., RVAE (AISTATS 2020): the re-fetch needed an approval nobody answered. It was opened in the earlier
  review (`proposal_review.md` §1.1(b)).
- SSRN 6987420, "Modality-Aware Reliability: Test-Time Reliability Gating…" (Raposo): fetch error.

---

## 6. Required changes to `proposal.tex`

### 6.1 Replace the "Deep reliability-aware fusion" paragraph and the novelty list (lines 73–90) with

> **Deep reliability-aware fusion.** An odd-one-out network weights leave-one-out fusions by the probability that each
> modality is perturbed~\cite{yang2021defending}, and quality heads are regressed to injected corruption
> severity~\cite{kumar2026prime,wu2026electronics}; all are \emph{trained on corruptions}. Reliability-weighted
> PoE learns a sigmoid weight from task labels alone, and its authors report that it does not track unseen
> corruption~\cite{palo2026reliability}. Conflict discounting~\cite{bezirganyan2025discounted}, generalized
> PoE~\cite{cao2014gpoe} and exact circuit-based credibility~\cite{tenali2026c2mf} re-weight beliefs by fixed or
> supervised rules.
> **Corruption-free detection.** Several methods need no corrupted training data. A PoE trained with modality dropout
> can drop an expert flagged by its own reconstruction error~\cite{lee2020detect}. Clean-only per-stream detectors can
> swap flagged streams for the missing-modality token~\cite{mohapatra2026carat}. World-model surprise can select a
> trusted sensor subset~\cite{zollicoffer2025wiser}. Cross-modal reconstruction can flag
> faults~\cite{jensen2026fault}, and clean-only multi-view densities can flag inconsistent views~\cite{wang2020hbmvad}.
> These detectors are either unimodal, so they cannot see a realistic but mismatched sense, or unlocalised, and all
> make hard decisions without calibrated predictive uncertainty.
> **Conformal.** Per-missingness-pattern (Mondrian) conformal calibration exists for multimodal
> regression~\cite{azizi2026conformal,mccf2026}, but it assumes the pattern is known.
>
> **What is new in CXF.**
> (1) An \emph{exact} posterior over all $2^M$ corruption configurations, computed from a \emph{cross-modal conditional}
> density of frozen JEPA beliefs that is fitted on clean, unlabelled data. Because the evidence is conditional, it
> catches swapped or mismatched senses that unimodal clean-only detectors miss.
> (2) \emph{Soft} corruption-as-missingness: Bayesian model averaging of the predictive densities of all trusted
> subsets, rather than a hard reject~\cite{lee2020detect,mohapatra2026carat}.
> (3) Per-pattern conformal intervals mixed by the posterior, so that intervals calibrated only on declared
> missingness are reused under inferred corruption. We measure coverage empirically and bound its loss by the
> posterior's mask error.
> (4) The empirical finding that fine-tuning erases cross-modal agreement, and an auditor/worker split that preserves it.
> (5) The first evaluation of multimodal \emph{regression} under silent corruption, including swaps, by coverage, CRPS
> and tracking ratio, on MOSEI and on an image-plus-two-captions COCO benchmark. Prior MOSEI reliability work reports
> classification calibration under injected corruption~\cite{wu2026electronics}.

### 6.2 Smaller fixes

- **Palo et al.** Remove "its authors list label-free reliability signals as an open problem". Their words are
  "learning dependable quality signals remains an open challenge", and their weight is learned from task labels, not
  corruptions.
- **Delta 1 ("system model is a JEPA").** Do not claim "learned latent world model" as new (WISER). Keep "JEPA" only
  with an ablation against a reconstruction-based witness, e.g. a VAE/PoE reconstruction error as in Lee et al.
- **CXF-C.** Do not imply a conformal guarantee under corruption. Write "conformal-calibrated per pattern;
  coverage under corruption measured empirically".
- **Risk "This is just fault detection".** Add "or just Lee et al. / CARAT with a soft posterior". The answer is the
  swap/mismatch results (G2/H2): unimodal clean-only detectors cannot detect swaps by construction.

### 6.3 Baselines that MUST be added

1. **CARAT-style / Lee-style hard missingness** (the most important).
   - A per-modality *marginal* Mahalanobis detector on auditor latents, thresholded at the 95th or 99th percentile of
     clean scores. If several modalities are flagged, choose the largest z-score, as in Lee.
   - Each flagged modality is set to missing in the PoE subset head, and its Mondrian interval is used.
   - This is the "marginal-only" ablation promoted to a named baseline. CXF must beat it on swaps (H2/G2) and on
     coverage (H4), or delta 1 collapses.
2. **Conditional hard-MAP.** CXF evidence with a hard MAP subset instead of BMA and mixed intervals. This isolates the
   value of the soft posterior (delta 2/3).
3. **WISER-style greedy min-surprise subset selection** using the same auditor. It is cheap: argmin over the 8
   subsets of a score.
4. **Reconstruction-witness variant.** The same posterior, but with cross-modal reconstruction error (Jensen/Lee
   style) as evidence instead of JEPA beliefs. It tests whether "JEPA" matters.
5. Keep Yang et al. (corruption-supervised, reported separately). Add the Wu et al. quality head as the in-domain
   supervised upper reference on MOSEI, trained on some corruption families and tested on others.

### 6.4 Bib entries to add

`lee2020detect` (arXiv 2012.00201), `mohapatra2026carat` (arXiv 2610.07499), `zollicoffer2025wiser`
(arXiv 2512.01119), `jensen2026fault` (arXiv 2609.16314), `wang2020hbmvad` (IJCAI 2020, pp. 2420–2426),
`tenali2026c2mf` (arXiv 2603.26629). The existing `mccf2026` key already points to Moayedikia, arXiv 2608.07183,
so it is fine. Fix the `palo2026reliability` title to the full title, "Reliability-Aware Probabilistic Fusion for
Multimodal Learning with Incomplete Evidence". Optional: `misic2026precision` (arXiv 2609.15065).
