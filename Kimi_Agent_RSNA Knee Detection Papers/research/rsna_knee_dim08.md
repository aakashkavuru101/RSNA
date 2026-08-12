# Dimension 08 — EVALUATION & VALIDATION STRATEGY
## RSNA Knee Abnormality Detection (Kaggle, 2026)

- **Research date:** 2026-08-10 (competition started 2026-07-30; ends 2026-10-22)
- **Researcher:** sub-agent dimension 08
- **Method:** 24 independent web searches (8 batched queries × 3) + direct browsing of the Kaggle competition pages (evaluation, data, efficiency, leaderboard, 6 discussion threads), RSNA press releases, arXiv papers, and past-RSNA-competition winner writeups. All key claims carry verbatim excerpts.
- **Citation format:** Claim / Source / URL / Date / Excerpt (verbatim) / Confidence

---

# SECTION 1 — THE COMPETITION'S EXACT EVALUATION METRIC (CONFIRMED)

## 1.1 Main metric: macro-averaged ROC AUC over 12 binary targets

**Claim:** The RSNA Knee Abnormality Detection 2026 main metric is the **macro-averaged (unweighted mean) per-class ROC AUC** across the 12 abnormality labels. Final Score = (1/12) Σᵢ AUCᵢ.
**Source:** Kaggle — RSNA Knee Abnormality Detection, Overview → Evaluation page (accessed directly 2026-08-10)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation
**Date:** accessed 2026-08-10 (competition live since 2026-07-30)
**Excerpt (verbatim):**
> "Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets: Final Score = 1/12 Σᵢ₌₀¹¹ AUCᵢ. The final score is, in other words, the macro-averaged AUC ROC."

The competition tags also list "Roc Auc Score".
**Confidence:** Confirmed (read directly from the live evaluation page).

## 1.2 Submission file format

**Claim:** One row per test study (StudyInstanceUID) with 12 confidence-score columns, in this exact column order.
**Source:** Kaggle evaluation page (same URL as above)
**Excerpt (verbatim):**
> "For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:
> `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`"
> (sample values all `0.5`)
**Confidence:** Confirmed.

## 1.3 Code / submission requirements

**Claim:** Notebook-only submissions; 9-hour CPU and 9-hour GPU limits; internet disabled; public external data allowed; file must be named `submission.csv`.
**Source:** Kaggle evaluation page → "Code Requirements" section
**Excerpt (verbatim):**
> "Submissions to this competition must be made through Notebooks. In order for the 'Submit' button to be active after a commit, the following conditions must be met: CPU Notebook <= 9 hours run-time / GPU Notebook <= 9 hours run-time / Internet access disabled / Freely & publicly available external data is allowed, including pre-trained models / Submission file must be named submission.csv"
**Confidence:** Confirmed.

## 1.4 Efficiency Prize metric (first-ever RSNA efficiency track)

**Claim:** A separate Efficiency Prize track ranks eligible submissions by a score combining private-test AUC and evaluation runtime (normalized by 32,400 s = 9 h, i.e. the notebook limit); objective is to **minimize**. Eligibility requires beating the `sample_submission.csv` benchmark on the Private Leaderboard.
**Source:** Kaggle — Overview → Efficiency Prize Evaluation
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation
**Excerpt (verbatim):**
> "For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance. To be eligible for an Efficiency Prize, a submission: Must be among the submissions selected by a team for the Leaderboard Prize… Must be ranked on the Private Leaderboard higher than the sample_submission.csv benchmark. … We compute a submission's efficiency score by: Efficiency = [KaTeX formula; page-rendering flattens to:] AUC Benchmark − max AUC + RuntimeSeconds/32400 where AUC is the submission's score on the main competition metric, Benchmark is the score of the benchmark sample_submission.csv, max AUC is the maximum AUC of all submissions on the Private Leaderboard, and RuntimeSeconds is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score."
> "During the training period of the competition, you may see a leaderboard for the public test data in the following notebook, updated daily: Efficiency Leaderboard [https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb]. … During the training period, this leaderboard will show only the rank of each team, but not the complete score."

**Interpretation caveat:** the KaTeX formula does not survive text extraction. Given the definitions and the "minimize" objective, the AUC term is almost certainly a normalized AUC shortfall, most plausibly `Efficiency = (maxAUC − AUC)/(maxAUC − BenchmarkAUC) + RuntimeSeconds/32400` (best submission → AUC term 0; benchmark-level → 1; runtime adds up to 1.0 at the 9-h cap). Practical implications are unaffected: **(a)** runtime only matters up to the 9-h cap and each saved hour is worth ~0.11 of the runtime term; **(b)** you must first clear the benchmark on the private LB; **(c)** heavy TTA/ensembles trade directly against the efficiency track — consider submitting one "accurate" and one "lean" notebook among your selected submissions.
**Confidence:** Formula structure — medium (rendering loss); definitions, eligibility, minimize-objective, daily public efficiency-rank notebook — Confirmed (verbatim).

## 1.5 Prizes & timeline

**Claim:** $77,000 total; main LB 1st–10th ($9,000/$7,000/$6,500/$6,000/$5,500/$5,000×5); efficiency 1st–3rd ($7,000/$6,000/$5,000). Entry/merger deadline 2026-10-15; final deadline 2026-10-22; winners' requirements 2026-11-05.
**Source:** Kaggle evaluation page — "Prizes" and "Timeline" sections (verbatim values transcribed from page)
**Excerpt (verbatim, timeline):**
> "July 30, 2026 - Start Date. October 15, 2026 - Entry Deadline. … October 15, 2026 - Team Merger Deadline. … October 22, 2026 - Final Submission Deadline. November 5, 2026 - Winners' Requirement Deadline."
**Confidence:** Confirmed.

## 1.6 Winners' obligations (affects strategy at the end)

**Claim:** Winners must open-source code AND weights, make a short video, and share the final model publicly.
**Source:** Kaggle evaluation page — "Prizes" section
**Excerpt (verbatim):**
> "…in addition to the standard Kaggle Winners' Obligations… the host team also asks that you: (i) create a short video presenting your approach and solution, and (ii) publish a link to your open sourced code and the weights on the competition forum (iii) Share final version of model as publicly available for open distribution and validation. Please see https://www.kaggle.com/models/tom99763/9th-place-models-rsna-iad/PyTorch/default as an example."
**Confidence:** Confirmed.

---

## 1.7 What the dataset page tells us about validation design

**Claim A:** Only a small subset of the 4,407 training studies carry per-condition labels; the rest must be labeled from reports. Test set ≈ 1,300 studies. Test rows contain StudyInstanceUID only (no report at inference). Series have 20–45 slices (median 30). DICOMs are stripped to an 86-tag allowlist.
**Source:** Kaggle — Data page
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Excerpt (verbatim):**
> "Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies."
> "There are about 1300 studies in the test set."
> "test.csv — Example test file with three study IDs from the public test set. During scoring, this example data will be replaced with the actual test data."
> "Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred."
> "Every DICOM has been stripped to an allowlisted set of 86 metadata tags."
**Confidence:** Confirmed.

**Claim B (critical for CV/LB correlation):** the organizers explicitly warn that prevalence differs between train, public, and private sets.
**Source:** Kaggle — Data page, "Dataset Distribution Notice"
**Excerpt (verbatim):**
> "Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets."
**Confidence:** Confirmed. **Implication:** per-class AUC on a ~1300-study test with shifted prevalence will be noisy for rare classes; expect public↔private divergence beyond the usual sampling noise (see §6).

## 1.8 Label semantics (annotation protocol) — matters for training targets and metric behavior

**Claim:** The 12 labels are exam-level binaries; borderline findings were graded negative (specificity-favoring); the expert reference set was double-read with adjudication.
**Source:** Kaggle discussion — pinned host thread "Knee Abnormality Detection AI Challenge Overview" (Po-Hao "Howard" Chen, Competition Host)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
**Date:** posted ~2026-08-06
**Excerpt (verbatim):**
> "Models are evaluated on twelve binary labels, each indicating the presence or absence of a specific finding in the imaged knee. … In each case, ambiguous or borderline findings ('on the fence') were graded as negative to favor specificity."
> "Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee."
**Confidence:** Confirmed.

## 1.9 Metrics used by the most similar past RSNA competitions (for calibration of expectations)

**Claim A (RSNA 2025 Intracranial Aneurysm Detection):** weighted multilabel AUC — "Aneurysm Present" weighted 13× vs 13 location labels; equivalent to ½·AUC_AneurysmPresent + ½·(mean of 13 location AUCs).
**Source:** Kaggle — RSNA Intracranial Aneurysm Detection evaluation page
**URL:** https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview/evaluation
**Excerpt (verbatim):**
> "Submissions are evaluated by a weighted multilabel [AUC ROC] between the predicted probabilities and the observed targets. For each of the fourteen target labels, we compute the [AUC ROC score]. The score for Aneurysm Present is assigned a weight of 13, while all the other scores are assigned a weight of 1. … Final Score = 1/2 (AUC_AP + 1/13 Σᵢ₌₀¹² AUCᵢ). You may find the metric code here: Mean Weighted Columnwise AUCROC."
**Confidence:** Confirmed. **Note:** knee 2026 drops the weighting — plain macro average; every one of the 12 classes contributes equally, so **rare classes (e.g. Fracture, MCL) are worth as much as common ones** — classic macro-AUC incentive to fix the worst class.

**Claim B (RSNA 2024 Lumbar Spine):** sample-weighted log loss (weights 1/2/4 for Normal-Mild/Moderate/Severe) + an any_severe_spinal component.
**Source:** Kaggle competition page + winner GitHub READMEs
**URL:** https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification ; https://github.com/DaoyuanLi2816/rsna-2024-lumbar-spine
**Date:** 2024
**Excerpt (verbatim):**
> "Submissions are evaluated using the average of sample weighted log losses. The sample weights are as follows: 1 for normal/mild. 2 for moderate [4 for severe]." (Kaggle page)
> "The submission evaluation uses the mean weighted log loss, Sample Weights: Normal/Mild: 1 Moderate: 2 Severe: 4" (GitHub README)
**Confidence:** Confirmed.

**Claim C (RSNA 2022 Cervical Spine):** weighted log loss with 7× penalty for patient-level missed fracture.
**Source:** Radiology: AI journal paper on the 2022 challenge winners (PMC)
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10831508/
**Excerpt (verbatim):**
> "Penalties were given for a missed fracture at each vertebral level (C1–C7), and a heavier penalty (seven times higher) was given if the algorithm classified a patient with one or more fractures as negative for fracture."
**Confidence:** Confirmed.

**Claim D (RSNA 2019 ICH / 2022 CSF / 2023 Abdominal Trauma):** weighted multi-label log loss family.
**Source:** MIT thesis (dspace.mit.edu) analyzing top RSNA Kaggle models
**URL:** https://dspace.mit.edu/bitstream/handle/1721.1/156974/sandadi-vsandadi-meng-bcs-2024-thesis.pdf
**Excerpt (verbatim):**
> "Models were evaluated using a weighted multi-label logarithmic loss. … The overall effect is such that each class is roughly equally important for the final score."
**Confidence:** Confirmed. **Trend note:** RSNA moved from weighted log loss (2019–2024) to AUC-family metrics (2025 aneurysm, 2026 knee). This changes the optimization target fundamentally: AUC is rank-based, so probability calibration and threshold tuning are largely irrelevant for the knee comp (see §2).

---

# SECTION 2 — MULTI-LABEL METRIC OPTIMIZATION FOR MACRO-AUC

## 2.1 AUC is a ranking metric — calibration does NOT change it; log loss DOES need calibration

**Claim:** Calibration (Platt/isotonic/beta/temperature) reorders nothing if monotonic, so AUC is preserved (isotonic exactly); calibration matters when the metric is log loss/Brier or when decisions use thresholds/costs. For this AUC comp, skip calibration; spend effort on ranking quality. (Had the metric been weighted log loss like RSNA 2024, calibration/temperature scaling and clipping would have been first-order important.)
**Source 1:** MetricGate blog — "Beta vs Platt vs Isotonic Calibration"
**URL:** https://metricgate.com/blogs/beta-calibration-vs-platt-vs-isotonic/
**Date:** 2026-03-11
**Excerpt (verbatim):**
> "Discrimination (AUC) measures whether the model can rank cases; calibration measures whether the numerical probabilities are trustworthy. The two are independent — a model can rank perfectly yet be wildly miscalibrated, and vice versa."
> "Recalibration with isotonic preserves AUC exactly; with Platt and beta it can change slightly. Audit both."
> "Calibrated probabilities are required whenever you combine model output with a cost or threshold: medical risk scores, fraud-decision policies, expected-value bets…"
**Confidence:** High (consistent with theory; monotonic maps preserve ROC).

**Source 2 (empirical demo):** TrainInData blog — "Probability Calibration in Machine Learning"
**URL:** https://www.blog.trainindata.com/probability-calibration-in-machine-learning/
**Date:** 2024-09-20
**Excerpt (verbatim):**
> "the model performance (assessed by the ROC-AUC) is similar regardless of probability calibration. However, the Brier score and log loss are noticeably smaller than those returned by the uncalibrated approach: Uncalibrated brier 0.177542, log_loss 0.539588, roc_auc 0.968339 / Platt Scaling 0.053869, 0.202653, 0.968106 / Isotonic Regression 0.051877, 0.188629, 0.967950"
> "A recommendable approach is calibrating the probabilities on a separate validation set (not seen in training) with the fitted model and then evaluating it using the test set."
**Confidence:** High.

**Claim (Platt vs isotonic selection rules):** Platt = 2-param sigmoid, good for small calibration sets; isotonic = non-parametric, needs ~500–1000+ points or overfits.
**Source:** MetricGate (same URL); corroborated by easydeeplearn Q&A (2026-08-01)
**Excerpt (verbatim, MetricGate):**
> "Use Platt if you have very little calibration data (under ~100 examples) and you have reason to believe the miscalibration is sigmoidal… Use isotonic if you have a lot of calibration data (thousands of points)… Isotonic overfits on small calibration sets and produces jagged probabilities."
**Confidence:** High. **Knee-specific note:** with only ~58 expert-labeled studies available locally, ANY fitted post-hoc mapping would be fitted on report-derived (weak) labels — do not fit post-hoc transforms on 58 studies; and for AUC there is no benefit anyway.

## 2.2 Log loss clipping (only relevant if you reuse past-RSNA log-loss intuitions)

**Claim:** Under log loss, overconfident deep nets are heavily penalized; Kaggle participants resort to "clipping" probabilities. Under AUC, clipping is a no-op (monotonic).
**Source:** arXiv — "Spline-Based Probability Calibration" (Lucena, 2018)
**URL:** https://arxiv.org/pdf/1809.07751
**Excerpt (verbatim):**
> "models which perform well in terms of accuracy may do poorly in terms of log-loss. This issue has arisen in Kaggle competitions, which increasingly use log-loss as the competitive metric for image classification problems. … many participants resort to the practice of 'clipping'. Simply put, clipping refers to assigning a minimum probability p_min to every class, and then re-normalizing the probability vector to sum to one."
**Confidence:** High.

## 2.3 Threshold tuning

**Claim:** Threshold tuning is irrelevant to the competition score (AUC integrates over all thresholds). It only matters for clinical-style operating-point analyses or efficiency-track narratives.
**Source:** arXiv 2402.02047v2 ("Quality and Trust in LLM-generated Code") — articulates the standard distinction
**URL:** https://arxiv.org/pdf/2402.02047v2.pdf
**Excerpt (verbatim):**
> "AUC is typically a favored metric as it is parameter free, meaning that one does not choose a threshold for when to label values as incorrect or correct from the score."
**Confidence:** High (definition-level fact).

## 2.4 Per-class view & class-wise ensembling under macro-AUC

**Claim:** In multi-label problems you can select/weight ensemble members **per class** by validation performance rather than globally — directly exploits the macro-mean structure of the metric.
**Source:** arXiv 2308.08853 (multi-label radiology report/label paper)
**URL:** https://arxiv.org/pdf/2308.08853
**Excerpt (verbatim):**
> "Class-Wise Ensemble. The multi-label nature allows us to train all categories simultaneously in order to leverage the dependency between classes, but to predict each category independently when testing. For each class, we select best models based on their performance on the development set and average their predictions… Note that, it is different from the ordinary way that selects the candidates for ensemble based on the model performance instead of each class performance."
**Confidence:** High. **Knee application:** track OOF AUC per class; the macro metric means your worst class (likely Fracture/MCL/Synovitis per weak-label recoverability — §4.3) drags the score as much as your best.

---

# SECTION 3 — CROSS-VALIDATION STRATEGY FOR THIS COMPETITION

## 3.1 Split at the patient/study level — series/planes of the same knee must never cross folds

**Claim:** Random image-level (or even series/study-level-with-repeat-patients) splits leak: adjacent slices and repeated studies from one patient are near-duplicates. The standard fix is patient-level (here: study-level, since labels are per-exam) grouping; institution-level splits can be even more conservative.
**Source:** PMC — "Mitigating Bias in Radiology Machine Learning: 1. Data…" (Radiology: AI)
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9533091/
**Excerpt (verbatim):**
> "Data may 'leak,' even if developers ensure that no data are repeated in both training and test sets because medical data are usually clustered at different levels. For example, a patient with a liver tumor may have four different liver MRI studies, each with more than one series, and each series with several images. In such a scenario, a random train-test split at the image level will result in biased training. … Even splitting based on series or study levels is not sufficient to prevent data leakage. … The standard way to prevent data leakage is to split medical data at the patient level. … if there are consistent differences in how a hospital handles patient scans, it may be valuable to separate data at the institution level."
**Confidence:** Confirmed (peer-reviewed). **Knee application:** group = StudyInstanceUID (a patient may rarely have 2 studies — check PatientID-like proxies via metadata); keep ALL series/slices of a study in one fold. Augmentation variants of one slice must also stay in-fold.

## 3.2 Multilabel stratification: iterative stratification (Sechidis/Szymański), combined with grouping

**Claim:** Plain StratifiedKFold cannot handle multilabel targets; the standard is iterative stratification (scikit-multilearn `IterativeStratification` / Kaggle's `MultilabelStratifiedKFold`), which preserves per-label (and label-combination) frequencies per fold. With 12 imbalanced labels this is essential so each fold has positives of rare classes.
**Source 1 (primary):** Szymański & Kajdanowicz, "A Network Perspective on Stratification of Multi-Label Data" (ECML-PKDD LSDA workshop 2017, PMLR v74)
**URL:** http://proceedings.mlr.press/v74/szymański17a/szymański17a.pdf
**Excerpt (verbatim):**
> "We present a new approach to stratifying multi-label data for classification purposes based on the iterative stratification approach proposed by Sechidis et. al. in an ECML PKDD 2011 paper. … The proposed approach lowers the variance of classification quality, improves label pair oriented measures and example distribution while maintaining a competitive quality in label-oriented measures."
**Source 2:** Sechidis, Tsoumakas, Vlahavas (2011), "On the stratification of multi-label data", ECML PKDD 2011, pp. 145–158 (as cited); tooling: scikit-multilearn `iterative_stratification` module (http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html)
**Source 3 (biomedical usage):** PMC — "Multi-label classification of biomedical data"
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11411592/
**Excerpt (verbatim):**
> "To carry out k-fold cross validation and account for the imbalanced multi-label dataset, high-order iterative stratification was implemented via the scikit-learn 'iterative_stratification' module. In brief, dataset splits are created while maintaining balanced representation of labels within each fold as much as possible."
**Confidence:** Confirmed. **Practical recipe for this comp:** stratify on the 12 weak labels (or on the 58-study gold subset separately); group by study; if using `iterative-stratification`'s `MultilabelStratifiedGroupKFold` is awkward, a common Kaggle pattern is greedy assignment of whole studies to folds balancing per-class counts. (A plain combination trick — encoding the 2^12 label-set as one class — is infeasible here due to combination sparsity; per-label iterative methods are the right tool.)

## 3.3 Site/scanner-aware splits — this dataset has 16 sites and ~13–265 scanner fingerprints

**Claim (competition-specific, measured):** Training DICOMs carry Manufacturer (Siemens/Philips/GE) and field strength (1.5T/3T) → ~13 scanner groups across 4,407 studies (Siemens_1.5T 1,148; Siemens_3T 781; GE_1.5T 698; Philips_3T 663; Philips_1.5T 619). A metadata-only classifier scores macro-AUC 0.652 under random 5-fold but only 0.598 under scanner-grouped folds — i.e. **~0.05 AUC of apparent skill is site memorization that does not transfer to unseen scanners**. OA targets show the largest drop (0.07–0.09).
**Source:** Kaggle discussion — "DICOM metadata findings: scanner-grouped CV and PatientSex priors" (morningduck)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004
**Date:** posted ~2026-08-09 (18h before access)
**Excerpt (verbatim):**
> "A standard random K-fold split places the same scanner in both train and val folds. To measure the effect, I trained a metadata-only classifier (no pixels) and evaluated it under both split types: Random 5-fold 0.652 / Scanner-grouped 0.598. The gap is ~0.05 from metadata alone. OA targets showed the largest drop (0.07–0.09), consistent with field-strength-dependent cartilage contrast differences."
> "PatientSex (tag 0010,0040) is present in the test set DICOM headers. Training distribution: M=2,076 / F=1,894. [Table:] ACL ~54% M / ~32% F; Medial OA ~12% M / ~45% F."
**Confidence:** High (community measurement, pre-registered thresholds claimed; consistent with known batch-effect literature). **Recommendation:** build TWO validation views — (a) study-grouped multilabel-stratified folds for model selection correlated with LB, and (b) a held-out-scanner-group (or held-out-site) fold to estimate generalization to the expert test set, since RSNA test sets are expert-annotated and the train labels here are report-derived (distribution shift is guaranteed; see §1.7 Claim B).

**Corroboration (finer fingerprints):** clustering on Manufacturer+Model+SoftwareVersions+ImagingFrequency+Coil gives 265 distinct scanner fingerprints, top-20 covering 45.5% of studies; metadata-only macro AUC 0.6515 random vs 0.5981 scanner-grouped.
**Source:** Kaggle discussion — "0.932 LB within one day. Tested for DICOM metadata shortcut" (Oleksii Zhukov)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517
**Date:** posted ~2026-08-07
**Excerpt (verbatim):**
> "DICOM metadata alone reaches 0.6515 macro AUC under random folds but only 0.598 across unseen scanners. … The 0.053 increment is site memorization and it does not transfer to unseen scanners."
> "265 fingerprints is finer-grained than institution. I'm separating individual scanners and software revisions within sites, so the grouped folds are stricter than true site holdout."
> "Targets are report-derived, not expert annotations. Only 58 studies carry per-condition labels, which is too few to fit or validate against. So part of the 0.053 may be metadata predicting reporting style rather than disease."
**Confidence:** High.

## 3.4 The 58 gold studies vs 4,349 weak-labeled studies — validation hierarchy

**Claim:** train.csv has 4,407 studies; only 58 carry the 12 expert per-condition labels; the other 4,349 must be supervised via report-derived weak labels.
**Source:** Kaggle discussion — "train.csv has 4,407 studies and 58 labels — the other 4,349 have reports" (maximo lorenzo y losada) and "58 labelled studies out of 4,407" (Luka Duvanov)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055
**Confidence:** Confirmed by multiple independent participants + data page wording ("Only a small subset of training studies carry per-condition labels").

**Claim (weak-label quality, measured):** LLM/keyword extraction of the 12 findings from reports reaches per-finding balanced accuracy 0.82 (Baker's) down to 0.56 (medial meniscus) vs the 58 gold studies; fracture extraction has 0.93 specificity but 0.44 sensitivity; extractor finds 2.6 findings/study vs annotators' 4.1 and returns nothing for 23% of studies.
**Source:** Kaggle discussion — "Weak labels for all 12 findings + how recoverable each one actually is" (Luka Duvanov)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734117
**Date:** posted 2026-08-10
**Excerpt (verbatim):**
> "Balanced accuracy ranges from 0.82 (Baker's cyst) to 0.56 (medial meniscus), and the ordering is not random — it tracks what kind of thing the finding is: Named objects (Baker's cyst, ACL) work… Graded severities (effusion) fail: 'minimal joint fluid', 'trace effusion' — a binary keyword has nowhere to put the adjective. … Unstated inferences (fracture) fail hardest: 0.93 specificity but 0.44 sensitivity, because more than half are described by appearance without the word ever appearing. … the extractor finds 2.6 findings per study where the annotators recorded 4.1, and returns nothing at all for 23% of studies. Treat those as unlabelled rather than as all-negative."
**Confidence:** High (measured on gold subset). **CV design implication:** (1) keep the 58 gold studies as a pristine validation anchor (or a tiny always-held-out set) to sanity-check that weak-label CV gains transfer to expert-label AUC; (2) treat report-silent studies as unlabeled (mask the loss) rather than negative; (3) expect label-noise-tolerant losses to matter (see §4).

## 3.5 What past RSNA winners actually used for CV

**Claim:** RSNA 2025 aneurysm 5th place used 5-fold CV for detectors, tracked OOF AUC per model, and selected ensemble weights on OOF + public LB; final ensemble of 6 models; single near-full-data models when time-constrained with "only 50 series for evaluation".
**Source:** Kaggle writeup — "5th place solution with code", RSNA Intracranial Aneurysm Detection (HoangHuyen)
**URL:** https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution
**Date:** 2025-10-15
**Excerpt (verbatim):**
> "vit large 384: OOF AUC = 0.8503 / eva large 384: OOF AUC = 0.8551 … Due to time constraints, I couldn't use 5 models (5 folds) for prediction. I could only use a single model trained on almost the full dataset, using only 50 series for evaluation, and selecting the best epoch based on the public leaderboard."
> "Final1: OOF AUC = 0.8823, public LB = 0.89 … [ensemble weights] try a few weight combinations to achieve the highest OOF AUC and the highest public LB"
**Confidence:** Confirmed (winner writeup). **Lesson:** OOF-based weight search is standard; note the risky practice of epoch selection on public LB (see §6 dangers).

**Claim:** RSNA 2024 lumbar silver medalist used 5-fold CV with a fold file shared as a dataset (`5folds.csv`); 9th place (adamnarai) selected best epoch by validation loss with `_best` checkpoints.
**Source:** GitHub READMEs
**URLs:** https://github.com/DaoyuanLi2816/rsna-2024-lumbar-spine ; https://github.com/adamnarai/kaggle-rsna-2024
**Excerpt (verbatim):**
> "Using 5-fold cross-validation, we train a separate YOLOv8 model for each degenerative disease pattern." / "Checkpoints with the `_best` postfix contain the weights for the best epoch (based on validation loss) and were used for all models."
**Confidence:** Confirmed.

---

# SECTION 4 — CLASS IMBALANCE HANDLING (12 labels, rare findings)

## 4.1 Asymmetric Loss (ASL) — the multi-label standard for positive-negative imbalance

**Claim:** ASL (Ridnik/Ben-Baruch et al., Alibaba DAMO, ICCV 2021) decouples focusing parameters for positives vs negatives, down-weights and hard-thresholds easy negatives; SOTA on MS-COCO etc.; no training-time overhead. Typical settings γ−=4, γ+=1, clip=0.05.
**Source:** arXiv 2009.14119 — "Asymmetric Loss For Multi-Label Classification"
**URL:** https://arxiv.org/pdf/2009.14119 (also https://github.com/Alibaba-MIIL/ASL)
**Excerpt (verbatim):**
> "In a typical multi-label setting, a picture contains on average few positive labels, and many negative ones. This positive-negative imbalance dominates the optimization process, and can lead to under-emphasizing gradients from positive labels during training, resulting in poor accuracy. In this paper, we introduce a novel asymmetry loss ('ASL'), which operates differently on positive and negative samples. The loss enables to dynamically down-weights and hard-thresholds easy negative samples, while also discarding possibly mislabeled samples. … With ASL, we reach state-of-the-art results on multiple popular multi-label datasets: MS-COCO, Pascal-VOC, NUS-WIDE and Open Images. … ASL is effective easy to implement, and does not increase the training time or complexity."
**Confidence:** Confirmed (primary source). **Knee fit:** 12 exam-level labels with many negatives per study + noisy report-derived labels — ASL's "discard possibly mislabeled negatives" property is directly relevant to the 23% report-silent studies problem (if not masked) and to fracture under-labeling.

**Corroborating benchmark:** in a 2026 long-tail multi-label study, "ASL dominates the multi-label long-tail family on every cell except ModernBERT at 1:1, where Distribution-Balanced is marginally ahead (within seed noise)." (arXiv 2605.24296v2, Appendix D; ASL configured γneg=4, γpos=1, clip=0.05)
**URL:** https://arxiv.org/html/2605.24296v2
**Confidence:** Medium-High.

## 4.2 Weighted BCE (pos_weight) and focal loss

**Claim:** pos_weight computed from neg/pos ratio per class is the standard first-line fix; focal loss (Lin et al. 2017) down-weights easy examples via (1−p_t)^γ (γ=2 ⇒ 100× loss reduction at p_t=0.9); α-balancing handles frequency.
**Source 1:** TUM lecture slides citing Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
**URL:** https://dvl.in.tum.de/slides/cv3dst-ws19/3.ObjectDetection2.pdf
**Excerpt (verbatim):**
> "When γ=0 it is equivalent to the cross-entropy loss. As γ goes towards 1, the easy examples are down-weighted. Example: γ=2, if p_t=0.9, FL is 100× lower than CE."
**Source 2 (comparative guidance):** abhik.ai focal-loss explainer
**URL:** https://www.abhik.ai/concepts/deep-learning/focal-loss
**Date:** 2025-01-31
**Excerpt (verbatim):**
> "For an easy example with pt = 0.9 and γ = 2, the factor equals (1 − 0.9)² = 0.01, reducing the loss by 100x. … Use Focal Loss when… Extreme class imbalance (1:1000+)… Easy negatives dominate gradient signal."
**Confidence:** High. **Caveat from label noise:** focal loss emphasizes hard examples — with noisy weak labels, hard examples are disproportionately mislabeled, so ASL (which clips negatives) or loss-masking is usually safer than pure focal here.

**Claim (medical-imaging usage precedent):** MRNet implementations used `pos_weights` on the loss for the (injury-skewed) knee dataset.
**Source:** GitHub — AlbertoUAH/Knee-Lesions-Classification-via-Deep-Learning
**URL:** https://github.com/AlbertoUAH/Knee-Lesions-Classification-via-Deep-Learning
**Excerpt (verbatim):** config list includes "`pos_weights` on Loss function"
**Confidence:** Medium (repo documentation).

## 4.3 Which knee classes will be rare/hard (planning priors)

- Weak-label recoverability (§3.4) suggests Fracture, Effusion, Synovitis, Medial Meniscus are the noisiest supervision; under macro-AUC these noisy classes are also where AUC is hardest to push.
- PatientSex priors (§3.3): ACL prevalence ~54% M vs ~32% F; Medial OA ~12% M vs ~45% F — sex is a legitimate, test-time-available prior (tag present in test headers), but beware it encodes site/population confounds.
- Oversampling multi-label: naive per-class oversampling conflicts across co-occurring labels; prefer loss-based weighting (ASL/weighted BCE) or multilabel-aware samplers; if oversampling, sample at the study level.

---

# SECTION 5 — ENSEMBLING, TTA, CHECKPOINT AVERAGING

## 5.1 TTA: benefits are real but NOT guaranteed — validate per model/dataset

**Claim (positive evidence):** TTA (augment test input, average predictions) is an established variance-reduction technique in medical imaging; it "helps to eliminate overconfident incorrect predictions" (Wang et al. formulation for MRI segmentation).
**Source:** PMC — "Test-time augmentation for deep learning-based cell segmentation" (Moshkov et al., Sci Rep 2020)
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7081314/
**Excerpt (verbatim):**
> "A theoretical formulation of test-time augmentation has recently been described by Wang et al. Their experiments show that TTA helps to eliminate overconfident incorrect predictions. … Our results indicate that on average TTA can provide higher segmentation accuracy compared to predicting based on the original images only… the high cost of TTA, related to the fact that multiple times more predictions are required for the same object, is also an issue to be considered."
**Confidence:** Confirmed (peer-reviewed).

**Claim (cautionary, 2026):** A systematic MedMNIST study found standard TTA pipelines consistently HURT medical image classification (up to −31.6 points for ResNet-18 on PathMNIST), driven by BN-statistics/distribution shift; intensity-only augs are safer; always include the identity view; never apply TTA by default.
**Source:** arXiv 2604.09697 — "I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image Classification"
**URL:** https://arxiv.org/html/2604.09697v1
**Date:** 2026-04-06
**Excerpt (verbatim):**
> "Our principal finding is that TTA with standard augmentation pipelines consistently degrades accuracy relative to single-pass inference, with drops as severe as 31.6 percentage points for ResNet-18 on pathology images. … Never apply TTA by default. Always validate on a held-out set first. Prefer intensity over geometric augmentations for models with BatchNorm on small images. Always include the unaugmented image as one of the TTA views…"
**Confidence:** High (directly on-point recent empirical study; note: 28×28 MedMNIST, so magnitude may not transfer to 256–384px MRI, but the "validate first" directive does).

**Claim (typical TTA set in medical classification):** identity + horizontal flip + small rotations (±5°), averaging softmax; gave +0.55% macro-AUC in a lung-CT study.
**Source:** Frontiers in Computer Science (2026-07-14) — hybrid CRNN lung CT paper
**URL:** https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1798168/full
**Excerpt (verbatim):**
> "test-time augmentation (TTA) applies a fixed set of label-preserving transformations—identity, horizontal flip, and rotations of ±5°—and averages the resulting softmax distributions. … TTA provides modest improvements: Enhanced CRNN accuracy increases from 0.4523 to 0.4557 (+0.34% absolute) and macro-AUC from 0.7302 to 0.7357 (+0.55% absolute)."
**Confidence:** High.

## 5.2 Which augmentations are "safe" for knee MRI — the medial/lateral trap

**Claim (cross-competition evidence):** The RSNA 2025 aneurysm 5th-place winner applied horizontal flip but **swapped left/right anatomy labels accordingly**, gaining ~0.01 AUC.
**Source:** Kaggle writeup (5th place, RSNA IAD 2025) — URL in §3.5
**Excerpt (verbatim):**
> "Horizontal Flip: It may sound unreasonable, but I applied horizontal flipping to the images and adjusted the labels as following: Left Infraclinoid Internal Carotid Artery <-> Right Infraclinoid Internal Carotid Artery [etc.] … This worked, and my model's accuracy improved by about 0.01"
**Confidence:** Confirmed. **Knee translation:** a left-right flip of a knee MRI swaps the medial and lateral compartments ⇒ labels "Medial Meniscus"↔"Lateral Meniscus" and "Medial OA"↔"Lateral OA" must be swapped on flip (both in training augmentation and TTA with label-aware un-flipping); ACL/MCL/Effusion/Synovitis/Baker's/Contusion/Fracture are flip-invariant at the exam level. Rotations: small angles only; elastic/intensity augs are safe; do NOT use vertical flips. Because the DICOM orientation is standardized, verify flip direction against ImageOrientationPatient before trusting it.

## 5.3 Aggregation across slices/series (volumetric TTA analog)

**Claim:** Winners of volumetric RSNA comps aggregate slice-level predictions per series (max over z) and per study (max/mean across series); the aneurysm 5th place used max over slices; lumbar medalists used max across condition-level groupings.
**Source:** Kaggle writeup (5th place, RSNA IAD 2025); GitHub DaoyuanLi2816 (RSNA 2024)
**Excerpt (verbatim):**
> "the series prediction is the maximum in z-axis (slice): num_slices×14 -> 14 probs" (RSNA IAD 5th place, comment)
> "The infer function aggregates confidence scores across the condition_level_severity categories, selecting the maximum score for each category to form the final predictions." (RSNA 2024 lumbar)
**Confidence:** Confirmed. **Knee note:** MRNet (Stanford knee MRI) instead combined 3 plane-level models with a per-task logistic regression stacker — a strong template for this comp's multi-series studies:
> "Given predictions from the sagittal T2, coronal T1, and axial PD MRNets on the training set, along with their corresponding original labels, we trained a logistic regression to weight the predictions from the 3 series and generate a single output for each exam" — PMC6258509, https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/ (Bien et al., PLOS Medicine 2018). MRNet achieved AUCs "0.937 (95% CI 0.895, 0.980), 0.965 (95% CI 0.938, 0.993), and 0.847 (95% CI 0.780, 0.914) for abnormality detection, ACL tear detection, and meniscal tear detection respectively" — a rough ceiling reference; external-site transfer dropped ACL AUC to 0.824 without retraining ("We validated MRNet on a dataset from a different institution… achieved an AUC of 0.824 (95% CI 0.757, 0.892)… with no additional training"), direct evidence for site-shift risk (§3.3).

## 5.4 Snapshot ensembles, SWA, EMA

**Claim:** Snapshot ensembling (save weights at LR-cycle minima, ensemble predictions) gives ensemble diversity for one training run; SWA averages late-training weights under constant/cyclic LR for flatter minima and better generalization (Izmailov et al. 2018); BN statistics must be recomputed after averaging; EMA is the continuous-decay variant.
**Source 1 (primary):** Izmailov, Podoprikhin, Garipov, Vetrov, Wilson — "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018 (arXiv:1803.05407)
**Excerpt (verbatim, via citing paper arXiv 2411.18704):**
> "SWA keeps a uniform average of checkpoints during the final epochs of an SGD trajectory, while holding a reasonably high and constant learning rate. SWA is argued to find flatter solutions than SGD, thus generalizing better to unseen data."
**Source 2:** Davies Meyer glossary (2026-02-12)
**URL:** https://ai-solutions.daviesmeyer.com/en/glossary/swa
**Excerpt (verbatim):**
> "Batch normalization must be recomputed after averaging. Not always effective on already optimally tuned models. … SWA averages discrete checkpoints equally weighted; EMA averages continuously with exponential decay."
**Source 3 (snapshot ensembles):** rohan-paul.com ML interview series
**URL:** https://www.rohan-paul.com/p/ml-interview-q-series-how-can-we-d8a
**Excerpt (verbatim):**
> "Snapshot Ensembles: During training, save model weights at different times (snapshots) using a cyclical or changing learning rate. Each snapshot can be viewed as a distinct model, and these models can be ensembled at inference."
**Confidence:** High. **Efficiency-track note:** SWA/EMA cost ~nothing at inference (single model) — ideal for the efficiency prize; snapshot/model ensembles multiply runtime — use for the main track only.

## 5.5 Ensemble weighting

**Claim:** Tune ensemble weights on OOF (optionally per class — §2.4); simple uniform averaging of folds is a robust default; the aneurysm winner grid-searched weights on OOF AUC + public LB (quote in §3.5).
**Confidence:** High (winner practice). Also: heterogeneous-architecture ensembles trade top-1 accuracy for higher macro-AUC — "heterogeneous ensembles can improve threshold-independent metrics such as AUC, yet this smoothing may blur class boundaries and degrade argmax-based accuracy" (Frontiers lung-CT paper, URL §5.1) — for an AUC comp that trade is pure upside.

---

# SECTION 6 — KAGGLE-SPECIFIC STRATEGY

## 6.1 Public/private split & shake-up risk for THIS comp

**Claim:** The knee public LB is computed on ~30% of the test data (~390 of ~1,300 studies); final standings use the other 70%. As of 2026-08-10, top public score is 0.942 and the top ~50 teams are all ≥0.900 — a compressed leaderboard where tiny private-set noise reorders ranks.
**Source:** Kaggle leaderboard page (accessed 2026-08-10)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard
**Excerpt (verbatim):**
> "This leaderboard is calculated with approximately 30% of the test data. The final results will be based on the other 70%, so the final standings may be different."
> (table) "1 Brandon Low 0.942 … 10 kaggle & graduate trade-off 0.925 … 50 [—] 0.900"
**Confidence:** Confirmed (live page). **Shake-up assessment:** small expert-annotated private set + prevalence-not-guaranteed notice (§1.7B) + macro-AUC on 12 classes with some rare classes ⇒ per-class AUC on ~900 private studies has wide CIs for rare classes (if a class has ~5% prevalence, ~45 positives; AUC SE is roughly ±0.03–0.05 per class, and macro-AUC averages 12 such noisy terms). Expect meaningful rank movement; robustness (scanner-grouped validation, no LB-fit hacks) is the defense.

**Claim (general shake-up framing):** "The difference between Public and Private leaderboards standings is called a 'shake-up'… The winner of the competition had only 1485th place on the Public Leaderboard." (extreme example)
**Source:** DataCamp — "Winning a Kaggle Competition in Python" course
**URL:** https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/kaggle-competitions-process?ex=8
**Excerpt (verbatim):**
> "Usually, competitions have a limit of about 5 submissions available per day. … If we only look at the results on the Public Leaderboard, we could potentially overfit to it. Thus, our Private Leaderboard score will be considerably worse together with our final place in the competition. To beat the overfitting in both real-life projects and competitions, we need to use a good validation strategy."
**Confidence:** High.

**Claim (academic estimate of safe test sizes):** ~10,000 test examples is a reasonable minimum against adaptive LB overfitting — this comp's ~1,300-study test is an order of magnitude smaller.
**Source:** arXiv 2407.02112 — "A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data" (2024-10-15)
**URL:** https://arxiv.org/html/2407.02112
**Excerpt (verbatim):**
> "Roelofs et al. [61] found that at least 10,000 test examples is a reasonable minimum test set size to protect against adaptive overfitting in Kaggle challenges."
**Confidence:** High. Note this comp partially mitigates via notebook-only hidden-test re-run (no direct CSV probing of labels), but LB feedback on 5 submissions/day still leaks bits.

**Counterpoint anecdote:** in NeurIPS Open Polymer 2025, a competitor switched final submission away from their best public model fearing LB overfit and lost — "Strong cross-validation and consistent public performance are usually the best predictors of private performance."
**Source:** GitHub — Gaurav-Kushwaha-1225/NeurIPS-Open-Polymer-Prediction-2025
**URL:** https://github.com/Gaurav-Kushwaha-1225/NeurIPS-Open-Polymer-Prediction-2025
**Date:** 2025-09-16
**Confidence:** Medium (single anecdote, but a common post-mortem pattern).

## 6.2 LB probing & shortcut hunting in THIS comp (observed)

**Claim:** Public LB passed 0.9 within a day of launch; a participant tested a DICOM-metadata shortcut and found metadata alone gives ≤0.65 macro AUC — "Leaderboard scores seem to reflect image reading." (and/or report-text-derived supervision leaking strong priors).
**Source:** Kaggle discussion 733517 (Oleksii Zhukov) — URL §3.3
**Excerpt (verbatim):**
> "Public LB scores passed 0.8 and 0.9 within about a day of launch. That seemed strangely high for this label set in such short period, so I tested whether DICOM metadata alone could account for this. It can't. Full header metadata, no pixels, reaches 0.6515 macro AUC under random folds and 0.5981 under scanner-grouped folds."
**Confidence:** Confirmed (thread + numbers). **Reading:** the 0.93+ LB is achievable because the public test presumably carries the same report-derived weak-label statistics as train for whatever supervision signal leaks through site/protocol demographics — so expect the private expert-annotated set to be harsher. Don't chase public-LB decimals; anchor on (a) grouped multilabel CV, (b) the 58 gold studies, (c) scanner-held-out folds.

## 6.3 Notebook constraints & offline tricks

**Claim:** 9-h CPU/GPU limit, internet off (§1.3). Standard trick for non-preinstalled libraries: download wheels locally, upload as a Kaggle Dataset, attach to the notebook, and `pip install --no-index --find-links=...`.
**Source:** zenn.dev article (kito2718), 2026-06-28
**URL:** https://github.com/kito2718/zenn_articles/blob/main/articles/zenn_20260714_0630_kaggle-offline-install.md
**Excerpt (verbatim):**
> "!pip install --no-index --find-links=/kaggle/input/datasets/aaaa1597/zarr-offline-installation-wheels/zarr_wheels_fixed zarr … インターネットオフの制約があるコンペでも、この方法であれば、どんなライブラリでも自由に使用できます。" ["Even in competitions with the internet-off constraint, this method lets you freely use any library."]
**Confidence:** Confirmed (standard, widely used technique). Same pattern applies to pretrained weights (mount as Kaggle Models/Datasets — explicitly allowed: "Freely & publicly available external data is allowed, including pre-trained models").

**Efficiency implications (verbatim definitions in §1.4):** RuntimeSeconds = full evaluation time on the hidden re-run; with a 569.76 GB dataset and 9-h cap, I/O and DICOM decoding dominate — efficient track favors: 2D/2.5D models over 3D, cached/pre-decoded arrays where legal, mixed precision, few series per study (plane selection via train_series.csv metadata, which IS provided for test), single-model SWA instead of deep ensembles, minimal TTA.

## 6.4 Submission mechanics observed in past RSNA comps

**Claim:** Past RSNA comps required per-row predictions for every test case; missing rows error out; for RSNA 2025 IAD submission was via an evaluation API (rerun notebook), same pattern as knee 2026 (hidden test swap-in: "this example data will be replaced with the actual test data").
**Source:** Kaggle pages; GitHub yavorska-iryna/Kaggle-challenge-2024
**URL:** https://github.com/yavorska-iryna/Kaggle-challenge-2024
**Excerpt (verbatim):**
> "Missing vertebrae predictions must still be made to avoid errors but won't be scored."
**Confidence:** High. **Knee action items:** (1) ensure the notebook writes submission.csv with exactly the 13 columns in §1.2 order for every test StudyInstanceUID; (2) make inference robust to unseen transfer syntaxes ("uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian" — data page); (3) handle variable series counts/slice counts (20–45, tail to hundreds) without OOM; (4) fail-safe to 0.5 (the benchmark value) per study on any per-study exception.

## 6.5 Rules watch-items (as of 2026-08-10, unresolved/important)

- **Reports at inference:** test.csv has only StudyInstanceUID (no Report column) — a participant asked the host to confirm reports are unavailable for the hidden test; unanswered at access time. Design for image-only inference; use reports for training-time supervision (weak labels / report distillation / multimodal pretraining). Source: discussion 734118 (verbatim: "Since train.csv includes a Report column but test.csv only includes StudyInstanceUID, why is the competition considered multimodal? Is text intended only for training, while inference must rely solely on MRI images and series metadata?"). Confidence: test format Confirmed via data page; host intent — pending.
- **External data:** "Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels from the reports" (discussion 733652) and "Use of Commercially Hosted LLMs" (host thread 733965) — external knee datasets (e.g., MRNet) appear allowed if freely/publicly available, but LLM-API labeling of competition reports was under rules clarification (MIRA Section 6 thread 734131). Verify before building a pipeline on LLM-derived weak labels at scale.
- **Metadata in test headers:** PatientSex, Manufacturer, MagneticFieldStrength, SeriesDescription, ImageOrientationPatient are all readable at inference (morningduck thread, §3.3) — legitimate test-time features/priors, but site-correlated (shortcut risk vs expert test set).

---

# SECTION 7 — SYNTHESIZED RECOMMENDATIONS (dimension-08 deliverable)

1. **Optimize ranking, not probabilities.** Metric = unweighted macro-AUC over 12 exam-level binaries (confirmed). No calibration, no thresholding, no clipping needed; per-class monotonic transforms are no-ops. Spend post-processing budget on per-class ensemble selection/weighting fitted on OOF.
2. **CV = study-grouped + multilabel iterative stratification; add a scanner/site-held-out view.** Random folds overestimate by ~0.05 AUC from site memorization alone (measured on this dataset). Group by StudyInstanceUID (all series/slices/augmentations of a study in one fold); stratify the 12 labels iteratively (scikit-multilearn); keep one validation variant grouped by Manufacturer×FieldStrength (or finer scanner fingerprint) to model the 16-institution expert test shift.
3. **Exploit the 58 gold studies as the only honest local anchor.** Weak labels are noisy (fracture sensitivity 0.44; 23% of studies report-silent). Hold the gold studies out of training (or use as final validation); check that CV improvements on weak labels also improve gold-study macro-AUC.
4. **Handle label noise and imbalance with ASL (γ−=4, γ+=1, clip 0.05) or pos_weight-BCE + loss masking on unlabeled studies**; treat report-silent findings as missing, not negative; prefer loss-based fixes over naive multilabel oversampling.
5. **TTA: small, validated set only.** Identity + hflip (with medial/lateral label swap!) + ±5° rotation is the defensible maximum; validate per model (2026 evidence shows TTA can hurt classification); skip TTA entirely for the efficiency-track submission.
6. **Use SWA/EMA for the efficiency track; fold-ensembles for the main track.** SWA gives free generalization at zero inference cost; the efficiency score charges RuntimeSeconds/32400 against you.
7. **Slice→series→study aggregation:** max-pool over slices (winner precedent), and either max/mean or a small logistic stacker over series per plane (MRNet template); plane/sequence metadata in test_series.csv allows plane-routed inference.
8. **Defend against shake-up:** private set ≈ 70% of ~1,300 expert-annotated studies with prevalence shift; choose final submissions by robust CV + gold-study performance, not public-LB decimals; prefer two diverse final submissions (max-ensemble accurate one + lean efficient one) to hedge both tracks.
9. **Submission engineering:** exact 13-column format; robust DICOM decoding across 4 transfer syntaxes; per-study exception fallback to 0.5; dry-run the notebook end-to-end within 9 h on the example test set.
10. **Track the open rules threads** (reports-at-test confirmation, LLM-API labeling legality, external datasets like MRNet/KneeCoT) before committing the pipeline; document external-data sources for the winners' open-license obligations.

---

# APPENDIX — SEARCH LOG (24 queries, 8 batches)

1. "RSNA Knee Abnormality Detection Kaggle competition 2026 evaluation metric" / "kaggle.com/competitions/rsna-knee-abnormality-detection overview evaluation" / "RSNA 2026 knee MRI abnormality detection challenge Kaggle"
2. "RSNA 2025 Intracranial Aneurysm Detection Kaggle evaluation metric AUC" / "RSNA 2024 lumbar spine degenerative classification weighted log loss severity weights evaluation" / "macro averaged AUC multilabel classification Kaggle metric optimization"
3. "iterative stratification multilabel cross-validation scikit-multilearn medical imaging" / "patient-level train test split data leakage medical imaging deep learning" / "asymmetric loss multi-label classification ASL paper Ridnik"
4. "probability calibration Platt scaling isotonic regression when it matters log loss vs AUC ranking metric" / "Kaggle public leaderboard shake-up small test set overfitting private leaderboard" / "test-time augmentation MRI deep learning horizontal flip safe augmentation medical imaging" (0 results → rephrased in batch 6)
5. "stochastic weight averaging SWA EMA checkpoint averaging deep learning competition" / "snapshot ensembles cyclic learning rate Kaggle image classification" / "MRNet knee MRI Stanford dataset ACL tear abnormality AUC deep learning"
6. "test-time augmentation improves medical image classification deep learning TTA averaging predictions" / "Kaggle notebook offline pip install wheel files dataset no internet trick" / "focal loss class imbalance medical image classification weighted binary cross entropy pos_weight"
7. "MultilabelStratifiedGroupKFold iterative stratification group k-fold multilabel Kaggle" / "focal loss Lin 2017 dense object detection down-weight easy examples gamma quote" / "RSNA Kaggle competition leaderboard shake up small test set medical imaging winners"
8. "Izmailov 2018 'Averaging Weights Leads to Wider Optima and Better Generalization' abstract stochastic weight averaging" / "RSNA 2024 lumbar spine Kaggle 1st place solution cross-validation strategy patient split writeup" (0 results → rephrased batch 8b) / "AUC estimate variance small test set rare class medical imaging challenge leaderboard reliability" (0 results)
8b. "RSNA 2024 lumbar spine degenerative classification winner solution discussion validation folds" / "Sechidis 2011 stratification of multi-label data iterative stratification quote" / "Kaggle grandmaster advice trust your local validation cross-validation leaderboard medical imaging competition" (0 results)
Plus direct page fetches: Kaggle knee evaluation/data/efficiency/leaderboard pages; discussions 733343, 733375, 733517, 733753, 734004, 734055, 734117, 734118; RSNA IAD evaluation page; RSNA IAD 5th-place writeup; efficiency-LB notebook shell.

**Known gaps / uncertainties:**
- Efficiency-score KaTeX formula does not survive page extraction; reconstructed interpretation flagged in §1.4 (medium confidence on exact fraction; all practical implications high confidence).
- Host had not yet answered (as of 2026-08-10): reports-at-inference confirmation, LLM-API labeling rules, efficiency hardware details.
- The 58-gold-study count is participant-reported (multiple consistent reports + data-page wording); treat as near-certain.
