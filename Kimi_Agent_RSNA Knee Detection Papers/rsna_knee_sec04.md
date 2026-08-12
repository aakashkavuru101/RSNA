# 4. Modeling and Validation Strategy

## 4.1 The Real Problem: Weak Supervision, Not Architecture

Strip the competition down to its supervisory structure and an uncomfortable fact emerges: of the 4,407 training studies, only 58 (~1.3%) carry the 12 expert image-derived labels; the remaining 4,349 have only free-text radiology reports, and `test.csv` contains no report column at all — reports are a train-time-only teacher, confirmed by the host and by the data page wording ("you may wish to derive the labels for the remaining studies").[^1^][^2^] The model that scores must read pixels alone. This competition is therefore not primarily an image-modeling problem; it is a weak-label learning problem wearing an image competition's clothes.

The numbers make the asymmetry explicit. Measured against the 58 gold studies, LLM-extracted report labels reach 0.8780 macro AUC versus 0.8136 for regex/lexicon extraction — a +0.064 gap from label quality alone.[^3^] By comparison, the entire architectural spread among strong, well-tuned knee-MRI methods documented in the literature and in past RSNA winning solutions is roughly 0.01–0.03 AUC (Section 4.3). Soft/graded targets reportedly add a further +0.056 macro AUC over hard binary targets, and a single targeted imputation of "not addressed" synovitis cells lifted that column's AUC from 0.678 to 0.790.[^3^][^4^] Label engineering is the largest controllable lever in the competition, by an order of magnitude.

Two structural properties of the silver labels must be internalized before any modeling decision. First, reports are silver, not gold: strict textual reading agrees with the provided image-derived labels in only ~82.5% of cells (positive recall 80%), and the host confirmed that image-derived labels — produced by two subspecialty MSK radiologists with a third adjudicator, with borderline findings graded negative — are authoritative where the two disagree.[^2^] Second, 25.4% of all report-label cells are "not addressed," and silence is not symmetric across findings: when a report is silent on Baker's cyst, the finding is present only ~3% of the time (silence ≈ negative), whereas silent synovitis (84% of cells) still carries a ~34% positivity rate (silence ≈ uninformative).[^3^] Coercing all silent cells to 0 is the single most damaging label-pipeline mistake available.

**Proven:** label quality dominates architecture as a score lever; silence semantics are per-finding. **Promising but unverified:** the +0.056 soft-target gain is a community-reported figure from a thread title, plausible but single-source.

## 4.2 Building Silver Labels from Multilingual Reports

Reports arrive in roughly ten languages (EN/ES/NL dominant; FR/DE/PT/IT/TR tail; BG/EL observed) across 16 sites.[^5^] The host has officially ruled that commercial LLM APIs may be used for report label extraction during development, and open-weight models run locally are unconstrained since the internet-off rule applies only to the submission notebook.[^6^] The recommended pipeline, synthesized from community measurements on the 58 gold studies and the multilingual clinical-NLP literature:

| Stage | Component | Evidence / expected quality |
|---|---|---|
| 0. Language ID | fastText `lid.176` per report; per-site priors are strong | Standard; ~10 languages attested[^5^] |
| 1. Section split | Per-language regex for Impression/Conclusión/Beurteilung/Conclusie; weight Impression mentions higher | CheXpert-on-Impression practice[^7^] |
| 2. LLM extraction (primary) | Open-weight multilingual instruct LLM (Qwen3-14B/32B, 119 languages; Gemma-3-27B, 140+; GPT-OSS-20B single-GPU), few-shot strict JSON: one field per finding, 5-way value {present, absent, uncertain, not_addressed, laterality-ambiguous} + evidence span; official positivity thresholds in the system prompt | 0.8780 macro AUC vs gold (measured); anatomy-aware prompting adds up to +0.08 macro-F1 in radiology extraction[^3^][^8^][^9^] |
| 3. "Not addressed" channel | Map `not_addressed` → soft label 0.5, never to 0; impute only per-finding-justified cells (effusion→synovitis: 0.678→0.790 column AUC); no blanket imputation (measured worse: 0.8805 vs 0.8873) | Community-measured on gold[^3^] |
| 4. Rule-labeler fallback | Multilingual CheXpert-style labeler: per-language positive/negative/uncertain phrase files, NegEx-style triggers, OA-consequence vocabulary (osteophytes, joint-space narrowing, chondral loss, "tricompartmental") | 0.727 macro AUC alone (0.638 naive → 0.667 +negation → 0.727 +OA vocabulary); adds precision on fracture/Baker's; deterministic audit trail[^10^] |
| 5. Distillation | Fine-tune XLM-R student on LLM soft labels for a scalable, consistent labeler; multi-LLM voting as a second-order upgrade | Fine-tuned BERT on rule labels adds +3–10 F1; XLM-R covers 100 languages[^11^][^12^] |
| 6. Calibration & audit | Validate every label-pipeline variant exclusively against the 58 gold; treat differences below ±0.02 macro AUC as unmeasurable noise at n=58 | Community-measured noise floor[^3^] |

Translate-to-English (NLLB-200) is a third voter at most: its own model card warns against medical-domain use, and GPT-4o-level translation still loses ~16–21% factual fidelity — acceptable for cross-checking, not as the primary channel.[^13^]

The output of this pipeline is a 4,407 × 12 matrix of soft silver labels in [0,1] plus per-cell confidence. Downstream image training consumes these with confidence-weighted losses, and the 58 gold studies never enter training — they are the only local labels drawn from the same protocol as the hidden test set.

## 4.3 The Architecture Recipe That Keeps Winning

Architecture choice is the *second* problem, and it is largely solved. Three independent lines of evidence — the knee-MRI literature (MRNet → ELNet → CoPAS), four years of RSNA volumetric-competition winners (2022–2025), and MIL theory — converge on the same skeleton: per-plane 2D/2.5D pretrained CNN slices → a slice-sequence aggregator (BiLSTM or gated attention-MIL) → study-level binary heads, optionally preceded by a localize-then-classify stage.[^14^][^15^][^16^]

| Stage | Recommended design | Evidence |
|---|---|---|
| Input handling | Per-series; sort slices by DICOM metadata; equidistant subsample or cardinality-invariant pooling; 2.5D stacks (adjacent slices as 3–6 channels); per-volume z-score or Nyúl intensity standardization | 2022/2023 RSNA 1st-place recipes; ELNet normalization ablation[^15^][^16^][^17^] |
| Slice encoder | ImageNet-pretrained ConvNeXt-small / EfficientNetV2-s / CoAtNet-class 2D CNN; small > large in this regime ("convnext-large < base < small") | 2024 lumbar 1st place; Transfusion (ImageNet benefit is mostly low-layer scaling + convergence speed)[^17^][^18^] |
| Slice aggregation | BiLSTM or gated attention-MIL over per-slice features; aux attention loss | Attention-MIL lifted the 2024 lumbar winner's public LB 0.3729→0.3588 (+0.020 private); max/mean MIL pooling trains unstably[^17^][^19^] |
| Heads | 12 independent binary heads (treat labels as independent tasks; helps under severe per-class imbalance) | 2025 aneurysm 1st place treated 14 labels as independent binaries[^20^] |
| Plane/sequence fusion | Per-plane models + logistic-regression stacking (MRNet template, proven) or CoPAS-style cross-plane/cross-sequence attention (best-published 12-class knee design; public code at github.com/zqiuak/CoPAS is a directly transplantable template); avoid naive concatenation | MRNet AUC 0.937/0.965/0.847; CoPAS 0.812 avg AUC internal, 0.72 external[^14^][^16^] |
| Optional booster | Localize-then-classify ROI stage (meniscus/joint-crop from a small U-Net or keypoint model trained on a tiny annotation set) | Reliable +0.01–0.03 across four consecutive RSNA volumetric competitions; 87 masks sufficed in 2022[^15^][^17^] |

Equally valuable is the negative knowledge — approaches that have repeatedly failed in this exact setting and its closest analogs:

| Approach | Where it failed | Evidence |
|---|---|---|
| End-to-end 3D CNN classification | RSNA 2022 1st place: "this method does not work"; memory forces small backbones | 2022 1st-place write-up[^15^] |
| Large ViTs / large backbones | 2024 lumbar 1st: conv nets > ViTs; convnext-large < base < small; ViT underperforms CNNs on MRNet-scale data | 2024 1st write-up; MRNet systematic study[^17^][^21^] |
| Isotropic resampling to 1 mm³ | RSNA 2025 MIC-DKFZ: "substantially worsened" results, discontinued early | 2025 7th-place write-up[^22^] |
| External-data pseudo-labeling / co-training | 2024 NVSpine: "pseudo labelling on external data" did not help; 2025 MIC-DKFZ: co-training "did not really help" | 2024 6th / 2025 7th write-ups[^22^][^23^] |
| Weighted BCE from scratch | 2023 Team Oxygen: "did not help model converge" under balanced sampling | 2023 1st-place write-up[^24^] |
| Image-level ComBat harmonization | No benefit for DL classifiers under scanner shift (GE/Philips/Siemens study) | Sci Rep 2023[^25^] |

The practical instruction for a solo practitioner: clone the skeleton (CoPAS's public code is the closest legal template), invest the saved architecture-exploration budget in the label pipeline (Section 4.2), and add the localization stage only after the baseline is validated.

## 4.4 Validation You Can Trust

The evaluation layer is deliberately adversarial to naive validation. The public leaderboard is computed on ~30% of ~1,300 test studies; the organizers explicitly warn that abnormality prevalence "is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets"; and the 58 gold studies are pathology-enriched (every study has ≥1 positive finding, mean 4.14 findings/study), so even the gold anchor is prevalence-biased.[^1^][^3^][^26^] Meanwhile, a metadata-only classifier scores 0.652 macro AUC under random folds but 0.598 under scanner-grouped folds — ~0.05 AUC of apparent skill on this dataset is site memorization that will not transfer (largest on OA targets, 0.07–0.09).[^27^] Random-fold CV is therefore optimistically biased by construction.

| Design element | Specification | Rationale / evidence |
|---|---|---|
| Primary split | Study-level GroupKFold (5 folds): all series/slices/augmentations of one StudyInstanceUID in one fold | Patient-level splitting is the leakage standard in radiology ML[^28^] |
| Stratification | Iterative multilabel stratification on the 12 silver labels (scikit-multilearn), combined with grouping (greedy whole-study assignment balancing per-class counts) | Preserves rare-class positives per fold; plain StratifiedKFold cannot handle multilabel[^29^] |
| Secondary view | One scanner-grouped validation (group by Manufacturer × FieldStrength, or finer fingerprint) to estimate site shift | ~0.05 AUC memorization measured on this dataset; MRNet dropped 0.911→0.824 zero-shot cross-site[^14^][^27^] |
| Gold anchor | 58 expert-labeled studies held out of training entirely; every modeling decision (label extractor, imputation, augmentation, loss, architecture) must move gold macro AUC, not just silver CV | Only local labels from the test-time protocol; ±0.02 macro AUC is the noise floor at n=58[^2^][^3^] |
| LB policy | Trust CV + gold over public LB; use ≤1–2 of 5 daily submissions for hypothesis testing; expect shake-up (rare-class AUC SE on ~900 private studies is ±0.03–0.05 per class) | NVSpine 2024: "we trusted our CV more than public LB" — final CV 0.382 / public 0.355 / private 0.401[^23^] |

Two disciplines follow. First, tune hyperparameters on one fold when compute-limited (MIC-DKFZ practice), then confirm on the full CV.[^22^] Second, record per-class OOF AUCs for every model — they are the raw material for Section 4.5's per-class ensembling.

## 4.5 Metric Optimization, Ensembling, and TTA

The metric is unweighted macro-averaged ROC AUC over the 12 exam-level binaries:[^30^]

$$\text{Final Score} = \frac{1}{12}\sum_{i=0}^{11} \text{AUC}_i$$

Three consequences are definition-level, not speculative. **(a) Calibration and thresholds are irrelevant** — AUC is rank-based, so any monotonic per-class transform (Platt, isotonic, clipping) is a no-op; do not spend a single experiment on probability calibration. **(b) The worst class costs as much as the best.** Fracture, MCL, and Synovitis — precisely the classes with the noisiest silver supervision (fracture extraction sensitivity 0.44; synovitis 84% silent) — each contribute 1/12 of the score. Fixing the weakest column is worth more than polishing the strongest. **(c) Ensemble selection should be per-class:** select or weight ensemble members per label by their OOF AUC rather than globally; this directly exploits the macro-mean structure.[^31^]

On training loss, the evidence favors handling imbalance in the sampler and noise in the loss:

| Option | Verdict for this task | Evidence |
|---|---|---|
| Balanced study-level sampling | **Preferred first line** — sample positive/negative studies equally per class group | 2023 1st place: sampling > loss weights[^24^] |
| Asymmetric loss (γ−=4, γ+=1, clip=0.05) | **Preferred loss** — down-weights and hard-thresholds easy/mislabeled negatives; fits noisy multilabel supervision and uncoerced 0.5 cells | ASL paper, SOTA on multilabel benchmarks[^32^] |
| Weighted BCE (pos_weight) | Fallback only; failed to converge from scratch for 2023 winners; usable in a two-step recipe (AUC pretrain → frozen-backbone weighted head fine-tune) | 2023 1st; 2024 5th place[^24^][^33^] |
| Focal loss | Caution: emphasizes hard examples, which under noisy silver labels are disproportionately mislabeled | Focal-loss mechanics; CoPAS used focal on final output only[^16^][^34^] |
| Soft/graded targets | Use wherever the label pipeline emits them | +0.056 macro AUC (community-reported, single source)[^4^] |
| Loss masking on not-addressed cells | Mandatory if cells are not soft-coded; treat report-silent as unlabeled, not negative | 23% of studies report-silent; 0.44 fracture sensitivity[^3^][^35^] |

**TTA has one trap with a bounty attached.** A horizontal flip of a knee MRI swaps the medial and lateral compartments: Medial Meniscus↔Lateral Meniscus and Medial OA↔Lateral OA labels must be swapped on flip, in both training augmentation and TTA (with label-aware un-flipping). Handled correctly this was worth ~0.01 AUC to the RSNA 2025 aneurysm 5th place (left/right vessel labels swapped); handled naively it actively corrupts the four side-specific labels.[^36^] ACL/MCL/Effusion/Synovitis/Baker's/Contusion/Fracture are flip-invariant at exam level. Verify flip direction against ImageOrientationPatient before trusting it, validate every TTA set on OOF (recent evidence shows standard TTA can *hurt* medical classification), and skip TTA entirely on the efficiency-track submission.[^37^] For aggregation, max-pool over slices per series and use a small logistic stacker over planes — the MRNet template — rather than elaborate fusion.[^14^]

## 4.6 Playing the Efficiency Track

The first-ever RSNA efficiency track distributes $18,000 ($7k/$6k/$5k) across three places, scored on runtime plus AUC rather than accuracy rank.[^38^] The published definition (KaTeX partially lost in page extraction — **medium confidence on the exact form**, high confidence on direction and eligibility) reduces to:

$$\text{Efficiency} \approx \frac{\text{maxAUC} - \text{AUC}}{\text{maxAUC} - \text{BenchmarkAUC}} + \frac{\text{RuntimeSeconds}}{32400} \quad (\text{minimize})$$

where Benchmark is the `sample_submission.csv` private-LB score, maxAUC is the best private-LB submission, and 32,400 s is the 9-hour notebook cap. Eligibility requires beating the benchmark on the private leaderboard.[^38^] Whatever the exact normalization, the strategy is invariant: every saved hour buys ~0.11 of the runtime term, runtime counts the full evaluation wall time (DICOM I/O included, on a 569.76 GB dataset), and you must first clear the accuracy bar.

The rational solo play is to treat efficiency as a separate, winnable game rather than a constraint on the main entry. The recipe: **(1)** an ELNet-class student — ~0.2M parameters trained from scratch matched MRNet's 183M-parameter ensemble on knee MRI (0.904 vs 0.826 meniscus AUC) — distilled from the main-track ensemble's soft OOF predictions;[^39^] **(2)** FP16/AMP plus `torch.compile` as the safe acceleration path; **(3)** minimal series routing via `test_series.csv` plane/sequence flags, cached preprocessing, zero TTA, single SWA/EMA-averaged model. The dangerous temptation is TensorRT INT8: on Kaggle's T4 environment, TensorRT libraries can be absent, forcing ONNX Runtime into GPU↔CPU fallback measured at ~147× *slower* than PyTorch FP16 — vendor the libraries or build engines in-notebook, otherwise stay on FP16.[^40^] Submitting one maximal-ensemble notebook and one lean distilled notebook among the allowed final selections hedges both tracks simultaneously.

## Sources

[^1^]: Kaggle — RSNA Knee Abnormality Detection, Data page — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data (accessed 2026-08-10)
[^2^]: Kaggle discussion 733826 — "Possible inconsistencies between MRI reports and provided labels" (host reply by Po-Hao Chen) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826 (2026-08-08)
[^3^]: Kaggle discussion 733932 — "'Not addressed' is a label too — what we learned reading 4,407 knee reports with an LLM" (stevenleehans) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 (2026-08-09)
[^4^]: Kaggle discussion 734105 — "Graded targets beat binary ones by +0.056 macro AUC" — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105 (2026-08)
[^5^]: Roman Rozen, Kaggle notebook "RSNA Knee | Data structure, EDA, baseline" — https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline (2026-08-06)
[^6^]: Kaggle discussion 733965 — "Use of Commercially Hosted LLMs" (Competition Host) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965 (2026-08-09)
[^7^]: CXR-LanIC paper — arXiv:2510.21464 — https://arxiv.org/html/2510.21464v4 (2026-05-19)
[^8^]: Qwen3 Technical Report — arXiv:2505.09388 — https://arxiv.org/html/2505.09388v1 (2025)
[^9^]: "Anatomy-aware prompting for radiology report classification with GPT-OSS-20B" — arXiv:2512.05537 — https://arxiv.org/pdf/2512.05537 (2025-12)
[^10^]: Kaggle discussion 734095 — "Labeling the 4,349 report-only studies without an LLM, and what it scores on the 58 gold" (Busya PRIME) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734095 (2026-08-10)
[^11^]: Thieme/Rofo — "German CheXpert Chest X-ray Radiology Report Labeler" (Wollek et al. 2023) — https://www.thieme-connect.com/products/ejournals/html/10.1055/a-2234-8268 (2024-01-31)
[^12^]: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R) — arXiv:1911.02116 — https://arxiv.org/pdf/1911.02116 (2020)
[^13^]: "Evaluation of GPT-4o for multilingual translation of radiology reports" (European Radiology experimental) — https://www.sciencedirect.com/science/article/pii/S0720048X25004279 (2025-08-29); NLLB-200 model card — https://www.modelscope.cn/models/facebook/nllb-200-1.3B/summary (2022)
[^14^]: Bien et al., "Deep-learning-assisted diagnosis for knee MRI: Development and retrospective validation of MRNet" (PLoS Medicine) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/ (2018-11-27)
[^15^]: Kaggle write-up — "1st Place Solution" (Qishen Ha), RSNA 2022 Cervical Spine Fracture Detection — https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution (2022-10-29)
[^16^]: Qiu et al., "Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities" (CoPAS), Nature Communications 15 — https://www.nature.com/articles/s41467-024-51888-4 ; code: https://github.com/zqiuak/CoPAS (2024-09-02)
[^17^]: Kaggle write-up — "1st place solution" (NANACHI), RSNA 2024 Lumbar Spine Degenerative Classification — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution (2024-10-28)
[^18^]: Raghu et al., "Transfusion: Understanding Transfer Learning for Medical Imaging" — arXiv:1902.07208 — https://arxiv.org/abs/1902.07208 (2019)
[^19^]: "Robust Weakly Supervised Learning for COVID-19 Recognition Using Multi-Center CT Images" — arXiv:2112.04984 — https://arxiv.org/pdf/2112.04984.pdf (2021)
[^20^]: Kaggle write-up — "1st Place Solution" (tomoon33), RSNA Intracranial Aneurysm Detection — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution (2025-10-15)
[^21^]: Yiu et al., "A Systematic Study of Deep Learning Models and xAI Methods for ROI Detection in MRI Scans" — arXiv:2508.14151 — https://arxiv.org/html/2508.14151v1 (2025-08-19)
[^22^]: Kaggle write-up — "7th place solution - 3D nnU-Net + blob regression (again)" (MIC-DKFZ), RSNA Intracranial Aneurysm Detection — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution (2025-10-16)
[^23^]: Kaggle write-up — "6th Place Solution" (NVSpine), RSNA 2024 Lumbar Spine Degenerative Classification — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/nvspine-6th-place-solution (2024-10)
[^24^]: Kaggle write-up — "1st Place Solution: Team Oxygen", RSNA 2023 Abdominal Trauma Detection — https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen (2023-10-22)
[^25^]: Kushol et al., cross-manufacturer MRI harmonization study, Scientific Reports 2023 — https://www.nature.com/articles/s41598-023-43715-5.pdf (2023)
[^26^]: Kaggle — RSNA Knee Abnormality Detection, Leaderboard page — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard (accessed 2026-08-10)
[^27^]: Kaggle discussion 734004 — "DICOM metadata findings: scanner-grouped CV and PatientSex priors" (morningduck); discussion 733517 — "0.932 LB within one day. Tested for DICOM metadata shortcut" (Oleksii Zhukov) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517 (2026-08-07/09)
[^28^]: "Mitigating Bias in Radiology Machine Learning: 1. Data" (Radiology: AI) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9533091/
[^29^]: Szymański & Kajdanowicz, "A Network Perspective on Stratification of Multi-Label Data" (PMLR v74) — http://proceedings.mlr.press/v74/szymański17a/szymański17a.pdf (2017)
[^30^]: Kaggle — RSNA Knee Abnormality Detection, Overview → Evaluation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation (accessed 2026-08-10)
[^31^]: "Class-Wise Ensemble" multi-label radiology label paper — arXiv:2308.08853 — https://arxiv.org/pdf/2308.08853
[^32^]: Ridnik et al., "Asymmetric Loss For Multi-Label Classification" — arXiv:2009.14119 — https://arxiv.org/pdf/2009.14119 (2021)
[^33^]: siwooyong, RSNA 2024 Lumbar Spine 5th-place solution repository — https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification (2024)
[^34^]: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017), via TUM lecture slides — https://dvl.in.tum.de/slides/cv3dst-ws19/3.ObjectDetection2.pdf
[^35^]: Kaggle discussion 734117 — "Weak labels for all 12 findings + how recoverable each one actually is" (Luka Duvanov) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734117 (2026-08-10)
[^36^]: Kaggle write-up — "5th place solution with code" (HoangHuyen), RSNA Intracranial Aneurysm Detection — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution (2025-10-15)
[^37^]: "I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image Classification" — arXiv:2604.09697 — https://arxiv.org/html/2604.09697v1 (2026-04-06)
[^38^]: Kaggle — RSNA Knee Abnormality Detection, Overview → Efficiency Prize Evaluation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation (accessed 2026-08-10)
[^39^]: Tsai et al., "Knee Injury Detection using MRI with Efficiently-Layered Network (ELNet)" (MIDL 2020) — arXiv:2005.02706 — https://arxiv.org/pdf/2005.02706 ; code: https://github.com/mxtsai/ELNet (2020-05-06)
[^40^]: "Guidance-Aware Quantization for Classifier-Free Diffusion" (Kaggle T4 TensorRT measurement) — arXiv:2607.08241 — https://arxiv.org/html/2607.08241v1 (2026-07-09)
