# 1. Competition Playbook

This chapter is the operational briefing for the RSNA Knee Abnormality Detection competition hosted on Kaggle: what the task is, how it is scored, what the data actually contains, which rules constrain solution design, how the prize money is split, and which mistakes the community has already made so that the reader does not repeat them. All facts are drawn from the official competition pages, host statements in the discussion forum, and community analyses published between the launch and 2026-08-10; where a claim is a community measurement rather than an organizer statement, that provenance is stated explicitly.

## 1.1 What This Competition Is and Why It Is Unusual

The RSNA Knee Abnormality Detection competition is a Kaggle "Research Code Competition" organized by the Radiological Society of North America (RSNA) and co-led by musculoskeletal (MSK) radiologists Po-Hao "Howard" Chen and Naveen Subhas of the Cleveland Clinic.[^1^] The task is multi-label classification of knee magnetic resonance imaging (MRI) studies: for each examination, predict the presence or absence of twelve findings.[^2^] Three properties make this competition atypical relative to prior medical-imaging competitions, and all three should shape strategy from day one.

First, supervision is deliberately scarce. Of the 4,407 training studies, only 58 (approximately 1.3%) carry expert, image-derived labels for all twelve findings; the remaining 4,349 studies ship with the free-text radiology report and nothing else.[^3^] The organizers frame this as intended design: "Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies."[^4^] Consequently, the dominant engineering problem is not model architecture but label engineering — converting multilingual clinical prose into usable training targets.

Second, the reports are train-only. `test.csv` contains no `Report` column, and the host has confirmed that reports will not be available for the hidden test set at any point.[^5^] Text is therefore a source of training targets, never an inference-time input; any pipeline that needs report text at prediction time is structurally invalid for this competition.

Third, this is the first RSNA challenge with a dedicated efficiency track: $18,000 of the $77,000 prize pool rewards submissions that combine high predictive performance with low wall-clock runtime, rather than model size or energy use.[^6^] RSNA's press materials also describe the competition as the society's first on musculoskeletal MRI and the first to combine images with multilingual report text.[^7^]

## 1.2 Task Definition: Twelve Labels, One Metric

Submissions are scored by the unweighted macro-averaged area under the receiver operating characteristic curve (ROC AUC) across the twelve binary targets: Final Score = (1/12) Σᵢ AUCᵢ.[^8^] Two consequences follow directly. Because the metric is unweighted, every label contributes equally — a rare or hard finding such as synovitis is worth exactly as much as a common one such as effusion, so the weakest per-class AUC drags the final score as much as the strongest lifts it. Because ROC AUC is a rank-based metric, probability calibration, threshold tuning, and log-loss-style clipping have no effect on the score; effort spent there is wasted, and effort is better spent on ranking quality and per-class ensemble weighting.

The submission file is `submission.csv` with one row per test study, identified by `StudyInstanceUID`, plus twelve confidence-score columns in a fixed order.[^8^] Table 1 lists the exact column names as they must appear in the header, together with the organizers' label definitions.

**Table 1. The twelve target labels and exact submission column names.**[^2^][^8^]

| # | Submission column name | Finding |
|---|---|---|
| 1 | `ACL` | Anterior cruciate ligament injury (0/1) |
| 2 | `MCL` | Medial collateral ligament injury (0/1) |
| 3 | `Medial Meniscus` | Medial meniscus tear (0/1) |
| 4 | `Lateral Meniscus` | Lateral meniscus tear (0/1) |
| 5 | `Medial OA` | Osteoarthritis of the medial tibiofemoral compartment (0/1) |
| 6 | `Lateral OA` | Osteoarthritis of the lateral tibiofemoral compartment (0/1) |
| 7 | `PF OA` | Patellofemoral osteoarthritis (0/1) |
| 8 | `Effusion` | Joint effusion / excess fluid (0/1) |
| 9 | `Synovitis` | Inflammation of the joint lining (0/1) |
| 10 | `Baker's` | Baker's (popliteal) cyst (0/1) |
| 11 | `Contusion` | Bone contusion / bone bruise (0/1) |
| 12 | `Fracture` | Fracture (0/1) |

The header row is verbatim: `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`.[^8^] Note that four targets form medial/lateral pairs and one column name contains an apostrophe (`Baker's`) — both details matter later (Section 1.7).

The reference labels were produced by independent double reading by subspecialty-trained MSK radiologists, with a third radiologist adjudicating disagreements; labels are exam-level and refer to a single knee.[^9^] The annotation rubric is specificity-favoring: borderline ("on the fence") findings were graded negative, and explicit positivity thresholds apply — for example, an anterior cruciate ligament (ACL) tear requires complete discontinuity or more than 50% of fibers disrupted, a meniscal tear requires abnormal signal contacting the surface on at least two images, and osteoarthritis requires roughly 1 cm or more of high-grade cartilage loss.[^9^] The host has further confirmed that the 58 labeled training studies carry image-derived labels (not report-derived), that image-derived labels are authoritative where the two disagree, and that the same protocol produced the hidden test labels.[^10^] A community audit quantified the resulting gap: re-reading 20 gold studies from reports alone reproduced the provided labels with only 82.5% cell-level agreement (positive predictive agreement 73.1%), meaning reports systematically over-call relative to the rubric.[^10^]

## 1.3 Dataset Anatomy

The download totals 819,640 files and 569.76 GB, almost all of it Digital Imaging and Communications in Medicine (DICOM) images.[^4^] The training portion comprises 4,407 studies, 24,371 series, and roughly 730,000 slices (~265 GB); the test set contains approximately 1,300 studies.[^4^][^11^] Imaging is DICOM only — one `.dcm` file per slice — organized as `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`. Series typically contain 20–45 slices (median 30) with a long tail to a few hundred, and transfer syntaxes are mixed (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian), so the decoding path must handle all four.[^4^] Every DICOM has been stripped to an allowlist of 86 metadata tags; intensities, orientations, and resolutions vary across series and studies.[^4^]

Three tabular files matter for pipeline design. `train.csv` holds one row per study with `StudyInstanceUID`, `PatientSex` (Male/Female, may be blank), and the free-text `Report`.[^4^] `train_series.csv` provides three organizer-curated flags per series — `Fluid_Sensitive` (1 for T2/PD/STIR-like sequences), `Fat_Suppression` (0/1), and `Anatomical_Plane` (Sagittal/Coronal/Axial) — which enable plane- and sequence-routed inference without parsing pulse-sequence names, which are not provided.[^4^] `test.csv` contains only `StudyInstanceUID`; the shipped version is a three-row example placeholder that is replaced by the real test data during scoring.[^4^]

Two distribution facts constrain validation design. First, the data originate from 16 institutions across 5 continents, and the reports are multilingual: community script analysis of all 4,407 reports found Latin script in 97.6%, Greek in 7.3%, and Cyrillic in 5.0% (a report may contain more than one script), with English, Spanish, and Dutch dominant and French, German, Portuguese, Italian, Turkish, and Bulgarian attested.[^12^] RSNA's own pages variously state "nine" and "a dozen" languages; the exact count is unresolved, but the practical point — that a substantial minority of reports are not English — is not in doubt.[^13^] Second, the organizers explicitly warn that abnormality prevalence is not guaranteed to match across the training, public leaderboard, and final evaluation splits.[^4^]

## 1.4 Rules, Constraints, and the External-Data Question

This is a code competition: submissions are Kaggle Notebooks re-run by the platform against hidden test data. The binding constraints are summarized in Table 2.

**Table 2. Rules and constraints summary.**[^14^][^15^]

| Constraint | Value |
|---|---|
| Submission format | Kaggle Notebook only; output file must be named `submission.csv` |
| Runtime limit | ≤ 9 hours (CPU notebook or GPU notebook) |
| Internet access | Disabled during submission |
| External data / pretrained models | Allowed if freely and publicly available at minimal cost |
| Commercial large language model (LLM) APIs | Permitted for report-label extraction (host ruling, 2026-08-09) |
| Submissions per day | 5 |
| Final submissions selected | Up to 2 |
| Maximum team size | 5 |
| Winner license | CC-BY-NC 4.0 (code and weights) |
| Competition data license | RSNA MIRA license |

The external-data clause permits data and models that are "reasonably accessible to all" participants and of "minimal cost"; buying an expensive proprietary dataset would violate the reasonableness test.[^15^] On 2026-08-09 the host ruled that sending competition report text to commercially hosted LLM application programming interfaces (APIs) — OpenAI, Anthropic, Google — for label extraction is permitted and does not constitute prohibited private sharing of competition data, subject to the same accessibility and minimal-cost conditions.[^16^] This ruling legitimizes what was already the dominant community strategy (Section 1.6).

One external-data question remains unresolved as of 2026-08-10: the eligibility of gated click-through research datasets such as MRNet, fastMRI+, OAI, SKM-TEA, and KneeCoT has been raised in the forum but not yet ruled on by the host.[^17^] Treat these as pending; do not build an irreplaceable pipeline component on them until the rules thread is answered.

Winners' obligations are heavier than in a typical competition: winners must deliver training and inference code, model weights published as a public Kaggle dataset, an environment specification (`requirements.txt` plus a Kaggle image or Dockerfile), a short video presenting the approach, and a publicly distributable final model — all licensed CC-BY-NC 4.0.[^6^][^15^] Teams that intend to compete for prizes should log external data sources and licenses from the start, because reconstructing this provenance after the deadline is error-prone.

## 1.5 Prize Structure and the Efficiency Track

The total prize pool is $77,000, split between a main leaderboard track (10 paid places, $59,000) and an efficiency track (3 paid places, $18,000); a single submission may win both.[^6^]

**Table 3. Prize breakdown (USD).**[^6^]

| Place | Main leaderboard | Efficiency track |
|---|---|---|
| 1st | $9,000 | $7,000 |
| 2nd | $7,000 | $6,000 |
| 3rd | $6,500 | $5,000 |
| 4th | $6,000 | — |
| 5th | $5,500 | — |
| 6th–10th | $5,000 each | — |
| **Subtotal** | **$59,000** | **$18,000** |

The efficiency score combines private-test AUC with the evaluation notebook's wall-clock runtime, normalized by 32,400 seconds (9 hours, the runtime cap); the objective is to minimize it.[^18^] Eligibility requires the submission to be among the team's selected final submissions and to outrank the all-0.5 `sample_submission.csv` benchmark on the private leaderboard.[^18^] Kaggle staff have clarified that graphics processing unit (GPU) notebooks are eligible and that `RuntimeSeconds` includes everything from execution start to end — package installation, model loading, and DICOM decoding.[^19^] The exact algebraic arrangement of the AUC term does not survive page-text extraction; the best-supported reading is Efficiency = (Benchmark − AUC)/(Benchmark − max AUC) + RuntimeSeconds/32,400, where max AUC is the best private-leaderboard score.[^18^] The practical implications hold regardless of the exact fraction: runtime only matters up to the cap (each hour saved is worth ~0.11 of the runtime term), the benchmark must be cleared first, and heavy test-time augmentation or deep ensembles trade directly against the efficiency score. A public efficiency leaderboard notebook, updated daily during the training phase, shows team ranks only, not full scores.[^20^]

## 1.6 Timeline and Current Leaderboard State

**Table 4. Competition timeline (all deadlines 11:59 PM UTC).**[^21^][^22^]

| Date | Event |
|---|---|
| 2026-07-30 | Start date (Kaggle timeline); public RSNA announcement ~2026-08-05 |
| 2026-10-15 | Entry deadline (rules acceptance) and team merger deadline |
| 2026-10-22 | Final submission deadline |
| 2026-11-05 | Winners' requirement deadline (code, video, method description) |
| 2026-11-29 to 2026-12-03 | RSNA 2026 annual meeting, Chicago; winners recognized in the AI Theater |

The Kaggle timeline lists 2026-07-30 as the start date while RSNA's public announcement appeared around 2026-08-05; the difference is a soft open versus public launch, not a data error.[^21^][^22^]

The leaderboard is two-stage: the public leaderboard is computed on approximately 30% of the test data, and final standings use the other 70%.[^11^] As of 2026-08-10, the top public score is approximately 0.942, and the top ~50 teams are all at or above 0.900 — a compressed board where small private-set differences can reorder ranks.[^11^] Participation at the same date stood at roughly 1,011 participants in 961 teams with about 4,060 submissions.[^23^]

Two measurements calibrate what public scores mean. A community probe established that DICOM metadata alone (no pixels) reaches only 0.6516 macro AUC under random folds and 0.5981 under scanner-grouped folds, ruling out a metadata shortcut and indicating that leaderboard scores above 0.9 reflect genuine image content.[^24^] Meanwhile, the top-voted public baseline — a DINOv2-based model trained on LLM-extracted report labels — scores 0.891 on the public leaderboard with about 6.5 minutes of inference on dual T4 GPUs, and the strongest public notebooks sit at 0.894–0.899.[^25^] The gap between public baselines (~0.89) and the leaderboard top (~0.94) within ten days of launch is consistent with the community consensus that label quality, not encoder choice, is the binding constraint.[^26^]

## 1.7 Known Pitfalls Identified by the Community

The following failure modes have been documented in the discussion forum with measurements; each is cheap to avoid and expensive to discover late.

**Empty label cells are not zeros.** In `train.csv`, the 4,349 unlabeled studies have empty strings in the twelve label columns. Filling them with 0 fabricates roughly 4,349 false negatives per class and collapses apparent prevalence to ~0.5% per finding.[^3^] Treat empty as missing and mask the loss.

**Reports contain embedded newlines.** `train.csv` is 58,556 lines long but has only 4,407 rows; naive line counting overstates the row count by ~13×. Read the file with a CSV parser, not line splitting.[^3^]

**English-only keyword extraction fails silently on non-Latin reports.** Greek and Cyrillic reports together cover roughly 12% of studies; a regex pipeline built on English or Spanish terms extracts nothing from them and returns a confident negative for every finding. Unicode near-duplicates compound this: U+03BC (Greek small letter mu) and U+00B5 (micro sign) render identically but do not string-match.[^3^][^12^] Measured against the 58 gold studies, LLM-extracted labels achieve 0.8780 macro AUC versus 0.8136 for regex extraction, and 25.4% of all report-label cells are "not addressed" by the report at all — silence that is informative for some findings (an unmentioned Baker's cyst is almost certainly absent) and uninformative for others (unmentioned synovitis is still present in ~34% of gold cases).[^26^]

**Horizontal flips swap medial and lateral labels.** Four targets are medial/lateral pairs (Medial/Lateral Meniscus, Medial/Lateral OA). A horizontal flip augmentation or test-time augmentation that does not swap these label pairs trains the model on mirrored anatomy with unmirrored labels.[^11^] Verify flip direction against the DICOM `ImageOrientationPatient` tag rather than assuming a display convention.

**Site memorization does not transfer.** A metadata-only classifier loses ~0.05 macro AUC when cross-validation folds are grouped by scanner instead of randomized (0.652 versus 0.598), with osteoarthritis targets dropping most.[^24^][^27^] Scanner-grouped or site-grouped validation folds are the defense; the same measurement rules out any DICOM-metadata shortcut to the leaderboard.

**Inference-time text is a dead end.** Because `test.csv` has no `Report` column, any architecture that consumes report text at prediction time cannot produce a valid submission; reports may only generate training targets.[^5^]

## Sources

[^1^]: RSNA News — AI Challenge Knee MRI — https://www.rsna.org/news/2026/august/ai-challenge-knee-mri (2026-08-06)
[^2^]: Kaggle — RSNA Knee Abnormality Detection, Data page — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data (accessed 2026-08-10)
[^3^]: Kaggle Discussion 734055 — "train.csv has 4,407 studies and 58 labels" (maximo lorenzo y losada) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055 (2026-08-09/10)
[^4^]: Kaggle — RSNA Knee Abnormality Detection, Data page (Dataset Description, DICOM Notes, Dataset Distribution Notice) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data (accessed 2026-08-10)
[^5^]: Kaggle Discussions 733932 / 733592 — reports are train-only; host confirmation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592 (2026-08-08/09)
[^6^]: Kaggle — RSNA Knee Abnormality Detection, Overview: Prizes — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/prizes (accessed 2026-08-10)
[^7^]: RSNA press release — https://www.rsna.org/media/press/2026/2669 (2026-08-05)
[^8^]: Kaggle — RSNA Knee Abnormality Detection, Overview: Evaluation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation (accessed 2026-08-10)
[^9^]: Kaggle Discussion 733343 — "Knee Abnormality Detection AI Challenge Overview" (pinned host post, courtesy of Dr. Jacob Kazam) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343 (2026-08-06)
[^10^]: Kaggle Discussion 733826 — host replies on image-derived labels; community report-only audit — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826 (2026-08-08)
[^11^]: Kaggle — Leaderboard page and Data page — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard (accessed 2026-08-10)
[^12^]: Kaggle Discussion 734055 — multilingual/multi-script report analysis — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055 (2026-08-10)
[^13^]: RSNA challenge page and press materials ("nine languages" vs "a dozen") — https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge ; https://www.rsna.org/news/2026/august/ai-challenge-knee-mri (accessed 2026-08-10)
[^14^]: Kaggle — RSNA Knee Abnormality Detection, Overview: Code Requirements — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements (accessed 2026-08-10)
[^15^]: Kaggle — RSNA Knee Abnormality Detection, Rules — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules (accessed 2026-08-10)
[^16^]: Kaggle Discussion 733965 — "Use of Commercially Hosted LLMs" (host ruling) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965 (2026-08-09)
[^17^]: Kaggle Discussion 733652 — external knee-MRI dataset eligibility (unresolved) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652 (2026-08)
[^18^]: Kaggle — Overview: Efficiency Prize Evaluation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation (accessed 2026-08-10)
[^19^]: Kaggle Discussion 733475 — RuntimeSeconds definition (Ryan Holbrook, Kaggle Staff) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475 (2026-08-07)
[^20^]: Kaggle Notebook — "RSNA Knee Abnormalities — Efficiency LB" (Ryan Holbrook) — https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb (accessed 2026-08-10)
[^21^]: Kaggle — Overview: Timeline — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline (accessed 2026-08-10)
[^22^]: RSNA News — competition dates and RSNA 2026 recognition — https://www.rsna.org/news/2026/august/ai-challenge-knee-mri (2026-08-06)
[^23^]: Kaggle — Overview page sidebar (participation snapshot) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview (accessed 2026-08-10)
[^24^]: Kaggle Discussion 733517 — "0.932 LB within one day. Tested for DICOM metadata shortcut" (Oleksii Zhukov) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517 (2026-08-07)
[^25^]: Kaggle — Code page sorted by votes (public baselines and scores) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code?sortBy=voteCount (accessed 2026-08-10)
[^26^]: Kaggle Discussion 733932 — "Not addressed is a label too" (stevenleehans): LLM vs regex label quality — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 (2026-08-09)
[^27^]: Kaggle Discussion 734004 — "DICOM metadata findings: scanner-grouped CV and PatientSex priors" (morningduck) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004 (2026-08-09)
