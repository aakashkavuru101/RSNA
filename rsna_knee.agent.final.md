# RSNA Knee Abnormality Detection — Research Dossier & Seed Knowledge Pack

**Prepared:** August 10, 2026 · **Competition deadline:** October 22, 2026 (entry/team deadline October 15, 2026)
**Audience:** A machine-learning practitioner with no medical background, building a solution with an agentic coding platform (Kimi Code) and Kaggle Notebooks.

---

## Executive Summary

The 2026 RSNA Knee Abnormality Detection AI Challenge asks participants to detect **twelve clinically important knee abnormalities** — ACL, MCL, medial/lateral meniscus tears, medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion, and fracture — from knee MRI exams, scored by **macro-averaged ROC AUC**. Three structural facts define the competition:

1. **It is a weak-supervision problem disguised as an imaging problem.** Only 58 of 4,407 training studies carry expert labels; the rest must be labeled by extracting findings from real-world radiology reports written in roughly ten languages. Measured on the 58 gold studies, LLM-extracted labels score 0.878 macro AUC versus 0.814 for regex — label quality is the single largest controllable lever.
2. **Reports are a train-time teacher only.** The test set contains no report text, so the winning frame is a text-supervised, image-only model — not a test-time multimodal system.
3. **The proven architecture recipe is already known.** Past RSNA winners and the knee-MRI literature converge on per-plane 2D/2.5D pretrained CNNs with attention-MIL or BiLSTM slice aggregation, optionally preceded by a localize-then-classify stage. End-to-end 3D networks and large vision transformers consistently fail at this data scale.

The dossier is organized as: Chapter 1, the competition playbook (rules, data, prizes, pitfalls); Chapter 2, a clinical primer requiring zero medical background; Chapter 3, an annotated seed library of 38 papers and the public datasets; Chapter 4, modeling and validation strategy; Chapter 5, an engineering roadmap with a week-by-week plan to the October 22 deadline.

**Ten cross-cutting insights** from the research synthesis: (1) label quality > architecture; (2) text-supervised image student is the correct frame; (3) evaluation is deliberately fragile (public LB = 30% of test, prevalence unmatched — expect shake-up); (4) 16-site domain shift is the hidden ceiling — scanner-grouped CV is mandatory; (5) the architecture recipe has converged — implement the skeleton, don't explore; (6) the $18k efficiency track is a separate, winnable game via distillation; (7) CoPAS (Nature Communications 2024) is a public-code template for the same 12 labels; (8) invest in validation before modeling — 58 gold studies is the only honest anchor; (9) the 12 labels are clinically coupled — exploit multi-task structure and beware the horizontal-flip laterality trap; (10) the leaderboard is already compressing (0.942 by day 5) — differentiate with label ensembles and localization, not baseline replication.

---
# 1. Competition Playbook

This chapter is the operational briefing for the RSNA Knee Abnormality Detection competition hosted on Kaggle: what the task is, how it is scored, what the data actually contains, which rules constrain solution design, how the prize money is split, and which mistakes the community has already made so that the reader does not repeat them. All facts are drawn from the official competition pages, host statements in the discussion forum, and community analyses published between the launch and 2026-08-10; where a claim is a community measurement rather than an organizer statement, that provenance is stated explicitly.

## 1.1 What This Competition Is and Why It Is Unusual

The RSNA Knee Abnormality Detection competition is a Kaggle "Research Code Competition" organized by the Radiological Society of North America (RSNA) and co-led by musculoskeletal (MSK) radiologists Po-Hao "Howard" Chen and Naveen Subhas of the Cleveland Clinic.[^1^] The task is multi-label classification of knee magnetic resonance imaging (MRI) studies: for each examination, predict the presence or absence of twelve findings.[^2^] Three properties make this competition atypical relative to prior medical-imaging competitions, and all three should shape strategy from day one.

First, supervision is deliberately scarce. Of the 4,407 training studies, only 58 (approximately 1.3%) carry expert, image-derived labels for all twelve findings; the remaining 4,349 studies ship with the free-text radiology report and nothing else.[^3^] The organizers frame this as intended design: "Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies."[^2^] Consequently, the dominant engineering problem is not model architecture but label engineering — converting multilingual clinical prose into usable training targets.

Second, the reports are train-only. `test.csv` contains no `Report` column, and the host has confirmed that reports will not be available for the hidden test set at any point.[^4^] Text is therefore a source of training targets, never an inference-time input; any pipeline that needs report text at prediction time is structurally invalid for this competition.

Third, this is the first RSNA challenge with a dedicated efficiency track: $18,000 of the $77,000 prize pool rewards submissions that combine high predictive performance with low wall-clock runtime, rather than model size or energy use.[^5^] RSNA's press materials also describe the competition as the society's first on musculoskeletal MRI and the first to combine images with multilingual report text.[^6^]

## 1.2 Task Definition: Twelve Labels, One Metric

Submissions are scored by the unweighted macro-averaged area under the receiver operating characteristic curve (ROC AUC) across the twelve binary targets: Final Score = (1/12) Σᵢ AUCᵢ.[^7^] Two consequences follow directly. Because the metric is unweighted, every label contributes equally — a rare or hard finding such as synovitis is worth exactly as much as a common one such as effusion, so the weakest per-class AUC drags the final score as much as the strongest lifts it. Because ROC AUC is a rank-based metric, probability calibration, threshold tuning, and log-loss-style clipping have no effect on the score; effort spent there is wasted, and effort is better spent on ranking quality and per-class ensemble weighting.

The submission file is `submission.csv` with one row per test study, identified by `StudyInstanceUID`, plus twelve confidence-score columns in a fixed order.[^7^] Table 1 lists the exact column names as they must appear in the header, together with the organizers' label definitions.

**Table 1. The twelve target labels and exact submission column names.**[^2^][^7^]

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

The header row is verbatim: `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`.[^7^] Note that four targets form medial/lateral pairs and one column name contains an apostrophe (`Baker's`) — both details matter later (Section 1.7).

The reference labels were produced by independent double reading by subspecialty-trained MSK radiologists, with a third radiologist adjudicating disagreements; labels are exam-level and refer to a single knee.[^8^] The annotation rubric is specificity-favoring: borderline ("on the fence") findings were graded negative, and explicit positivity thresholds apply — for example, an anterior cruciate ligament (ACL) tear requires complete discontinuity or more than 50% of fibers disrupted, a meniscal tear requires abnormal signal contacting the surface on at least two images, and osteoarthritis requires roughly 1 cm or more of high-grade cartilage loss.[^8^] The host has further confirmed that the 58 labeled training studies carry image-derived labels (not report-derived), that image-derived labels are authoritative where the two disagree, and that the same protocol produced the hidden test labels.[^9^] A community audit quantified the resulting gap: re-reading 20 gold studies from reports alone reproduced the provided labels with only 82.5% cell-level agreement (positive predictive agreement 73.1%), meaning reports systematically over-call relative to the rubric.[^9^]

## 1.3 Dataset Anatomy

The download totals 819,640 files and 569.76 GB, almost all of it Digital Imaging and Communications in Medicine (DICOM) images.[^2^] The training portion comprises 4,407 studies, 24,371 series, and roughly 730,000 slices (~265 GB); the test set contains approximately 1,300 studies.[^2^][^10^] Imaging is DICOM only — one `.dcm` file per slice — organized as `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`. Series typically contain 20–45 slices (median 30) with a long tail to a few hundred, and transfer syntaxes are mixed (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian), so the decoding path must handle all four.[^2^] Every DICOM has been stripped to an allowlist of 86 metadata tags; intensities, orientations, and resolutions vary across series and studies.[^2^]

Three tabular files matter for pipeline design. `train.csv` holds one row per study with `StudyInstanceUID`, `PatientSex` (Male/Female, may be blank), and the free-text `Report`.[^2^] `train_series.csv` provides three organizer-curated flags per series — `Fluid_Sensitive` (1 for T2/PD/STIR-like sequences), `Fat_Suppression` (0/1), and `Anatomical_Plane` (Sagittal/Coronal/Axial) — which enable plane- and sequence-routed inference without parsing pulse-sequence names, which are not provided.[^2^] `test.csv` contains only `StudyInstanceUID`; the shipped version is a three-row example placeholder that is replaced by the real test data during scoring.[^2^]

Two distribution facts constrain validation design. First, the data originate from 16 institutions across 5 continents, and the reports are multilingual: community script analysis of all 4,407 reports found Latin script in 97.6%, Greek in 7.3%, and Cyrillic in 5.0% (a report may contain more than one script), with English, Spanish, and Dutch dominant and French, German, Portuguese, Italian, Turkish, and Bulgarian attested.[^3^] RSNA's own pages variously state "nine" and "a dozen" languages; the exact count is unresolved, but the practical point — that a substantial minority of reports are not English — is not in doubt.[^11^] Second, the organizers explicitly warn that abnormality prevalence is not guaranteed to match across the training, public leaderboard, and final evaluation splits.[^2^]

## 1.4 Rules, Constraints, and the External-Data Question

This is a code competition: submissions are Kaggle Notebooks re-run by the platform against hidden test data. The binding constraints are summarized in Table 2.

**Table 2. Rules and constraints summary.**[^12^][^13^]

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

The external-data clause permits data and models that are "reasonably accessible to all" participants and of "minimal cost"; buying an expensive proprietary dataset would violate the reasonableness test.[^13^] On 2026-08-09 the host ruled that sending competition report text to commercially hosted LLM application programming interfaces (APIs) — OpenAI, Anthropic, Google — for label extraction is permitted and does not constitute prohibited private sharing of competition data, subject to the same accessibility and minimal-cost conditions.[^14^] This ruling legitimizes what was already the dominant community strategy (Section 1.6).

One external-data question remains unresolved as of 2026-08-10: the eligibility of gated click-through research datasets such as MRNet, fastMRI+, OAI, SKM-TEA, and KneeCoT has been raised in the forum but not yet ruled on by the host.[^15^] Treat these as pending; do not build an irreplaceable pipeline component on them until the rules thread is answered.

Winners' obligations are heavier than in a typical competition: winners must deliver training and inference code, model weights published as a public Kaggle dataset, an environment specification (`requirements.txt` plus a Kaggle image or Dockerfile), a short video presenting the approach, and a publicly distributable final model — all licensed CC-BY-NC 4.0.[^5^][^13^] Teams that intend to compete for prizes should log external data sources and licenses from the start, because reconstructing this provenance after the deadline is error-prone.

## 1.5 Prize Structure and the Efficiency Track

The total prize pool is $77,000, split between a main leaderboard track (10 paid places, $59,000) and an efficiency track (3 paid places, $18,000); a single submission may win both.[^5^]

**Table 3. Prize breakdown (USD).**[^5^]

| Place | Main leaderboard | Efficiency track |
|---|---|---|
| 1st | $9,000 | $7,000 |
| 2nd | $7,000 | $6,000 |
| 3rd | $6,500 | $5,000 |
| 4th | $6,000 | — |
| 5th | $5,500 | — |
| 6th–10th | $5,000 each | — |
| **Subtotal** | **$59,000** | **$18,000** |

The efficiency score combines private-test AUC with the evaluation notebook's wall-clock runtime, normalized by 32,400 seconds (9 hours, the runtime cap); the objective is to minimize it.[^16^] Eligibility requires the submission to be among the team's selected final submissions and to outrank the all-0.5 `sample_submission.csv` benchmark on the private leaderboard.[^16^] Kaggle staff have clarified that graphics processing unit (GPU) notebooks are eligible and that `RuntimeSeconds` includes everything from execution start to end — package installation, model loading, and DICOM decoding.[^17^] The exact algebraic arrangement of the AUC term does not survive page-text extraction; the best-supported reading is Efficiency = (Benchmark − AUC)/(Benchmark − max AUC) + RuntimeSeconds/32,400, where max AUC is the best private-leaderboard score.[^16^] The practical implications hold regardless of the exact fraction: runtime only matters up to the cap (each hour saved is worth ~0.11 of the runtime term), the benchmark must be cleared first, and heavy test-time augmentation or deep ensembles trade directly against the efficiency score. A public efficiency leaderboard notebook, updated daily during the training phase, shows team ranks only, not full scores.[^18^]

## 1.6 Timeline and Current Leaderboard State

**Table 4. Competition timeline (all deadlines 11:59 PM UTC).**[^19^][^1^]

| Date | Event |
|---|---|
| 2026-07-30 | Start date (Kaggle timeline); public RSNA announcement ~2026-08-05 |
| 2026-10-15 | Entry deadline (rules acceptance) and team merger deadline |
| 2026-10-22 | Final submission deadline |
| 2026-11-05 | Winners' requirement deadline (code, video, method description) |
| 2026-11-29 to 2026-12-03 | RSNA 2026 annual meeting, Chicago; winners recognized in the AI Theater |

The Kaggle timeline lists 2026-07-30 as the start date while RSNA's public announcement appeared around 2026-08-05; the difference is a soft open versus public launch, not a data error.[^19^][^1^]

The leaderboard is two-stage: the public leaderboard is computed on approximately 30% of the test data, and final standings use the other 70%.[^10^] As of 2026-08-10, the top public score is approximately 0.942, and the top ~50 teams are all at or above 0.900 — a compressed board where small private-set differences can reorder ranks.[^10^] Participation at the same date stood at roughly 1,011 participants in 961 teams with about 4,060 submissions.[^20^]

Two measurements calibrate what public scores mean. A community probe established that DICOM metadata alone (no pixels) reaches only 0.6516 macro AUC under random folds and 0.5981 under scanner-grouped folds, ruling out a metadata shortcut and indicating that leaderboard scores above 0.9 reflect genuine image content.[^21^] Meanwhile, the top-voted public baseline — a DINOv2-based model trained on LLM-extracted report labels — scores 0.891 on the public leaderboard with about 6.5 minutes of inference on dual T4 GPUs, and the strongest public notebooks sit at 0.894–0.899.[^22^] The gap between public baselines (~0.89) and the leaderboard top (~0.94) within ten days of launch is consistent with the community consensus that label quality, not encoder choice, is the binding constraint.[^4^]

## 1.7 Known Pitfalls Identified by the Community

The following failure modes have been documented in the discussion forum with measurements; each is cheap to avoid and expensive to discover late.

**Empty label cells are not zeros.** In `train.csv`, the 4,349 unlabeled studies have empty strings in the twelve label columns. Filling them with 0 fabricates roughly 4,349 false negatives per class and collapses apparent prevalence to ~0.5% per finding.[^3^] Treat empty as missing and mask the loss.

**Reports contain embedded newlines.** `train.csv` is 58,556 lines long but has only 4,407 rows; naive line counting overstates the row count by ~13×. Read the file with a CSV parser, not line splitting.[^3^]

**English-only keyword extraction fails silently on non-Latin reports.** Greek and Cyrillic reports together cover roughly 12% of studies; a regex pipeline built on English or Spanish terms extracts nothing from them and returns a confident negative for every finding. Unicode near-duplicates compound this: U+03BC (Greek small letter mu) and U+00B5 (micro sign) render identically but do not string-match.[^3^][^3^] Measured against the 58 gold studies, LLM-extracted labels achieve 0.8780 macro AUC versus 0.8136 for regex extraction, and 25.4% of all report-label cells are "not addressed" by the report at all — silence that is informative for some findings (an unmentioned Baker's cyst is almost certainly absent) and uninformative for others (unmentioned synovitis is still present in ~34% of gold cases).[^4^]

**Horizontal flips swap medial and lateral labels.** Four targets are medial/lateral pairs (Medial/Lateral Meniscus, Medial/Lateral OA). A horizontal flip augmentation or test-time augmentation that does not swap these label pairs trains the model on mirrored anatomy with unmirrored labels.[^10^] Verify flip direction against the DICOM `ImageOrientationPatient` tag rather than assuming a display convention.

**Site memorization does not transfer.** A metadata-only classifier loses ~0.05 macro AUC when cross-validation folds are grouped by scanner instead of randomized (0.652 versus 0.598), with osteoarthritis targets dropping most.[^21^][^23^] Scanner-grouped or site-grouped validation folds are the defense; the same measurement rules out any DICOM-metadata shortcut to the leaderboard.

**Inference-time text is a dead end.** Because `test.csv` has no `Report` column, any architecture that consumes report text at prediction time cannot produce a valid submission; reports may only generate training targets.[^4^]

# 2. Clinical Primer for Non-Doctors

This chapter gives you the minimum clinical literacy needed to work on knee magnetic resonance imaging (MRI) data: what the anatomy is, how the images are made, what the twelve competition labels actually mean, and how much the humans who produced those labels disagree with each other. Every medical term is defined in plain language at first use, with an analogy; afterwards it is used as-is. No prior anatomy or radiology knowledge is assumed.

## 2.1 Knee Anatomy in Fifteen Minutes

**The joint.** The knee is the largest synovial joint in the body — a synovial joint being one whose bone ends live inside a fluid-filled capsule, like engine parts sealed inside an oil bath. Structurally it is a modified hinge between three bones: the **femur** (thigh bone), the **tibia** (shin bone), and the **patella** (kneecap).[^24^] Think of a door hinge that also permits a small amount of twist. The bottom of the femur splits into two rounded knobs called **condyles** (medial = inner side, lateral = outer side) that rest on the nearly flat top of the tibia, the **tibial plateau** — two balls sitting on a table. Because balls-on-a-table is inherently unstable, the knee relies on soft-tissue "ropes," "washers," and "padding." There are really two joints in one: the **tibiofemoral joint** (femur on tibia, with a medial and a lateral compartment) and the **patellofemoral joint** (kneecap gliding in a femoral groove called the **trochlea**).

**Ligaments — the ropes.** Four ligaments stabilize the knee. The **anterior cruciate ligament (ACL)** and **posterior cruciate ligament (PCL)** cross each other in the center of the joint (cruciate = crossing); the ACL stops the tibia sliding forward and controls rotation, while the thicker PCL stops it sliding backward.[^25^] The ACL is the most commonly torn knee ligament — the classic pivoting sports injury. On the sides, the **medial collateral ligament (MCL)** is a broad flat band resisting forces that buckle the knee inward, and the **lateral collateral ligament (LCL)** is a cord resisting outward bowing. Normal ligaments are taut, low-water structures, and — this matters later — they appear dark on every MRI sequence.[^26^]

**Menisci — the washers.** Sitting on the tibial plateau are two C-shaped wedges of **fibrocartilage** (a tough, rubbery cartilage): the medial and lateral **menisci**. Each has an **anterior horn** (front tip), a **body** (middle), and a **posterior horn** (rear tip). Like rubber washers between pipe fittings, they deepen the socket, spread load, and absorb shock: they transmit roughly 50% of the load through the medial compartment and 70% through the lateral, and removing one raises contact stress by 100–300%.[^27^] Only the outer rim has a blood supply (the "red zone," which can heal); the inner "white zone" cannot.

**Cartilage — the non-stick coating.** The bone ends and the back of the patella are covered by a few millimeters of **articular (hyaline) cartilage**, a smooth, water-rich tissue that lets bone glide on bone almost friction-free — the joint's Teflon coating.[^28^] It has no blood supply and no nerves, so it heals poorly, and its loss is the core lesion of osteoarthritis.

**Supporting cast.** Tendons are cables from muscle to bone: the quadriceps tendon runs into the top of the patella and the patellar tendon runs from its bottom to the tibia (together, the **extensor mechanism** — the pulley system that straightens the knee). **Bursae** are small fluid-filled sacs that work like bubble wrap, reducing friction where tendons glide over bone; one of them, behind the knee, is where a Baker cyst forms. The **synovium** is the membrane lining the joint capsule; it secretes lubricating **synovial fluid**. Normally only a trace of fluid exists, so excess fluid is always a sign that something is wrong. Finally, **Hoffa's fat pad** is a cushion of fat behind the patellar tendon, and **bone marrow** — the fatty tissue inside bones — turns out to be one of the most informative tissues on MRI, because injury and overuse change its water content.

## 2.2 How Knee MRI Works: Planes, Sequences, and Signal Logic

**The physics in one paragraph.** MRI exploits hydrogen protons, which are abundant in water and fat. A strong magnet (1.5 or 3 tesla in clinical knee imaging) aligns the protons like compass needles; a radiofrequency pulse tips them out of alignment; as they relax back, they emit radio signals that receiver coils detect; magnetic field gradients encode where each signal came from; a computer reconstructs the image.[^29^] Two relaxation constants generate contrast: **T1** (how fast protons realign with the main field) and **T2** (how fast they dephase relative to each other). By choosing pulse timing parameters — the repetition time (TR) and echo time (TE) — the scanner emphasizes T1 contrast, T2 contrast, or a high-detail intermediate called **proton density (PD)**. No ionizing radiation is involved.

**Planes.** A knee MRI exam is not one image but several **series**, each a stack of roughly 3–4 mm slices acquired in one of three orthogonal planes:[^30^][^31^]

| Plane | What the slice shows | Structures best evaluated |
|---|---|---|
| Sagittal | Side-view slices, left to right | ACL, PCL, meniscal horns, extensor mechanism, marrow |
| Coronal | Front-view slices, front to back | MCL, LCL, meniscal bodies, roots and extrusion, compartment cartilage |
| Axial | Top-down cross-sections | Patellofemoral cartilage, trochlea, popliteal fossa (Baker cyst), bursae |

The plane–pathology mapping to memorize: **sagittal → cruciate ligaments and meniscal horns; coronal → collateral ligaments and meniscal bodies; axial → patellofemoral cartilage and Baker cysts.**[^30^][^31^]

**Sequences and signal logic.** The one mental model you need: **water is the star of pathology detection.** Almost everything that goes wrong in a knee — tears, bruises, inflammation, cysts — involves fluid or edema (swelling: excess water in tissue). Sequences either render anatomy crisply (T1, PD) or make fluid glow (T2 and its fat-suppressed variants). Meanwhile ligaments, tendons, menisci, and cortical bone are normally **dark on every sequence**, so any bright signal inside them is suspicious. Fat is bright on T1 and stays annoyingly bright on fast T2; **fat suppression (FS)** deliberately turns fat dark so that fluid "pops" against a dark background, and **STIR** (short tau inversion recovery) is an alternative suppression method that is very uniform and robust near metal.[^32^][^33^][^34^]

| Sequence | Fat | Fluid/edema | Ligament, meniscus, tendon | Primary role |
|---|---|---|---|---|
| T1-weighted | Bright | Dark | Dark | Anatomy, marrow fat, fracture lines |
| T2-weighted | Bright | **Bright** | Dark | Fluid, cysts, effusion, inflammation |
| PD | Bright | Intermediate | Dark, high detail | Highest anatomic detail: menisci, ligaments, cartilage |
| PD-FS / T2-FS | **Dark** | **Bright** | Dark | The workhorse: edema, tears, cartilage, effusion |
| STIR | Very dark | Very bright | Dark | Uniform suppression; sensitive to edema; robust near metal |

A standard clinical knee protocol combines the three orthogonal planes with fluid-sensitive fat-suppressed sequences plus one T1 series, typically four to seven series per exam; sagittal and coronal PD-FS are the meniscal workhorses.[^27^] For you as a modeler, this means each exam is a small set of 3D volumes that differ in plane and contrast, are spatially ordered within a series, but are **not** co-registered across series — and protocol details vary across institutions, a major domain-shift axis.

Why do radiologists insist on multiple planes and sequences? Three reasons that translate directly into model design. First, **confirmation**: partial-volume averaging (a slice straddling two structures) can fake a lesion on a single image, so a real abnormality should appear in two planes or on two consecutive images. The classic quantification: meniscal signal reaching the articular surface on at least two images has a positive predictive value (PPV) for a true tear of 94% (medial) and 96% (lateral); on only one image it drops to 43% and 18%.[^27^] Second, **geometry**: no single plane contains an oblique structure like the ACL. Third, **contrast complementarity**: a lesion invisible on T1 may glow on T2-FS. Models that pool evidence across planes and sequences are mimicking this cross-confirmation step.

## 2.3 The Twelve Target Abnormalities

The competition asks for twelve binary exam-level labels, scored as the unweighted macro-average of the twelve per-label areas under the receiver operating characteristic curve (AUC).[^35^][^36^] The labels are: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial osteoarthritis (OA), Lateral OA, patellofemoral (PF) OA, Effusion, Synovitis, Baker's cyst, Contusion, and Fracture.[^35^] Gold-standard labels were produced by two subspecialty musculoskeletal radiologists with a third adjudicator, using published **positivity thresholds** — and, critically, **borderline cases were graded negative**, a specificity-favoring convention you should mirror when choosing decision thresholds.[^35^]

| Label | What it is (plain language) | Best plane & sequence | Competition positivity threshold |
|---|---|---|---|
| ACL | Tear of the anterior cruciate ligament; the commonest knee ligament injury[^37^] | Sagittal PD-FS/T2-FS; confirm coronal/axial | >50% fiber disruption[^35^] |
| MCL | Sprain or tear of the inner-side stabilizer band[^38^] | Coronal PD-FS/T2-FS | Not separately published; borderline graded negative[^35^] |
| Medial Meniscus | Cleft in the inner C-shaped cartilage washer[^39^] | Sagittal PD-FS (horns) + coronal (body) | Signal reaching an articular surface on ≥2 images[^35^] |
| Lateral Meniscus | Same, outer washer (more mobile, tears less often) | Sagittal PD-FS + coronal | Signal reaching an articular surface on ≥2 images[^35^] |
| Medial OA | Wear-and-tear of the inner femur–tibia compartment: cartilage loss, bone spurs, marrow lesions[^40^] | Coronal + sagittal PD-FS | ≥1 cm of high-grade cartilage loss in the compartment[^35^] |
| Lateral OA | Same, outer compartment | Coronal + sagittal PD-FS | ≥1 cm high-grade cartilage loss[^35^] |
| PF OA | Wear behind the kneecap / trochlear groove | Axial PD-FS | ≥1 cm high-grade cartilage loss[^35^] |
| Effusion | Excess joint fluid; the joint's nonspecific distress signal[^41^] | Sagittal + axial T2-FS/PD-FS | Not separately published; borderline graded negative[^35^] |
| Synovitis | Inflamed, thickened joint lining, often with fluid[^42^] | Sagittal/axial PD-FS (Hoffa's fat pad surrogate) | Not separately published; borderline graded negative[^35^] |
| Baker's | Fluid outpouching into a bursa behind the knee; secondary to intra-articular disease in 87–98% of cases[^43^][^44^] | Axial T2-FS ("speech-bubble" neck) | Not separately published; borderline graded negative[^35^] |
| Contusion | Bone bruise: trabecular microfracture + marrow edema after impaction[^45^] | Any plane, fluid-sensitive (T2-FS/PD-FS/STIR); T1 to characterize | Not separately published; borderline graded negative[^35^] |
| Fracture | Broken bone, often radiographically occult (tibial plateau, patella) | Sagittal/coronal T1 (dark line) + fluid-sensitive edema | Not separately published; borderline graded negative[^35^] |

Three label-interpretation cautions. First, the **meniscal** threshold is the two-image rule from §2.2 — intrameniscal signal that never reaches a surface is common degeneration, not a tear, and the old numeric "grade 1/2/3 signal" reporting scheme is now considered clinically obsolete.[^39^][^46^] Second, **OA** is a whole-joint disease; the competition operationalizes it as high-grade cartilage loss of at least 1 cm in a compartment, which corresponds roughly to the most severe grade (full-thickness loss) of the standard modified Outerbridge cartilage scale.[^47^] The radiographic Kellgren–Lawrence grade you may see in reports cannot be assigned from MRI alone.[^48^] Third, **contusion patterns are diagnostic clues**: a pivot-shift injury leaves bruises on the lateral femoral condyle and posterolateral tibial plateau and strongly implies an ACL tear — so labels are correlated, not independent.[^45^]

## 2.4 How Radiologists Read a Knee MRI — and How Much They Disagree

**The clinical framing.** Most knee MRIs are ordered for "internal derangement of the knee" — an umbrella term for suspected mechanical problems inside the joint (torn meniscus, torn ligaments, loose fragments). MRI is the non-invasive test of choice for this indication and has largely replaced diagnostic arthroscopy (camera surgery into the joint).[^49^]

**The search pattern.** Radiologists do not free-browse; they sweep a fixed checklist so nothing is missed:[^50^] (1) technical quality (motion artifact, failed fat suppression can fake or hide edema); (2) fluid (effusion, Baker cyst); (3) bone marrow (edema patterns map the injury mechanism); (4) ligaments (cruciates on sagittal, collaterals on coronal); (5) menisci on sagittal plus coronal, applying the two-image rule; (6) cartilage on all six articular surfaces; (7) tendons and extensor mechanism; (8) synovium and fat pads; (9) peripheral "don't-forget" areas. The report then follows a fixed skeleton — clinical indication, technique, systematic Findings, and a numbered Impression — and a survey found 47% of referring clinicians skip straight to the Impression.[^51^] Structured-report templates itemize exactly the compartments you see in the label taxonomy (fluid, each meniscus, each ligament, each cartilage compartment).[^52^] The modeling takeaway: the radiologist's checklist is effectively a multi-task classifier over anatomically indexed regions, and each abnormality class has a characteristic (plane, sequence, region) where its signal-to-noise is highest.

**Inter-reader agreement — why your labels are noisy.** Agreement between readers is measured with **Cohen's kappa**, a chance-corrected statistic where 1.0 is perfect agreement and 0 is chance. The conventional interpretation bands are: 0.0–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.0 almost perfect.[^53^] Representative values for knee MRI:

| Finding | Kappa | Context | Band |
|---|---|---|---|
| ACL tear | 0.94–0.98 | Two readers, standard and 3D protocols[^54^] | Almost perfect |
| Acute ACL injury | 0.89–0.93 | Two radiologists[^55^] | Almost perfect |
| ACL (MRI vs arthroscopy) | 0.75 | Proxy for ground-truth noise[^56^] | Substantial |
| Medial meniscus | 0.91 reader / 0.60 arthroscopy | [^54^][^56^] | Substantial–almost perfect |
| Lateral meniscus | 0.89 reader / 0.35 arthroscopy | [^54^][^56^] | Weak–almost perfect |
| Cartilage | 0.84 best / 0.03–0.32 arthroscopy | [^54^][^56^] | None–almost perfect |
| Tibial cartilage area (MOAKS) | 0.36 | OA whole-joint scoring[^57^] | Fair |
| Hoffa-synovitis (MOAKS, intra-rater) | 0.42 | Same expert re-reading[^57^] | Moderate |
| Tibial osteophytes (MOAKS) | 0.49 | OA scoring[^57^] | Moderate |

The pattern is consistent: **agreement is best for ACL tears (κ ≈ 0.75–0.98) and worst for cartilage grading (κ ≈ 0.03–0.84 depending on reference standard), with synovitis grading in between (κ ≈ 0.42 intra-rater).**[^54^][^56^][^57^] Even the same expert re-reading Hoffa-synovitis only reaches moderate agreement. Expect label noise to be lowest for ACL and gross meniscal tears and highest for cartilage, synovitis, and lateral meniscus — an argument for soft or ordinal targets on graded findings, and for treating the twelve label AUCs as having very different achievable ceilings.

**A calibration anchor.** The methodological ancestor of this competition is Stanford's MRNet (Bien et al., 2018): a deep learning model trained on 1,370 knee MRIs achieved AUC 0.965 for ACL tears, 0.847 for meniscal tears, and 0.937 for general abnormality detection, with external-validation ACL accuracy (82.4%) comparable to radiologists.[^58^] Those numbers, ranked in the same order as the kappa table above, are not a coincidence: model ceilings track human label reliability.

## 2.5 Glossary and Self-Study Resources

**Glossary.** Terms appear roughly in the order you will meet them in reports and labels.

| Term | Plain-language meaning |
|---|---|
| Condyle | Rounded knob at the end of the femur that rolls on the tibia |
| Tibial plateau | Flat top of the shin bone that carries the menisci |
| Intercondylar notch | Groove between the femoral condyles housing the cruciate ligaments |
| Trochlea | Femoral groove the kneecap glides in |
| Meniscus (medial/lateral) | C-shaped fibrocartilage shock absorber; parts: anterior horn, body, posterior horn, roots |
| Fibrocartilage | Tough, rubbery cartilage (menisci), vs the smooth hyaline cartilage coating bone ends |
| Cruciate ligaments (ACL/PCL) | Crossing central ropes limiting forward/backward sliding of the tibia |
| Collateral ligaments (MCL/LCL) | Side ropes resisting inward/outward buckling |
| Extensor mechanism | Quadriceps tendon → patella → patellar tendon; straightens the knee |
| Hoffa's fat pad | Fat cushion behind the patellar tendon; a standard site for grading synovitis |
| Bursa | Small fluid sac reducing friction, like bubble wrap |
| Synovium / synovial fluid | Joint lining and the lubricant it secretes |
| Internal derangement | Umbrella term for structural problems inside the joint (tears, loose bodies) |
| Sprain / partial / complete tear | Ligament injury severity grades 1–3 |
| Avulsion | Ligament or tendon pulling off its bony attachment, sometimes with a bone chip |
| Tear orientation | Horizontal (cleavage), longitudinal vertical, radial, oblique, complex — the cleft's geometry |
| Bucket-handle tear | Displaced longitudinal meniscal tear flipped into the notch; can lock the knee |
| Meniscal extrusion | Meniscus bulging ≥3 mm past the tibial edge; marker of root tear and OA |
| Parameniscal cyst | Fluid cyst at the meniscal rim; implies an underlying tear |
| Chondromalacia | Cartilage softening/degeneration, graded I–IV (modified Outerbridge) |
| Osteochondral lesion | Damage involving cartilage plus the bone beneath it |
| Loose body | Free fragment of cartilage or bone floating in the joint ("joint mouse") |
| Bone marrow lesion (BML) / contusion | Ill-defined marrow signal change (bright on fluid-sensitive, dark on T1) from bruising or overload |
| Effusion | Excess joint fluid; hemarthrosis = blood in the joint |
| Synovitis | Inflamed, thickened joint lining; definitively diagnosed on contrast-enhanced MRI |
| Baker (popliteal) cyst | Joint fluid squeezed into a bursa behind the knee; rupture mimics a leg blood clot |
| Osteophyte | Bone spur at joint margins; hallmark of OA |
| Subchondral | "Beneath the cartilage" — where BMLs, cysts, and sclerosis of OA occur |
| Kellgren–Lawrence (KL) grade | Radiographic OA severity scale 0–4; not assignable from MRI alone |
| WORMS / BLOKS / MOAKS | Research scoring systems that grade the whole knee on MRI, feature by feature |
| Sagittal / coronal / axial | The three orthogonal imaging planes (side / front / top-down slices) |
| T1W / T2W / PD | Contrast weightings: anatomy / fluid-bright / high-detail intermediate |
| Fat suppression (FS), STIR | Techniques that darken fat so fluid and edema light up |
| Hyperintense / hypointense | Brighter / darker than the reference tissue on a given sequence |

**Self-study resources, in recommended order.**

1. **Chien et al., "Magnetic resonance imaging of the knee," Polish Journal of Radiology 2020** — the single best primer: protocol tables, per-structure MRI sections, and the two-image PPV rule.[^27^]
2. **Radiopaedia** (free, peer-reviewed articles plus real cases): start with "Knee joint,"[^24^] "Anterior cruciate ligament tear,"[^37^] and "Baker cyst."[^43^]
3. **Chana-Rodríguez et al., "Reporting knee meniscal tears," Insights into Imaging 2016** — tear taxonomy, pitfalls, and why old signal-grading is obsolete.[^39^]
4. **Kohn et al., "Classifications in Brief: Kellgren–Lawrence," 2016** — the OA radiographic scale, its history, and its reliability limits.[^59^]
5. **Hunter et al., "MOAKS," Osteoarthritis and Cartilage 2011** — the current MRI whole-joint OA scoring standard; its reliability tables tell you which findings humans score consistently.[^57^]
6. **Bien et al., MRNet (2018)** — read for the task formulation and baseline performance of deep learning on exactly this data type.[^58^]

# 3. Seed Research Library

This chapter is an annotated bibliography for a machine-learning practitioner with no radiology background entering the RSNA Knee Abnormality Detection competition (knee MRI, 12 binary exam-level labels scored by macro ROC AUC; only 58 of 4,407 training studies carry expert image-derived labels; all 4,407 have free-text reports in roughly ten languages).[^2^] Every entry states what the paper did, key results with numbers, and why it matters for this task. Nothing here is aspirational: each paper supplies a verified architecture pattern, a verified label-extraction method, or a verified public asset.

## 3.1 How to Use This Library

The competition is two problems bolted together: (1) weak-label learning — deriving training labels from ~4,349 multilingual reports that agree with image-derived gold labels only ~82.5% of the time[^60^] — and (2) volumetric image classification, scored image-only because test.csv has no Report column.[^61^] Read in that order: label quality is the largest controllable lever (LLM-extracted labels scored 0.878 vs 0.814 macro AUC for regex extraction against the 58 gold studies[^60^]), while architecture choices among strong published methods span only ~0.01–0.03 AUC.

If you read only ten papers, read them in this order:

| Rank | Paper | One-line reason |
|---|---|---|
| 1 | Bien et al. 2018 (MRNet, PLoS Med) | The task definition, dataset format, and baseline every solution builds on. |
| 2 | Qiu et al. 2024 (CoPAS, Nat Commun) | The exact 12-abnormality knee problem with open-source architecture and loss design. |
| 3 | Irvin et al. 2019 (CheXpert, AAAI) | The canonical template for turning free-text reports into training labels. |
| 4 | Smit et al. 2020 (CheXbert, EMNLP) | The proven recipe: weak labels at scale + tiny expert set ≈ radiologist-level labels. |
| 5 | Tsai et al. 2020 (ELNet, MIDL) | How a 0.2M-parameter from-scratch net beats MRNet's 183M — the efficiency-track blueprint. |
| 6 | Astuto et al. 2021 (Radiol: AI) | Multi-tissue detection/grading system design: ROI segmentation → per-tissue 3D CNN banks. |
| 7 | Belton et al. 2021 (MPFuseNet) | Why learned attention plane-fusion beats MRNet's logistic-regression stacking. |
| 8 | Liu et al. 2019 + Chang et al. 2019 (read together) | The localize-then-classify cascade and 2.5D adjacent-slice inputs for focal lesions. |
| 9 | Atito et al. 2022 (SB-SSL) | Self-supervised slice transformers reaching AUC 0.954 with <1,000 labels — your exact label regime. |
| 10 | Han et al. 2025 (MLFANet-SA, Med Phys) | Current MRNet ACL SOTA (0.981); Top-K slice pooling and slice alignment as final polish. |

On the text side, add Wollek et al. 2025 (§3.4) right after CheXbert: the only published end-to-end demonstration that labels automatically extracted from non-English reports can train a *better* image classifier than manual labels.

## 3.2 The Knee-MRI Deep-Learning Canon

This subsection traces the architecture lineage from MRNet (2018) to the 2025 state of the art. The convergent recipe — per-plane 2D/2.5D pretrained CNN → slice aggregation (attention or Top-K pooling) → multi-head binary classifier — is what to implement first; end-to-end 3D and large ViTs consistently underperform at this data scale.

**[1] Bien N, Rajpurkar P, Ball RL, et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet." *PLoS Med.* 2018;15(11):e1002699 (https://doi.org/10.1371/journal.pmed.1002699; PMC6258509)** — The foundational paper: per-slice AlexNet features, element-wise max-pooling over slices per plane, logistic regression stacking across sagittal/coronal/axial planes; labels manually extracted from clinical reports (the same noisy-label provenance as this competition). Key results: internal-validation AUC 0.937 (abnormal) / 0.965 (ACL) / 0.847 (meniscus); zero-shot external validation on the Rijeka dataset dropped to 0.824, recovering to 0.911 after fine-tuning. Why it matters: defines the data format (3 series/exam, 17–61 slices, patient-disjoint splits) and the external-validation penalty (~0.07–0.14 AUC) to expect on the hidden test set.

**[2] Azcona D, McGuinness K, Smeaton AF. "A comparative study of existing and new deep learning methods for detecting knee injuries using the MRNet dataset." IDSTA 2020 (https://arxiv.org/abs/2010.01947)** — Swaps MRNet's AlexNet for ResNet-18/50/152 with photometric augmentation and horizontal flips. Key results: validation AUC up to 0.96 ACL / 0.91 meniscus / 0.94 abnormal (ResNet-18); per-plane, per-task augmentation-probability grid search was the biggest single contributor. Why it matters: this is your starter code, and its negative-transfer finding (joint multi-plane or multi-task training hurt) is a caution for 12-head designs.

**[3] Tsai CH, Kiryati N, Konen E, Eshed I, Mayer A. "Knee Injury Detection using MRI with Efficiently-Layered Network (ELNet)." MIDL 2020 (https://arxiv.org/abs/2005.02706; code: github.com/mxtsai/ELNet)** — A ~0.2M-parameter CNN trained from scratch on a *single* plane, using multi-slice normalization and BlurPool downsampling. Key results: MRNet validation AUC 0.960 ACL / 0.904 meniscus / 0.941 abnormal — beating MRNet (~183M parameters, three planes, pretrained) on all three, including meniscus 0.904 vs 0.826. Why it matters: architecture design plus pathology-specific plane selection (coronal for meniscus, axial for ACL) beats scale at ~1k-exam sizes; the blueprint for the $18k efficiency track.[^6^]

**[4] Belton N, et al. "Optimising Knee Injury Detection with Spatial Attention and Validating Localisation Ability." MIUA 2021 (https://arxiv.org/abs/2108.08136)** — Adds spatial attention to a ResNet-18 slice encoder and replaces MRNet's logistic-regression plane stacking with learned multi-plane fusion (MPFuseNet). Key results: MRNet test AUC 0.977 ACL / 0.957 abnormal / 0.831 meniscus; logistic-regression fusion "can be detrimental" to ACL/meniscus performance. Why it matters: answers how to fuse sagittal/coronal/axial predictions — learned attention fusion for tear-type labels.

**[5] Dai Y, Gao Y, Liu F. "TransMed: Transformers Advance Multi-Modal Medical Image Classification." *Diagnostics* 2021;11(8):1384 (https://doi.org/10.3390/diagnostics11081384)** — First CNN+ViT hybrid for knee MRI, treating the three MRI planes as multiple modalities fused by a transformer branch alongside a CNN branch. Key results: MRNet AUC 0.98 ACL / 0.976 abnormal / 0.95 meniscus (accuracies 94.9/91.8/85.3%), the largest single published jump over MRNet. Why it matters: the architectural template for cross-sequence (and, by extension, cross-modal) transformer fusion — though note meniscus remains its weakest task (0.95 vs 0.98 ACL), reinforcing that hybrids are not free wins.

**[6] Atito S, Anwar SM, Awais M, Kittler J. "SB-SSL: Slice-Based Self-Supervised Transformers for Knee Abnormality Classification from MRI." MICCAI Workshop 2022 (https://arxiv.org/abs/2208.13923)** — ViT per slice → transformer over slice embeddings → [CLS] exam classifier, with self-supervised pretraining and no external data. Key results: with <1,000 labeled cases, ACL AUC 0.954 / accuracy 89.17% on MRNet, beating supervised SOTA. Why it matters: your label regime exactly — when silver labels are noisy and gold labels number 58, SSL pretraining on the ~570 GB of unlabeled pixels is the high-upside play, and the [CLS]-per-label extension to 12 heads is natural.

**[7] Han S, et al. "Anterior cruciate ligament injuries diagnosis using slice-aligning and multi-level feature aggregation." *Med Phys.* 2025;52(11):e70130 (https://pubmed.ncbi.nlm.nih.gov/41206350/)** — MLFANet-SA: a slice-aligning module that identifies and unifies diagnostic slice regions, plus multi-level feature aggregation via channel-wise Top-K pooling and cross-slice fusion, with no ROI/segmentation labels. Key results: MRNet ACL AUC 0.981, accuracy 0.949, MCC 0.892 — the 2025 published SOTA. Why it matters: channel-wise Top-K pooling over slices is a near-free drop-in upgrade over MRNet's max-pool, and slice alignment removes non-informative slices before aggregation.

**[8] Qiu Z, Xie Z, Lin H, et al. "Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities." *Nat Commun.* 2024;15:7540 (https://www.nature.com/articles/s41467-024-51888-4; code: github.com/zqiuak/CoPAS)** — The single most competition-relevant paper: multi-task diagnosis of **12 knee abnormality types** from multi-plane, multi-sequence MRI (1,748 patients, 5 centers, arthroscopy-referenced), via weight-shared 3D ResNet-18 plane branches, cross-plane/cross-sequence attention, and a learned abnormality×plane probability-matrix fusion; focal loss on final heads, BCE on branches. Key results: average AUC 0.812 internal, 0.721/0.726 on two external sets; beats adapted MRNet/ELNet/MPFuseNet in 8/12 classes; matches senior radiologists (avg accuracy 0.78 vs 0.80). Why it matters: label taxonomy, plane-preference matrix, and loss design are directly transplantable; the ~0.09 internal→external AUC drop is the honest generalization gap to budget for.[^62^]

**[9] Astuto B, Flament I, Namiri NK, et al. "Automatic Deep Learning–assisted Detection and Grading of Abnormalities in Knee MRI Studies." *Radiol Artif Intell.* 2021;3(3):e200165 (https://doi.org/10.1148/ryai.2021200165)** — The closest published analogue to the competition's multi-abnormality output: 17 3D CNNs on V-Net-segmented ROIs detecting/grading cartilage, bone-marrow-edema, meniscus, and ACL lesions in 1,435 exams. Key results: sensitivity 70–88%, specificity 85–89%, AUC 0.83–0.93 across tissues; AI assistance improved intergrader κ in 10/16 comparisons. Why it matters: the full system design — auto-segmentation → volumetric bounding boxes → per-tissue classifier banks — that RSNA winners independently converged on (§3.6).

**[10] Norman B, Pedoia V, Majumdar S. "Use of 2D U-Net Convolutional Neural Networks for Automated Cartilage and Meniscus Segmentation of Knee MR Imaging Data." *Radiology* 2018;288(1):177–185 (https://doi.org/10.1148/radiol.2018172322)** — Compartment-wise 2D U-Nets for cartilage and meniscus. Key results: Dice 0.770–0.878 (cartilage), 0.809 lateral / 0.753 medial meniscus, ~5 s per exam. Why it matters: the reference for an auxiliary segmentation branch whose masks serve as attention maps, ROI crops, or extra channels — the trick that won RSNA 2023–2025 (§3.6).

**[11] Liu F, Guan B, Zhou Z, et al. "Fully Automated Diagnosis of Anterior Cruciate Ligament Tears on Knee MR Images by Using Deep Learning." *Radiol Artif Intell.* 2019;1(3):180091 (https://doi.org/10.1148/ryai.2019180091); and Chang PD, Wong TT, Rasiej MJ. "Deep Learning for Detection of Complete Anterior Cruciate Ligament Tear." *J Digit Imaging.* 2019;32:980–986 (https://doi.org/10.1007/s10278-019-00193-4)** — Read together: Liu's cascaded detect-then-classify system (two CNNs isolating the ACL, then a classifier) reached AUC 0.98, sens/spec 0.96/0.96 against arthroscopy, indistinguishable from five radiologists with only n=350 training cases; Chang's five-slice 2.5D input with dynamic patch sampling reached test AUC 0.971, beating three-slice (0.865) and single-slice (0.765) inputs. Why they matter: localize-then-classify plus 2.5D slice stacking is the most reliable published trick for small structures (ACL, MCL, menisci); the slice-count ablation is the cleanest 2.5D evidence in the literature.

**[12] Schiratti JB, Dubois R, Herent P, et al. "A deep learning method for predicting knee osteoarthritis radiographic progression from MRI." *Arthritis Res Ther.* 2021;23:262 (https://doi.org/10.1186/s13075-021-02634-4)** — Weakly supervised CNN on 9,280 OAI knee MRIs (3,268 patients) predicting 12-month progression from exam-level labels only. Key results: AUC 0.65 vs radiologists' 0.587; WOMAC pain AUC 0.72. Why it matters: proof that exam-level weak labels on ~9k knee MRIs suffice to beat expert readers — this competition's exact supervision regime — with Grad-CAM saliency as a free localization sanity check.

**[13] Panfilov E, Tiulpin A, Nieminen MT, Saarakkala S. "End-To-End Prediction of Knee Osteoarthritis Progression With Multi-Modal Transformers." arXiv:2307.00873 (2023; https://arxiv.org/abs/2307.00873)** — Modality-specific encoders with transformer fusion over X-ray, structural/compositional MRI, and clinical data from OAI (n=2,421–3,967); public code and weights. Key results: ROC AUC 0.70–0.76 across 2–8-year progression horizons; 1-year best with full multimodal fusion (0.76). Why it matters: the most mature published recipe for fusing knee imaging with non-image data via transformers — the closest template for MRI + report-derived features, since no knee MRI+report fusion paper exists (an open opportunity, not an oversight).

A caution flag: a 2025 paper ("KneeXNet," PMC12088959) claims MRNet test AUCs of 0.985/0.972/0.968 via graph convolutions and contrastive learning; the near-perfect metrics in a non-top-tier venue warrant skepticism (possible test-set leakage), so treat it only as an existence proof of graph-based approaches.

## 3.3 Learning from Radiology Reports: Label Extraction and Weak Supervision

Because test-time reports do not exist,[^61^] reports are a train-time teacher: the winning frame is a text-supervised image student. The literature splits into label extraction from reports (CheXpert lineage → LLMs), learning images directly from paired reports (vision-language pretraining), and training robustly on the resulting noisy labels. The host has ruled that commercial LLM APIs may be used for label extraction during development,[^14^] but everything below also works offline.

**[14] Irvin J, Rajpurkar P, Ko M, et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI 2019 (https://arxiv.org/abs/1901.07031)** — The canonical three-stage rule labeler (mention extraction → negation/uncertainty classification → aggregation, positive > uncertain > negative) producing 14 silver labels from free text. Key results: micro-F1 0.969 mention / 0.952 negation / 0.848 uncertainty. Why it matters: the 3–4-state scheme (positive/negative/uncertain/unmentioned) is the correct output schema here — 25.4% of competition report cells are "not addressed," and silence is per-finding informative (Baker's silence ≈ negative; synovitis silence ≈ uninformative);[^60^] the U-Ignore/U-Zeros/U-Ones policies are the playbook for ambiguous labels.

**[15] Peng Y, Wang X, Lu L, et al. "NegBio: a high-performance tool for negation and uncertainty detection in radiology reports." AMIA Jt Summits Transl Sci Proc. 2018:188–196 (https://pmc.ncbi.nlm.nih.gov/articles/PMC5961822/)** — Negation/uncertainty detection via patterns over universal-dependency graphs rather than surface regex windows. Key results: +9.5% precision / +5.1% F1 on average over NegEx. Why it matters: negation scoping is the largest error source in report labels, and dependency-graph scopes generalize better than word windows for the long, verb-less noun phrases of non-English report styles; on competition data, adding sentence-scope negation to a keyword labeler moved macro AUC 0.638 → 0.667.[^61^]

**[16] Smit A, Jain S, Rajpurkar P, Pareek A, Ng A, Lungren M. "Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT (CheXbert)." EMNLP 2020 (https://aclanthology.org/2020.emnlp-main.117/)** — A BioClinicalBERT labeler pretrained on rule-labeler outputs, then fine-tuned on a small expert set augmented by back-translation. Key results: +0.055 F1 (95% CI 0.039–0.070) over the CheXpert rule labeler; within 0.007 F1 of the radiologist benchmark; expert-labels-only training underperforms the two-stage recipe. Why it matters: the competition's central recipe — weak labels at scale plus a tiny hand-checked set — with back-translation especially valuable across ~10 languages.

**[17] Jain S, Agrawal A, Saporta A, et al. "RadGraph: Extracting Clinical Entities and Relations from Radiology Reports." CHIL 2021 (https://arxiv.org/abs/2106.14463)** — A schema and DYGIE++ benchmark for extracting (anatomy, observation, certainty) entities and relations. Key results: micro-F1 0.94 entity / 0.82 relation extraction (MIMIC-CXR) vs human 0.99/0.95; "Observation: Uncertain" is the hardest class. Why it matters: parsing knee reports into (medial meniscus, tear, definitely present) triples resolves contradictory mentions before label assignment and provides multi-task auxiliary targets.

**[18] Dorfner FJ, Jürgensen L, Donle L, et al. "Is Open-Source There Yet? A Comparative Study on Commercial and Open-Source LLMs in Their Ability to Label Chest X-Ray Reports." *J Digit Imaging* 2024 (https://arxiv.org/abs/2402.12298)** — Benchmarks GPT-4 vs open 70B-class models for zero-/few-shot structured report labeling. Key results: GPT-4 micro-F1 0.975–0.984; Llama2-70B 0.970–0.972 and Qwen1.5-72B 0.952–0.965 reach near-parity few-shot. Why it matters: offline open-weight LLMs can regenerate or audit competition labels at ~0.95+ micro-F1 with no API dependency; ensemble 2–3 with majority vote. A 2025 companion study (J Digit Imaging, PMC12920854) adds that open LLMs beat the CheXpert labeler vs human annotations (95% vs 51% sensitivity on rib fracture) and that image classifiers trained on noisier labels retained most performance on clean evaluation — invest in a clean validation set, not perfectly clean training labels.

**[19] Zhang Y, Jiang H, Miura Y, Manning CD, Langlotz CP. "Contrastive Learning of Medical Visual Representations from Paired Images and Text (ConVIRT)." MLHC 2022 (https://arxiv.org/abs/2010.00747)** — The foundational proof that reports are supervision: bidirectional image–sentence InfoNCE on naturally paired studies. Key results: with 1% of labels, AUC 90.7 on an RSNA task vs 82.8 ImageNet init / 86.6 MoCo v2. Why it matters: image–report contrastive pretraining on the competition's own pairs is legal, needs no label extraction, and targets exactly this low-label regime.

**[20] Huang SC, Shen L, Lungren MP, Yeung S. "GLoRIA." ICCV 2021 (code+weights: github.com/marshuang80/gloria); Boecking B, et al. "BioViL/CXR-BERT." ECCV 2022 (https://arxiv.org/abs/2204.09817; HuggingFace weights); Bannur S, et al. "BioViL-T." CVPR 2023 (weights: microsoft/BiomedVLP-BioViL-T)** — The global–local lineage: GLoRIA adds image-region ↔ word attention (free localization for focal lesions); BioViL shows text-side modeling (radiology vocabulary, section-aware objectives) is where the gains are; BioViL-T's CNN–Transformer multi-image encoder is robust to missing inputs. Why they matter: local alignment suits focal knee findings; BioViL-T's multi-image encoder maps onto multi-series/multi-plane knee exams with missing-series dropout. Weights are public but English-only — the recipe, not the weights, transfers to the multilingual corpus.

**[21] Wang Z, Wu Z, Agarwal D, Sun J. "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text." EMNLP 2022 (https://arxiv.org/abs/2210.10163; code public)** — Replaces one-to-one InfoNCE pairing with entity-based soft semantic-similarity targets, eliminating false negatives. Key results: beats GLoRIA with ~1/10 the pretraining data (20K vs ~200K); >10% average accuracy gains zero-shot and supervised. Why it matters: many knees share findings, so pair-based contrastive learning wastes supervision; soft similarity targets derived from extracted labels both scale and denoise.

**[22] Tiu E, Talius E, Patel P, et al. "CheXzero: Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning." *Nat Biomed Eng.* 2022;6:1399–1406 (https://pubmed.ncbi.nlm.nih.gov/36109605/; checkpoints public)** — CLIP-style training on raw report impressions with no extracted labels at all. Key results: zero-shot mean AUC 0.889 on CheXpert (vs supervised DenseNet-121 0.902); not significantly different from board-certified radiologists on 5 pathologies. Why it matters: the strongest evidence that raw reports alone can supervise an image model to near-expert level — a direct template here, given a suitable multilingual text encoder.

**[23] Zhang S, Xu Y, Usuyama N, et al. "BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs." arXiv:2303.00915 (weights: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, Apache-2.0)** — CLIP-style ViT-B/16 + PubMedBERT pretrained on PMC-15M. Key results: SOTA across biomedical retrieval, classification, and VQA, beating radiology-specific BioViL on RSNA pneumonia detection. Why it matters: Apache-2.0 makes it unambiguously Kaggle-legal as an image-encoder init or report-embedding backbone, subject to external-data rules.

**[24] Mai Tan H, et al. "LumbarCLIP." arXiv:2509.20813 (2025); and Yu K, Wang D, Yuan Z, et al. "OrthoFoundation." GitHub 2026 (github.com/ytrsk/OrthoFoundation)** — The two musculoskeletal-specific foundation efforts. LumbarCLIP aligns lumbar MRI with report text (95.0% accuracy / 94.75% F1 downstream; linear projection heads beat non-linear — a copyable ablation). OrthoFoundation is a DINO-style vision model pretrained on 1.25M knee images (357,670 radiographs + 893,985 MRI slices), weights released, evaluated on 17 MSK tasks. Why they matter: LumbarCLIP proves MRI+report contrastive pretraining works in MSK imaging; OrthoFoundation-L is the most on-target pretrained backbone available — knee MRI is its home domain (verify license/rules; single-source evidence).

**[25] Ratner A, Bach SH, Ehrenberg H, et al. "Snorkel: Rapid Training Data Creation with Weak Supervision." PVLDB 2018 (https://arxiv.org/abs/1711.10160)** — Data programming: combine multiple noisy labeling functions via a generative model that estimates each source's accuracy and correlations without ground truth. Key results: within 3.6% of models trained on large hand-labeled sets; generative weighting adds 5.81% over majority vote. Why it matters: the competition gives one noisy label source; Snorkel says *create more* (rule labeler, LLM ensemble, zero-shot image model, per-language keyword lists) and train on the learned soft consensus.

**[26] Han B, et al. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." NeurIPS 2018 (https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html); and Li J, Socher R, Hoi SCH. "DivideMix." ICLR 2020 (https://arxiv.org/abs/2002.07394)** — The two workhorse noisy-label algorithms: co-teaching trains two networks exchanging small-loss (likely clean) samples; DivideMix fits a Gaussian mixture on per-sample losses, then treats noisy samples as unlabeled in a semi-supervised loop. Key results: DivideMix 93.4% vs cross-entropy 85.0% on CIFAR-10 at 40% asymmetric noise. Why they matter: co-teaching is nearly free; DivideMix's loss-GMM operationalizes "LLM–image-model disagreement flags prime noise suspects."

**[27] Karimi D, et al. "Deep learning with noisy labels: exploring techniques and remedies in medical image analysis." *IEEE TMI* 2020 (https://arxiv.org/abs/1912.02911); and Gündel S, et al. "Robust Classification from Noisy Labels." arXiv:2104.05261 (2021)** — The evidence base for cheap remedies: Karimi documents label smoothing improving AUC by up to 0.08 on report-derived chest X-ray labels; Gündel injected per-class label-error priors (measured on a radiologist re-read subset) into the loss, reaching avg AUC 0.880 over 17 abnormalities on 297,541 radiographs. Why they matter: start with label smoothing (ε≈0.05–0.1) — the cheapest evidence-backed gain — and approximate the re-read priors with the 58 gold studies or an LLM ensemble. Asymmetric loss and soft/graded targets (community-reported +0.056 macro AUC over binary here[^63^]) belong to the same toolbox.

## 3.4 Multilingual Radiology NLP

The reports span roughly ten languages (English, Spanish, Dutch dominant; French, German, Portuguese, Italian, Turkish in the tail; Bulgarian and Greek observed).[^64^] Two strategic facts: single multilingual models beat per-language models on report tasks (arXiv:2310.00100), and translate-to-English is lossy — GPT-4o translation of radiology reports averaged only 79% factual correctness, with potentially harmful errors in 4% (Eur Radiol Exp 2025, S0720048X25004279). Prefer natively multilingual extraction.

**[28] Conneau A, et al. "Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)." ACL 2020 (https://arxiv.org/abs/1911.02116)** — A single encoder pretrained on 100 languages (2.5 TB CommonCrawl). Key results: 80.9% average XNLI accuracy zero-shot cross-lingual; 83.6% translate-train-all. Why it matters: XLM-R(-large) is the default student for distilling LLM silver labels into a fast multilingual report classifier that labels the full corpus at BERT speed.

**[29] Wang L, et al. multilingual-E5 (2024); and Chen J, et al. "BGE-M3." 2024 (descriptions via arXiv:2510.23896)** — Multilingual embedding models for similarity-based label assignment: mE5-large-instruct (XLM-R-large backbone, ~100 languages, 1024-dim, contrastive pretraining on ~1B pairs; best in the initial MMTEB release) and BGE-M3 (100+ languages, 8,192-token context, dense+sparse+multi-vector, MIT license). Why they matter: embeddings + logistic heads are a cheap, strong alternative to full fine-tuning; BGE-M3's 8k context fits full reports untruncated.

**[30] NLLB Team. "No Language Left Behind (NLLB-200)." 2022 (model card: huggingface.co/facebook/nllb-200-distilled-1.3B)** — 200-language translation; distilled 600M/1.3B variants run offline on Kaggle; CC-BY-NC license. Key caveat (verbatim model card): "not intended to be used with domain specific texts, such as medical domain" and inputs ≤512 tokens. Why it matters: use translate-to-English only as a third voter for cross-checking ensemble disagreement, never as the primary extraction path.

**[31] Qiu P, et al. "Towards Building Multilingual Language Model for Medicine (MMedC/MMedBench)." 2024 (https://arxiv.org/abs/2402.13963)** — A 25.5B-token multilingual medical corpus (EN/ZH/JA/FR/RU/ES) and 8,518-pair benchmark; MMedLM 2 (7B) rivaled GPT-4 on MMedBench. Why it matters: medical-domain multilingual pretraining closes most of the gap to GPT-4 at 7B scale — but its language cover overlaps the competition's only partially (Spanish/French yes; Dutch, Turkish, Bulgarian, Greek no), so continued MLM pretraining on the competition's own reports remains necessary.

**[32] Chapman BE, et al. "Extending the NegEx Lexicon for Multiple Languages." MEDINFO 2013 (https://pmc.ncbi.nlm.nih.gov/articles/PMC3923890/); and Stricker V, Iacobacci L, Cotik V. "SpRadNeg." IJCAI WS 2016 (staff.dc.uba.ar/vcotik/docs/papers/NegatedFindingsDetectionIJCAISWS.pdf)** — Rule-based negation ports: French recall 85%/precision 89%, Swedish 82%/75%, German translated but unevaluated; the Spanish radiology adaptation SpRadNeg reached precision 0.87 but recall only 0.49. Why they matter: per-language negation triggers are mandatory (naive keyword matching without them costs −0.03 to −0.09 macro AUC here[^61^]); SpRadNeg's recall collapse is the documented failure mode of naive trigger porting — validate each language's rules against hand-checked reports.

**[33] Wollek A, Hyska S, et al. "German CheXpert Chest X-ray Radiology Report Labeler." *Fortschr Röntgenstr* 2024 (https://pubmed.ncbi.nlm.nih.gov/38295825/); and Wollek A, Haitzer P, et al. "Language model-based labeling of German thoracic radiology reports." *Fortschr Röntgenstr* 2025 (https://pubmed.ncbi.nlm.nih.gov/38663428/; code: gitlab.lrz.de/IP/german-lm-radiology-report-labeler)** — The most relevant multilingual pipeline published: a rule-based German CheXpert port (mention F1 up to 0.995; per-observation positive/negative/uncertain phrase files handle implicit negation), then a German-BERT labeler weak-pretrained on 66k rule-labeled reports and fine-tuned on ~1k manual labels. Key results (2025): the DL labeler beat rules on all tasks (mention F1 0.938 vs 0.844; negation 0.891 vs 0.821; uncertainty 0.624 vs 0.518), and a DenseNet-121 trained on DL labels (AUC 0.939) **beat one trained on manual labels (0.934)** — consistent automatic labels can outperform sparse manual ones. Why it matters: the per-language recipe for exactly this competition — seed rules → weak-pretrain a language BERT → fine-tune on ≤1k checked reports → label at scale.

**[34] Wan Z, Liu C, Zhang M, et al. "Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias." NeurIPS 2023 (https://arxiv.org/abs/2305.19894; code public)** — Cross-lingual text-alignment regularization unifying English and Spanish report semantics in medical VLP. Key results: SOTA across 5 task types / 10 datasets / 30+ diseases; reducing language-community bias improved even uni-modal visual tasks. Why it matters: naive multilingual training lets language identity contaminate embeddings; use language disentanglement or balanced per-language sampling, and audit per-language performance.

## 3.5 Public Datasets Worth Knowing

External data and pretrained models are allowed if publicly accessible at minimal cost, but the competition's ruling on gated click-through datasets (MRNet, OAI, fastMRI) is **unresolved** as of this writing — treat their eligibility as pending and monitor the Kaggle rules thread.[^13^] Architecture and label-taxonomy reuse (e.g., CoPAS's plane-preference matrix) is safe regardless; weight reuse from models pretrained on these datasets (e.g., OrthoFoundation, BiomedCLIP) is the intermediate-risk path.

| Name | Size | Labels | Access | Competition-usability note |
|---|---|---|---|---|
| MRNet (Stanford; Bien 2018) | 1,370 exams; 3 series/exam (sag T2 FS, cor T1, ax PD FS); splits 1130/120/120 patient-disjoint | 3 binary exam labels (abnormal 80.6%, ACL tear 23.3%, meniscal tear 37.1%), manually extracted from reports | Gated click-through (stanfordmlgroup.github.io/competitions/mrnet/) | De-facto benchmark and on-domain pretraining source; eligibility pending the external-data ruling.[^13^] |
| Rijeka KneeMRI (Štajduhar 2017, doi:10.1016/j.cmpb.2016.12.006) | 917 exams, sagittal PD 1.5T | ACL: 690 healthy / 172 partial / 55 complete | Public download (linked from MRNet page) | Standard external/OOD validation set for MRNet-style models (MRNet zero-shot 0.824 → fine-tuned 0.911). |
| OAI (Osteoarthritis Initiative; Peterfy 2008, doi:10.1016/j.joca.2008.06.016) | ~4,796 participants, longitudinal 3T MRI + clinical data | WORMS/MOAKS semiquantitative scores, KL grades, progression labels | Gated application (nda.nih.gov/oai) | Largest labeled knee-MRI cohort for weak-supervision pretraining (Schiratti 2021 used 9,280 images); eligibility pending.[^13^] |
| OAI-ZIB masks (Ambellan 2019, doi:10.1016/j.media.2018.11.009) | 507 manual bone+cartilage segmentations on OAI | Segmentation masks | Publicly released by authors | Ready-made supervision for the auxiliary segmentation/localization stage (§3.6 trick #1). |
| fastMRI knee (Zbontar 2018, arXiv:1811.08839) | ~1,500+ raw k-space knee volumes (coronal PD/PD-FS) + DICOMs of ~10k clinical exams | None (reconstruction benchmark) | Gated click-through | Best public source of unlabeled knee MRI at scale for SSL pretraining (SB-SSL/BYOL style); eligibility pending.[^13^] |
| SKI10 (MICCAI 2010) | 100 knee MRIs | Bone/cartilage segmentation labels | Public challenge data | Small but usable segmentation pretraining; used by Ambellan 2019. |
| CoPAS 5-center dataset (Qiu 2024) | 1,748 patients, 5 centers; PDW×3 planes + cor T1W + sag T2W | **Same 12-abnormality taxonomy**, arthroscopy + MRI consensus | Restricted academic access on request; **code public** (github.com/zqiuak/CoPAS) | Even if data access fails, its label definitions and abnormality×plane matrix are a design blueprint; clone architecture, not data.[^62^] |

## 3.6 Past RSNA Competition Solutions as Secondary Literature

No past RSNA winner used report text, but the volumetric half of this task is well-trodden: four consecutive RSNA competitions (2022–2025) produced open solution write-ups that function as applied literature. The recurring pattern is unambiguous — two-stage **localize-then-classify** dominates; 2.5D beats 3D under Kaggle constraints; slice fusion is BiLSTM/GRU or attention-MIL; auxiliary localization losses are the most reliable booster.

**[35] Qishen Ha, 1st place, RSNA 2022 Cervical Spine Fracture Detection (write-up: kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution)** — 3D vertebra segmentation trained on only 87 masks → per-vertebra crops → 2.5D CNN (5-channel adjacent-slice stacks + mask channel) + LSTM. Key lessons: end-to-end 3D CNN classification "did not give satisfactory results"; ensembling 6 models added ~+0.02 CV within a 7.5 h Kaggle runtime. Why it matters: a small stage-1 annotation set (87 masks) suffices for localization — the highest-value first investment for knee ROI cropping.

**[36] Team Oxygen, 1st place, RSNA 2023 Abdominal Trauma Detection (code: github.com/Nischaydnk/RSNA-2023-1st-place-solution)** — 3D organ segmentation → crops → 2.5D CNN + GRU, with soft per-slice targets (patient label × organ-visibility curve) and 4-fold patient-level GroupKFold. Key lessons: auxiliary segmentation loss on the shared encoder gave +0.01–0.03; balanced sampling beat weighted BCE ("did not help model converge"). Why it matters: soft slice-level targets and sampling-over-loss-weighting transfer directly to 12 imbalanced knee labels.

**[37] NANACHI, 1st place, RSNA 2024 Lumbar Spine Degenerative Classification (write-up: kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)** — The closest analog (MRI, multi-series, 25 condition-level labels): 3D ConvNeXt slice-position + 2D coordinate localizers → cropped 5-slice stacks → ConvNeXt-small/EfficientNetV2-s + BiLSTM + attention-MIL. Key lessons: attention-MIL improved public LB 0.37→0.35; BiLSTM + aux losses + ensembling → 0.33; jittering predicted coordinates during stage-2 training (matched to measured stage-1 error) was "crucial"; MAMBA, large models, and ViTs all lost to conv nets. Why it matters: the complete, proven recipe for multi-series MRI classification under Kaggle constraints, including stage-2 robustness to stage-1 errors.

**[38] tomoon33, 1st place, and MIC-DKFZ, 7th place, RSNA 2025 Intracranial Aneurysm Detection (code: github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution)** — Two contrasting 2025 patterns on multimodal CT/MR with 14 labels. tomoon33: coarse-to-fine nnU-Net vessel segmentation → ROI classifier with segmentation-pretrained backbone and an auxiliary detection loss weighted 10–20× above classification losses (ablation: segmentation pretraining 0.794→0.902; aux loss +0.026); each label an independent binary problem. MIC-DKFZ: a single nnU-Net blob-regression model (TopK-20% BCE, no TTA, public/private 0.83/0.83), with explicit negative results: isotropic 1-mm resampling "substantially worsened" results and external-data co-training "did not really help." Why they matter: independent binary heads, aux-localization weighting, and the negative results are directly actionable; their CV-over-LB discipline is what this competition's prevalence-mismatched splits demand.[^2^]

**Synthesis for this competition.** Implement the skeleton first: per-series 2.5D ConvNeXt-small/EfficientNetV2-s → BiLSTM or attention-MIL → 12 independent binary heads; add a small segmentation/keypoint stage for meniscus/cartilage/ACL ROIs as the first upgrade; train on soft silver labels with label smoothing and co-teaching; validate only against the 58 gold studies plus scanner-grouped CV, and treat <0.02 macro-AUC deltas as noise.[^60^]

# 4. Modeling and Validation Strategy

## 4.1 The Real Problem: Weak Supervision, Not Architecture

Strip the competition down to its supervisory structure and an uncomfortable fact emerges: of the 4,407 training studies, only 58 (~1.3%) carry the 12 expert image-derived labels; the remaining 4,349 have only free-text radiology reports, and `test.csv` contains no report column at all — reports are a train-time-only teacher, confirmed by the host and by the data page wording ("you may wish to derive the labels for the remaining studies").[^2^][^9^] The model that scores must read pixels alone. This competition is therefore not primarily an image-modeling problem; it is a weak-label learning problem wearing an image competition's clothes.

The numbers make the asymmetry explicit. Measured against the 58 gold studies, LLM-extracted report labels reach 0.8780 macro AUC versus 0.8136 for regex/lexicon extraction — a +0.064 gap from label quality alone.[^4^] By comparison, the entire architectural spread among strong, well-tuned knee-MRI methods documented in the literature and in past RSNA winning solutions is roughly 0.01–0.03 AUC (Section 4.3). Soft/graded targets reportedly add a further +0.056 macro AUC over hard binary targets, and a single targeted imputation of "not addressed" synovitis cells lifted that column's AUC from 0.678 to 0.790.[^4^][^63^] Label engineering is the largest controllable lever in the competition, by an order of magnitude.

Two structural properties of the silver labels must be internalized before any modeling decision. First, reports are silver, not gold: strict textual reading agrees with the provided image-derived labels in only ~82.5% of cells (positive recall 80%), and the host confirmed that image-derived labels — produced by two subspecialty MSK radiologists with a third adjudicator, with borderline findings graded negative — are authoritative where the two disagree.[^9^] Second, 25.4% of all report-label cells are "not addressed," and silence is not symmetric across findings: when a report is silent on Baker's cyst, the finding is present only ~3% of the time (silence ≈ negative), whereas silent synovitis (84% of cells) still carries a ~34% positivity rate (silence ≈ uninformative).[^4^] Coercing all silent cells to 0 is the single most damaging label-pipeline mistake available.

**Proven:** label quality dominates architecture as a score lever; silence semantics are per-finding. **Promising but unverified:** the +0.056 soft-target gain is a community-reported figure from a thread title, plausible but single-source.

## 4.2 Building Silver Labels from Multilingual Reports

Reports arrive in roughly ten languages (EN/ES/NL dominant; FR/DE/PT/IT/TR tail; BG/EL observed) across 16 sites.[^64^] The host has officially ruled that commercial LLM APIs may be used for report label extraction during development, and open-weight models run locally are unconstrained since the internet-off rule applies only to the submission notebook.[^14^] The recommended pipeline, synthesized from community measurements on the 58 gold studies and the multilingual clinical-NLP literature:

| Stage | Component | Evidence / expected quality |
|---|---|---|
| 0. Language ID | fastText `lid.176` per report; per-site priors are strong | Standard; ~10 languages attested[^64^] |
| 1. Section split | Per-language regex for Impression/Conclusión/Beurteilung/Conclusie; weight Impression mentions higher | CheXpert-on-Impression practice[^65^] |
| 2. LLM extraction (primary) | Open-weight multilingual instruct LLM (Qwen3-14B/32B, 119 languages; Gemma-3-27B, 140+; GPT-OSS-20B single-GPU), few-shot strict JSON: one field per finding, 5-way value {present, absent, uncertain, not_addressed, laterality-ambiguous} + evidence span; official positivity thresholds in the system prompt | 0.8780 macro AUC vs gold (measured); anatomy-aware prompting adds up to +0.08 macro-F1 in radiology extraction[^4^][^66^][^67^] |
| 3. "Not addressed" channel | Map `not_addressed` → soft label 0.5, never to 0; impute only per-finding-justified cells (effusion→synovitis: 0.678→0.790 column AUC); no blanket imputation (measured worse: 0.8805 vs 0.8873) | Community-measured on gold[^4^] |
| 4. Rule-labeler fallback | Multilingual CheXpert-style labeler: per-language positive/negative/uncertain phrase files, NegEx-style triggers, OA-consequence vocabulary (osteophytes, joint-space narrowing, chondral loss, "tricompartmental") | 0.727 macro AUC alone (0.638 naive → 0.667 +negation → 0.727 +OA vocabulary); adds precision on fracture/Baker's; deterministic audit trail[^61^] |
| 5. Distillation | Fine-tune XLM-R student on LLM soft labels for a scalable, consistent labeler; multi-LLM voting as a second-order upgrade | Fine-tuned BERT on rule labels adds +3–10 F1; XLM-R covers 100 languages[^68^][^69^] |
| 6. Calibration & audit | Validate every label-pipeline variant exclusively against the 58 gold; treat differences below ±0.02 macro AUC as unmeasurable noise at n=58 | Community-measured noise floor[^4^] |

Translate-to-English (NLLB-200) is a third voter at most: its own model card warns against medical-domain use, and GPT-4o-level translation still loses ~16–21% factual fidelity — acceptable for cross-checking, not as the primary channel.[^70^]

The output of this pipeline is a 4,407 × 12 matrix of soft silver labels in [0,1] plus per-cell confidence. Downstream image training consumes these with confidence-weighted losses, and the 58 gold studies never enter training — they are the only local labels drawn from the same protocol as the hidden test set.

## 4.3 The Architecture Recipe That Keeps Winning

Architecture choice is the *second* problem, and it is largely solved. Three independent lines of evidence — the knee-MRI literature (MRNet → ELNet → CoPAS), four years of RSNA volumetric-competition winners (2022–2025), and MIL theory — converge on the same skeleton: per-plane 2D/2.5D pretrained CNN slices → a slice-sequence aggregator (BiLSTM or gated attention-MIL) → study-level binary heads, optionally preceded by a localize-then-classify stage.[^71^][^72^][^73^]

| Stage | Recommended design | Evidence |
|---|---|---|
| Input handling | Per-series; sort slices by DICOM metadata; equidistant subsample or cardinality-invariant pooling; 2.5D stacks (adjacent slices as 3–6 channels); per-volume z-score or Nyúl intensity standardization | 2022/2023 RSNA 1st-place recipes; ELNet normalization ablation[^72^][^73^][^74^] |
| Slice encoder | ImageNet-pretrained ConvNeXt-small / EfficientNetV2-s / CoAtNet-class 2D CNN; small > large in this regime ("convnext-large < base < small") | 2024 lumbar 1st place; Transfusion (ImageNet benefit is mostly low-layer scaling + convergence speed)[^74^][^75^] |
| Slice aggregation | BiLSTM or gated attention-MIL over per-slice features; aux attention loss | Attention-MIL lifted the 2024 lumbar winner's public LB 0.3729→0.3588 (+0.020 private); max/mean MIL pooling trains unstably[^74^][^76^] |
| Heads | 12 independent binary heads (treat labels as independent tasks; helps under severe per-class imbalance) | 2025 aneurysm 1st place treated 14 labels as independent binaries[^77^] |
| Plane/sequence fusion | Per-plane models + logistic-regression stacking (MRNet template, proven) or CoPAS-style cross-plane/cross-sequence attention (best-published 12-class knee design; public code at github.com/zqiuak/CoPAS is a directly transplantable template); avoid naive concatenation | MRNet AUC 0.937/0.965/0.847; CoPAS 0.812 avg AUC internal, 0.72 external[^71^][^73^] |
| Optional booster | Localize-then-classify ROI stage (meniscus/joint-crop from a small U-Net or keypoint model trained on a tiny annotation set) | Reliable +0.01–0.03 across four consecutive RSNA volumetric competitions; 87 masks sufficed in 2022[^72^][^74^] |

Equally valuable is the negative knowledge — approaches that have repeatedly failed in this exact setting and its closest analogs:

| Approach | Where it failed | Evidence |
|---|---|---|
| End-to-end 3D CNN classification | RSNA 2022 1st place: "this method does not work"; memory forces small backbones | 2022 1st-place write-up[^72^] |
| Large ViTs / large backbones | 2024 lumbar 1st: conv nets > ViTs; convnext-large < base < small; ViT underperforms CNNs on MRNet-scale data | 2024 1st write-up; MRNet systematic study[^74^][^78^] |
| Isotropic resampling to 1 mm³ | RSNA 2025 MIC-DKFZ: "substantially worsened" results, discontinued early | 2025 7th-place write-up[^79^] |
| External-data pseudo-labeling / co-training | 2024 NVSpine: "pseudo labelling on external data" did not help; 2025 MIC-DKFZ: co-training "did not really help" | 2024 6th / 2025 7th write-ups[^79^][^80^] |
| Weighted BCE from scratch | 2023 Team Oxygen: "did not help model converge" under balanced sampling | 2023 1st-place write-up[^81^] |
| Image-level ComBat harmonization | No benefit for DL classifiers under scanner shift (GE/Philips/Siemens study) | Sci Rep 2023[^82^] |

The practical instruction for a solo practitioner: clone the skeleton (CoPAS's public code is the closest legal template), invest the saved architecture-exploration budget in the label pipeline (Section 4.2), and add the localization stage only after the baseline is validated.

## 4.4 Validation You Can Trust

The evaluation layer is deliberately adversarial to naive validation. The public leaderboard is computed on ~30% of ~1,300 test studies; the organizers explicitly warn that abnormality prevalence "is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets"; and the 58 gold studies are pathology-enriched (every study has ≥1 positive finding, mean 4.14 findings/study), so even the gold anchor is prevalence-biased.[^2^][^4^][^10^] Meanwhile, a metadata-only classifier scores 0.652 macro AUC under random folds but 0.598 under scanner-grouped folds — ~0.05 AUC of apparent skill on this dataset is site memorization that will not transfer (largest on OA targets, 0.07–0.09).[^23^] Random-fold CV is therefore optimistically biased by construction.

| Design element | Specification | Rationale / evidence |
|---|---|---|
| Primary split | Study-level GroupKFold (5 folds): all series/slices/augmentations of one StudyInstanceUID in one fold | Patient-level splitting is the leakage standard in radiology ML[^83^] |
| Stratification | Iterative multilabel stratification on the 12 silver labels (scikit-multilearn), combined with grouping (greedy whole-study assignment balancing per-class counts) | Preserves rare-class positives per fold; plain StratifiedKFold cannot handle multilabel[^84^] |
| Secondary view | One scanner-grouped validation (group by Manufacturer × FieldStrength, or finer fingerprint) to estimate site shift | ~0.05 AUC memorization measured on this dataset; MRNet dropped 0.911→0.824 zero-shot cross-site[^71^][^23^] |
| Gold anchor | 58 expert-labeled studies held out of training entirely; every modeling decision (label extractor, imputation, augmentation, loss, architecture) must move gold macro AUC, not just silver CV | Only local labels from the test-time protocol; ±0.02 macro AUC is the noise floor at n=58[^9^][^4^] |
| LB policy | Trust CV + gold over public LB; use ≤1–2 of 5 daily submissions for hypothesis testing; expect shake-up (rare-class AUC SE on ~900 private studies is ±0.03–0.05 per class) | NVSpine 2024: "we trusted our CV more than public LB" — final CV 0.382 / public 0.355 / private 0.401[^80^] |

Two disciplines follow. First, tune hyperparameters on one fold when compute-limited (MIC-DKFZ practice), then confirm on the full CV.[^79^] Second, record per-class OOF AUCs for every model — they are the raw material for Section 4.5's per-class ensembling.

## 4.5 Metric Optimization, Ensembling, and TTA

The metric is unweighted macro-averaged ROC AUC over the 12 exam-level binaries:[^7^]

$$\text{Final Score} = \frac{1}{12}\sum_{i=0}^{11} \text{AUC}_i$$

Three consequences are definition-level, not speculative. **(a) Calibration and thresholds are irrelevant** — AUC is rank-based, so any monotonic per-class transform (Platt, isotonic, clipping) is a no-op; do not spend a single experiment on probability calibration. **(b) The worst class costs as much as the best.** Fracture, MCL, and Synovitis — precisely the classes with the noisiest silver supervision (fracture extraction sensitivity 0.44; synovitis 84% silent) — each contribute 1/12 of the score. Fixing the weakest column is worth more than polishing the strongest. **(c) Ensemble selection should be per-class:** select or weight ensemble members per label by their OOF AUC rather than globally; this directly exploits the macro-mean structure.[^85^]

On training loss, the evidence favors handling imbalance in the sampler and noise in the loss:

| Option | Verdict for this task | Evidence |
|---|---|---|
| Balanced study-level sampling | **Preferred first line** — sample positive/negative studies equally per class group | 2023 1st place: sampling > loss weights[^81^] |
| Asymmetric loss (γ−=4, γ+=1, clip=0.05) | **Preferred loss** — down-weights and hard-thresholds easy/mislabeled negatives; fits noisy multilabel supervision and uncoerced 0.5 cells | ASL paper, SOTA on multilabel benchmarks[^86^] |
| Weighted BCE (pos_weight) | Fallback only; failed to converge from scratch for 2023 winners; usable in a two-step recipe (AUC pretrain → frozen-backbone weighted head fine-tune) | 2023 1st; 2024 5th place[^81^][^87^] |
| Focal loss | Caution: emphasizes hard examples, which under noisy silver labels are disproportionately mislabeled | Focal-loss mechanics; CoPAS used focal on final output only[^73^][^88^] |
| Soft/graded targets | Use wherever the label pipeline emits them | +0.056 macro AUC (community-reported, single source)[^63^] |
| Loss masking on not-addressed cells | Mandatory if cells are not soft-coded; treat report-silent as unlabeled, not negative | 23% of studies report-silent; 0.44 fracture sensitivity[^4^][^89^] |

**TTA has one trap with a bounty attached.** A horizontal flip of a knee MRI swaps the medial and lateral compartments: Medial Meniscus↔Lateral Meniscus and Medial OA↔Lateral OA labels must be swapped on flip, in both training augmentation and TTA (with label-aware un-flipping). Handled correctly this was worth ~0.01 AUC to the RSNA 2025 aneurysm 5th place (left/right vessel labels swapped); handled naively it actively corrupts the four side-specific labels.[^90^] ACL/MCL/Effusion/Synovitis/Baker's/Contusion/Fracture are flip-invariant at exam level. Verify flip direction against ImageOrientationPatient before trusting it, validate every TTA set on OOF (recent evidence shows standard TTA can *hurt* medical classification), and skip TTA entirely on the efficiency-track submission.[^91^] For aggregation, max-pool over slices per series and use a small logistic stacker over planes — the MRNet template — rather than elaborate fusion.[^71^]

## 4.6 Playing the Efficiency Track

The first-ever RSNA efficiency track distributes $18,000 ($7k/$6k/$5k) across three places, scored on runtime plus AUC rather than accuracy rank.[^16^] The published definition (KaTeX partially lost in page extraction — **medium confidence on the exact form**, high confidence on direction and eligibility) reduces to:

$$\text{Efficiency} \approx \frac{\text{maxAUC} - \text{AUC}}{\text{maxAUC} - \text{BenchmarkAUC}} + \frac{\text{RuntimeSeconds}}{32400} \quad (\text{minimize})$$

where Benchmark is the `sample_submission.csv` private-LB score, maxAUC is the best private-LB submission, and 32,400 s is the 9-hour notebook cap. Eligibility requires beating the benchmark on the private leaderboard.[^16^] Whatever the exact normalization, the strategy is invariant: every saved hour buys ~0.11 of the runtime term, runtime counts the full evaluation wall time (DICOM I/O included, on a 569.76 GB dataset), and you must first clear the accuracy bar.

The rational solo play is to treat efficiency as a separate, winnable game rather than a constraint on the main entry. The recipe: **(1)** an ELNet-class student — ~0.2M parameters trained from scratch matched MRNet's 183M-parameter ensemble on knee MRI (0.904 vs 0.826 meniscus AUC) — distilled from the main-track ensemble's soft OOF predictions;[^92^] **(2)** FP16/AMP plus `torch.compile` as the safe acceleration path; **(3)** minimal series routing via `test_series.csv` plane/sequence flags, cached preprocessing, zero TTA, single SWA/EMA-averaged model. The dangerous temptation is TensorRT INT8: on Kaggle's T4 environment, TensorRT libraries can be absent, forcing ONNX Runtime into GPU↔CPU fallback measured at ~147× *slower* than PyTorch FP16 — vendor the libraries or build engines in-notebook, otherwise stay on FP16.[^93^] Submitting one maximal-ensemble notebook and one lean distilled notebook among the allowed final selections hedges both tracks simultaneously.

# 5. Engineering and Execution Roadmap

This chapter converts the research findings of Chapters 1–4 into a build plan for a solo practitioner working with an agentic coding platform (Kimi Code) plus Kaggle Notebooks. It covers DICOM engineering (§5.1), Kaggle platform mechanics (§5.2), agentic workflow discipline (§5.3), a ten-week plan from 2026-08-10 to the 2026-10-22 final deadline (§5.4), and a risk register (§5.5).

## 5.1 DICOM Engineering Essentials

The competition data is 819,640 DICOM files / 569.76 GB, one slice per file, organized as `<StudyUID>/<SeriesUID>/<SOPUID>.dcm`, in a mix of four transfer syntaxes (uncompressed Explicit VR Little Endian, Implicit VR Little Endian, JPEG Lossless, JPEG 2000), with every file stripped to an allowlisted set of 86 metadata tags — so rich protocol metadata is simply absent and must not be relied upon.[^2^] Series typically contain 20–45 slices (median 30), with a long tail to a few hundred.[^2^]

**Slice ordering.** Never trust InstanceNumber alone. Sort slices by projecting `ImagePositionPatient` (0020,0032) onto the slice normal computed as `cross(row, col)` from `ImageOrientationPatient` (0020,0037); use InstanceNumber only as a fallback, and derive slice spacing from the actual first/last slice positions rather than the `SliceThickness` tag.[^94^] Watch for multi-frame/enhanced DICOMs, where spacing and position live inside `PerFrameFunctionalGroupsSequence`/`SharedFunctionalGroupsSequence` and naive `ds.PixelSpacing` access fails.[^94^]

**Pixel values.** `ds.pixel_array` returns raw stored values. Apply `apply_modality_lut` (rescale slope/intercept) *before* `apply_voi_lut` (windowing), per the official pydicom docs.[^95^] If `PhotometricInterpretation` is `MONOCHROME1`, invert (`arr = arr.max() - arr`); MRI is usually MONOCHROME2, but check per-series.[^96^]

**Reading at scale.** Two viable toolchains: (a) pydicom (+ pylibjpeg for the JPEG transfer syntaxes) with the manual sorting above; (b) SimpleITK `ImageSeriesReader.GetGDCMSeriesFileNames`, which returns GDCM-sorted filenames and enumerates series via `GetGDCMSeriesIDs`.[^97^] **Pin your SimpleITK version**: 2.4.0 changed DICOM series direction handling (Z-component sign flip vs 2.3.1), which silently flips volumes if train and inference environments differ.[^98^] For NIfTI conversion, dcm2niix + nibabel is the de-facto standard and handles JPEG Lossless/JPEG 2000, but it cannot emit fields the anonymized header lacks.[^99^]

```python
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut

def read_series(paths):
    """Read one single-frame MRI series into a geometrically sorted float32 volume."""
    slices = [pydicom.dcmread(p) for p in paths]
    iop = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])              # slice-axis unit vector
    try:
        slices.sort(key=lambda ds: float(np.dot(
            np.asarray(ds.ImagePositionPatient, dtype=float), normal)))
    except AttributeError:                            # tag anonymized away
        slices.sort(key=lambda ds: int(ds.InstanceNumber))
    out = []
    for ds in slices:
        arr = apply_modality_lut(ds.pixel_array, ds)  # rescale FIRST
        arr = apply_voi_lut(arr, ds)                  # windowing SECOND
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr                     # invert
        out.append(arr.astype(np.float32))
    return np.stack(out)                              # (slices, rows, cols)
```

**Preprocessing.** For CNN classification, percentile-clip (e.g., 0.5–99.5%) plus per-volume z-score is sufficient — the peer-reviewed comparison of z-score, WhiteStripe, and Nyul found all three performed similarly for classification accuracy.[^100^] N4 bias-field correction costs minutes per volume; defer it to an offline ablation (precompute and cache, never on-the-fly).[^101^] Normalize slice count by interpolation to a fixed length (24–96 slices per series), the standard MRNet-style practice, instead of full isotropic resampling.[^102^] At 569.76 GB raw, cache preprocessed volumes once as float16 `.npy` files or tar shards; MONAI `CacheDataset`/`PersistentDataset` give up to ~10× training speedup by caching pre-random-transform outputs in RAM or on disk.[^103^] WebDataset-style tar shards (`{000000..N}.tar`) add 3–10× I/O throughput over random file access and conveniently sidestep Kaggle's 1000-file dataset limit.[^104^]

| # | DICOM processing checklist item | Tool / method | Why it matters here |
|---|--------------------------------|---------------|---------------------|
| 1 | Group files by SeriesInstanceUID (0020,000E) | pydicom header scan | Prevents mixing planes/sequences into one volume |
| 2 | Sort by IPP · cross(row, col) of IOP; InstanceNumber fallback | numpy + pydicom | InstanceNumber ordering is unreliable across vendors[^94^] |
| 3 | Spacing from first/last positions, not SliceThickness | numpy | Anonymized/irregular spacing[^94^] |
| 4 | Detect enhanced multi-frame DICOMs | functional-groups sequences | `ds.PixelSpacing` raises on these[^94^] |
| 5 | `apply_modality_lut` → `apply_voi_lut` (in that order) | pydicom.pixels | Raw stored values are not display/analysis values[^95^] |
| 6 | Invert MONOCHROME1 | `arr.max() - arr` | Inverted contrast otherwise[^96^] |
| 7 | Smoke-test all 4 transfer syntaxes | pydicom + pylibjpeg / gdcm | JPEG Lossless + JPEG 2000 present in data[^2^] |
| 8 | Assume only the 86 allowlisted tags exist | defensive `getattr` | Anonymization removed protocol metadata[^2^][^99^] |
| 9 | Percentile-clip + per-volume z-score | numpy | As good as Nyul/WhiteStripe for CNN classification[^100^] |
| 10 | Defer N4 bias field to cached ablation | SimpleITK/ANTs offline | Minutes per volume — kills on-the-fly pipelines[^101^] |
| 11 | Interpolate to fixed slice count (24–96) | `np.interp` over z | MRNet-standard; batching needs fixed shapes[^102^] |
| 12 | Cache volumes as float16 npy/tar shards | MONAI PersistentDataset / webdataset | 569.76 GB raw; 1000-file dataset limit[^103^][^104^] |
| 13 | Pin SimpleITK version in both train and submit envs | requirements pin | 2.4.0 flipped Z direction sign[^98^] |

## 5.2 Kaggle Platform Mechanics

Everything below is from Kaggle's official documentation, accessed 2026-08-10.[^105^] Weekly GPU quota is 30 hours (P100 or T4×2), corroborated by Kaggle staff.[^106^] The competition's code requirements cap submission notebooks at **9 h runtime (CPU or GPU), internet disabled**, output `submission.csv`.[^12^] Trained artifacts move between notebooks as auto-saved output (20 GB) or Kaggle Datasets; datasets have an intentional **1000-file limit**, so package weights and caches as tar/zip archives.[^107^] Offline dependencies install from a wheel dataset via `pip install --no-index --find-links=...`, or via Kaggle's Dependency Manager, which builds a wheel-bearing installation notebook for internet-off submissions.[^108^] Note for the efficiency track: RuntimeSeconds is the full notebook wall time — package installs, model loading, **and DICOM reading** all count.[^17^]

```bash
# Local (or internet-on notebook): build the wheelhouse
pip download monai==1.4.0 pydicom pylibjpeg -d ./wheels
# Upload ./wheels as a Kaggle Dataset, then in the offline submission notebook:
pip install --no-index --find-links=/kaggle/input/my-wheels/wheels monai pydicom pylibjpeg
```

| Kaggle resource / limit | Value | Source |
|--------------------------|-------|--------|
| GPU options | 1× Tesla P100 (16 GB) or 2× Tesla T4; 4 CPU cores, 29 GB RAM | Kaggle docs[^105^] |
| GPU quota | 30 h/week (TPU ~20 h/week); max 2 concurrent batch GPU sessions | Kaggle Book + staff Q&A[^106^] |
| Session runtime | 12 h CPU/GPU notebooks; 9 h TPU | Kaggle docs[^105^] |
| Submission cap | ≤ 9 h CPU or GPU notebook; internet off | Code requirements[^12^] |
| Auto-saved output | 20 GB in /kaggle/working; reusable as input to later notebooks | Kaggle docs[^105^] |
| Dataset file limit | 1000 files per user dataset → ship tar shards | Kaggle staff[^107^] |
| Interactive idle timeout | ~20 min; long runs need Save & Run All (top-to-bottom, ≤12 h) | Kaggle docs[^105^] |
| Docker image | Updated ~every 2 weeks; pin "original environment" in Session options | Kaggle docs[^105^] |
| Extra GPU hours | Colab Pro/Pro+ promo: +15/+30 h/week on the same hardware | Kaggle docs[^105^] |
| Efficiency runtime | Full wall time incl. installs, model load, DICOM decode | Kaggle staff[^17^] |

One mixed-precision note: P100/T4 have no bf16, so use fp16 with `torch.amp.GradScaler('cuda')` (the `torch.cuda.amp` API is deprecated).[^109^] Budget arithmetic: 10 weeks × 30 GPU-h ≈ 300 GPU-hours total — treat this as the hard planning currency in §5.4.

## 5.3 Working Effectively with an Agentic Coding Platform (Kimi Code)

Agentic coding pays off only with structure; practitioners who moved "from vibe coding to agentic engineering" converge on the same patterns: a persistent project-instructions file, plan-mode artifacts with checkbox status, and treating context as a scarce resource.[^110^] Concretely for this project:

1. **Maintain `AGENTS.md` at repo root** with environment facts the agent cannot re-derive: Kaggle GPU/CPU specs, the 30 h/week quota, pinned versions (SimpleITK pin with the 2.4.0 warning[^98^]), the offline-install mechanism, the 86-tag allowlist, and "do-not-touch" paths (gold labels, raw DICOM cache). A Kaggle-specific precedent is ExpAgent: git-tracked `project.yml` for competition metadata plus git-ignored `.env` for `KAGGLE_USERNAME`/`KAGGLE_KEY`.[^111^]
2. **Config-driven experiments.** Winner repos structure work as one folder per experiment with a config (backbone, planes, slices, augmentation, lr), a precomputed shared 5-fold split, and shell entry points (`preprocess.sh`, `run.sh`).[^112^] The agent edits YAML, not training code.
3. **Thin Kaggle notebooks over local code.** Keep training/inference code locally executable in `src/`; Kaggle notebooks are wrappers that pip-install the wheel dataset, attach weight/cache datasets, and call `src/train.py` / `src/infer.py` — the pattern used by the ARC Prize 2024 winner for offline submissions.[^113^]
4. **Reproducibility contract:** fixed seeds logged per config, out-of-fold (OOF) prediction artifacts saved per experiment, per-fold weights and metrics under each model directory.[^112^] Instruct the agent to never overwrite OOF artifacts — they are the ensemble's raw material.

```
rsna-knee/
  AGENTS.md             # env facts, pinned versions, quotas, do-not-touch paths
  project.yml           # competition metadata (git-tracked)
  .env.example          # KAGGLE_USERNAME / KAGGLE_KEY placeholders (git-ignored real .env)
  configs/              # one YAML per experiment: backbone, planes, n_slices, aug, lr, fold
  input/                # raw CSVs + cache manifests (DICOMs stay on Kaggle)
  src/
    dicom_io/           # series grouping, IPP sort, LUT order, photometric fix
    preprocess/         # clip + z-score, resize, slice interpolation -> npy/tar cache
    datasets/           # MONAI CacheDataset/PersistentDataset, shard writers
    models/             # 2.5D CNN encoder + attention-MIL / BiLSTM slice aggregator
    train.py            # fp16 AMP loop, fold + seed from config, CSV/JSON logging
    infer.py            # study-level aggregation, submission.csv writer
  models/               # per-experiment: weights/, oof/, metrics.json
  notebooks/            # thin Kaggle wrappers: prep / train / submit
  kaggle/               # scripts: build wheels dataset, push weights dataset via API
```

## 5.4 Ten-Week Execution Plan

Anchors from the official timeline (all 11:59 PM UTC): **entry/team-merger deadline Oct 15; final submission Oct 22; winners' materials (code, weights as a public dataset, video, method description) Nov 5.**[^19^] The plan below assumes ~30 GPU-hours/week and reflects two strategic facts established earlier: label quality is the largest controllable lever, and commercial LLM APIs are explicitly permitted for report label extraction (host ruling, 2026-08-09).[^14^]

| Week | Dates (2026) | Focus | Deliverables / exit criteria | GPU-h budget |
|------|--------------|-------|------------------------------|--------------|
| W1 | Aug 10–16 | EDA + DICOM pipeline | §5.1 checklist implemented; per-series volumes cached as shards; decode tested on all 4 transfer syntaxes; first dry-run submission (all-0.5 / trivial baseline) to validate the notebook path | ~10 |
| W2 | Aug 17–23 | Label extraction v1 + evaluation harness | Multi-LLM ensemble extractor (multilingual-aware); 58-gold evaluation harness reporting macro AUC per extractor variant — recall LLM labels already beat regex 0.8780 vs 0.8136 against the gold studies, and ~25.4% of cells are "not addressed" (mask, don't zero-fill)[^4^] | ~15 |
| W3 | Aug 24–30 | Baseline image model | MRNet-style 2.5D CNN + gated attention-MIL per plane/sequence; scanner-grouped 5-fold CV running; first real submission | ~30 |
| W4 | Aug 31–Sep 6 | Baseline iteration | Fix data bugs found by CV-vs-LB deltas; augmentation set validated (label-aware hflip, §5.5); OOF artifact store complete | ~30 |
| W5 | Sep 7–13 | Localization stage | ROI/meniscus localizer feeding cropped classification; measure delta on gold + grouped CV | ~30 |
| W6 | Sep 14–20 | Architecture iteration | 1–2 alternatives (3D-ResNet w/ MedicalNet init, BiLSTM aggregator); keep only grouped-CV winners | ~30 |
| W7 | Sep 21–27 | Labels v2 | Refined extractor prompts, targeted gap-filling (never blanket imputation), soft/graded targets; retrain best config | ~30 |
| W8 | Sep 28–Oct 4 | Ensembling + CV hardening | Fold × seed × backbone ensemble; OOF-weighted stacking; final CV vs LB correlation report | ~30 |
| W9 | Oct 5–11 | Efficiency-track candidate | Distill ensemble into small student; fp16 + torch.compile; measure notebook wall time; freeze both candidate pipelines | ~35 |
| W10 | Oct 12–22 | Buffer + submission selection | **Accept rules before Oct 15**; end-to-end dry runs under 9 h; pick 2 final submissions (best-CV ensemble + lean efficient one) by grouped CV + gold, not LB decimals; submit by Oct 22 | ~25 |

Total ≈ 265 GPU-hours planned against ~300 available — keep the ~35 h reserve for reruns and deadline-week failures.

## 5.5 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Silver-label noise caps the image model (reports are a noisy source; image-derived labels are authoritative) | High | High | Multi-LLM ensemble extraction; evaluate every label variant on the 58 gold studies; mask "not addressed" cells; soft targets[^4^] |
| Site/scanner shift across 16 institutions (265 scanner fingerprints; metadata memorization does not transfer) | High | High | Scanner-grouped CV as the primary selection metric; per-volume z-score + aggressive intensity augmentation[^21^] |
| Public LB shake-up: public = ~30% of ~1,300 test studies, prevalence not matched across splits | Medium-High | High | Select finals on grouped CV + gold, not LB; two diverse final submissions; conservative ensembling[^10^] |
| External-data ruling pending: MRNet/OAI are gated click-through datasets, borderline under "equally accessible" wording | Medium | Medium | Treat as unusable until the host rules; clone architectures (MRNet/CoPAS) rather than data; monitor the rules thread[^114^] |
| Compute budget overrun (30 GPU-h/week; 12 h session cap) | Medium | High | Cache preprocessing once (§5.1); iterate on small models; weekly GPU-h ledger; queue long runs via Save & Run All[^106^] |
| TensorRT/INT8 backfire on Kaggle T4 (~147× slower than fp16 when TensorRT libs absent) | Medium | Medium | Efficiency track uses PyTorch fp16/AMP + torch.compile; vendor TensorRT libs or build engines in-notebook only if benchmarked[^93^] |
| Horizontal-flip laterality trap: hflip swaps medial↔lateral labels | Medium | High | Label-aware flip (swap Medial↔Lateral Meniscus and Medial↔Lateral OA) or disable hflip; verify flip direction against ImageOrientationPatient — the trick was worth ~0.01 AUC in RSNA 2025 when done correctly[^90^] |
| Non-Latin report long tail (Greek 7.3%, Cyrillic 5.0%): keyword/regex pipelines silently return confident negatives | High | Medium | Multilingual LLM extractor; script-detection audit of extractor output; no regex-only fallback[^3^] |
| Submission notebook fails or exceeds 9 h at deadline | Medium | High | Dry-run the full notebook weekly from W8; per-study try/except with fallback predictions; pin Docker environment[^105^][^12^] |
| SimpleITK 2.4.0 direction-matrix change flips volumes between train and inference | Low | Medium | Pin SimpleITK in both environments; volume-orientation smoke test in the submission notebook[^98^] |

# Consolidated Reference List

[^1^]: RSNA News — AI Challenge Knee MRI — https://www.rsna.org/news/2026/august/ai-challenge-knee-mri (2026-08-06)
[^2^]: Kaggle — RSNA Knee Abnormality Detection, Data page — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data (accessed 2026-08-10)
[^3^]: Kaggle Discussion 734055 — "train.csv has 4,407 studies and 58 labels" (maximo lorenzo y losada) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055 (2026-08-09/10)
[^4^]: Kaggle Discussions 733932 / 733592 — reports are train-only; host confirmation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592 (2026-08-08/09)
[^5^]: Kaggle — RSNA Knee Abnormality Detection, Overview: Prizes — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/prizes (accessed 2026-08-10)
[^6^]: RSNA press release — https://www.rsna.org/media/press/2026/2669 (2026-08-05)
[^7^]: Kaggle — RSNA Knee Abnormality Detection, Overview: Evaluation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation (accessed 2026-08-10)
[^8^]: Kaggle Discussion 733343 — "Knee Abnormality Detection AI Challenge Overview" (pinned host post, courtesy of Dr. Jacob Kazam) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343 (2026-08-06)
[^9^]: Kaggle Discussion 733826 — host replies on image-derived labels; community report-only audit — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826 (2026-08-08)
[^10^]: Kaggle — Leaderboard page and Data page — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard (accessed 2026-08-10)
[^11^]: RSNA challenge page and press materials ("nine languages" vs "a dozen") — https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge ; https://www.rsna.org/news/2026/august/ai-challenge-knee-mri (accessed 2026-08-10)
[^12^]: Kaggle — RSNA Knee Abnormality Detection, Overview: Code Requirements — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements (accessed 2026-08-10)
[^13^]: Kaggle — RSNA Knee Abnormality Detection, Rules — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules (accessed 2026-08-10)
[^14^]: Kaggle Discussion 733965 — "Use of Commercially Hosted LLMs" (host ruling) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965 (2026-08-09)
[^15^]: Kaggle Discussion 733652 — external knee-MRI dataset eligibility (unresolved) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652 (2026-08)
[^16^]: Kaggle — Overview: Efficiency Prize Evaluation — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation (accessed 2026-08-10)
[^17^]: Kaggle Discussion 733475 — RuntimeSeconds definition (Ryan Holbrook, Kaggle Staff) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475 (2026-08-07)
[^18^]: Kaggle Notebook — "RSNA Knee Abnormalities — Efficiency LB" (Ryan Holbrook) — https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb (accessed 2026-08-10)
[^19^]: Kaggle — Overview: Timeline — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline (accessed 2026-08-10)
[^20^]: Kaggle — Overview page sidebar (participation snapshot) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview (accessed 2026-08-10)
[^21^]: Kaggle Discussion 733517 — "0.932 LB within one day. Tested for DICOM metadata shortcut" (Oleksii Zhukov) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517 (2026-08-07)
[^22^]: Kaggle — Code page sorted by votes (public baselines and scores) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code?sortBy=voteCount (accessed 2026-08-10)
[^23^]: Kaggle Discussion 734004 — "DICOM metadata findings: scanner-grouped CV and PatientSex priors" (morningduck) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004 (2026-08-09)
[^24^]: Radiopaedia, "Knee joint" — https://radiopaedia.org/articles/knee-joint-1 (accessed 2026-08-10)
[^25^]: EPOS/ESR educational exhibit, "Acute trauma of the knee ligaments: Following the contusion pattern" (ECR 2016, C-1617) — https://epos.myesr.org/poster/esr/ecr2016/C-1617/background (2016-03-02)
[^26^]: ESSR Refresher Course 2008, "Knee collateral ligaments" (A. Karantanas) — https://www.essr.org/content-essr/uploads/2016/10/Refresher-Course2008.pdf (2008)
[^27^]: Chien A, et al. "Magnetic resonance imaging of the knee." Pol J Radiol 2020 (PMC7571514) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7571514/ (2020)
[^28^]: Radiopaedia, "Synovial joints" (rID-42705) — https://radiopaedia.org/articles/synovial-joints (accessed 2026-08-10)
[^29^]: NIST, "How Does an MRI Machine Work?" — https://www.nist.gov/how-do-you-measure-it/how-does-mri-machine-work (2025-05-14)
[^30^]: RadiologyKey, "Lower Limb II: Knee" — https://radiologykey.com/lower-limb-ii-knee/ (2016-07-24)
[^31^]: MusculoskeletalKey, "The Knee" (MRI protocol chapter) — https://musculoskeletalkey.com/the-knee/ (2016-05-28)
[^32^]: MXR Imaging, "T1 vs. T2 MRI: Key Differences" — https://mxrimaging.com/blogs/t1-vs-t2-mri-imaging/ (2026-03-30)
[^33^]: AJR, "Comparison of Fat-Suppressed T2-Weighted FSE and Modified STIR" — https://ajronline.org/doi/10.2214/ajr.185.2.01850371 (2012)
[^34^]: RadioGraphics, "Fat-Suppression Techniques for 3-T MR Imaging of the Musculoskeletal System" (PMC4359893) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4359893/ (2015)
[^35^]: Cross-Verification — RSNA Knee Abnormality Detection Research (annotation protocol, thresholds, label list, metric) — /mnt/agents/output/research/rsna_knee_cross_verification.md (2026-08-10)
[^36^]: RuntimeWire, "RSNA opens $77,000 challenge for AI that reads knee MRI and reports" — https://runtimewire.com/article/rsna-knee-mri-ai-challenge-2026 (2026-08-07)
[^37^]: Radiopaedia, "Anterior cruciate ligament tear" (rID-12490) — https://radiopaedia.org/articles/anterior-cruciate-ligament-tear (accessed 2026-08-09)
[^38^]: Radsource, "Medial Supporting Structures of the Knee with Emphasis on the MCL" — https://radsource.us/medial-supporting-structures-knee-emphasis-medial-collateral-ligament/ (2024-04-24)
[^39^]: Chana-Rodríguez F, et al. "Reporting knee meniscal tears: technical aspects, typical pitfalls and how to avoid them." Insights Imaging 2016 (PMC4877346) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4877346/ (2016)
[^40^]: Radsource, "Osteoarthritis (OA) of the Knee" — https://radsource.us/osteoarthritis-oa-of-the-knee/ (2023-05-08)
[^41^]: "Joint effusion of the knee: potentialities and limitations…" Insights Imaging 2015 (PMC4630268) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4630268/ (2015)
[^42^]: Burke CJ, et al. "MRI of Synovitis and Joint Fluid" (PMC6504589) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6504589/ (2019)
[^43^]: Radiopaedia, "Baker cyst" (rID-21117) — https://radiopaedia.org/articles/baker-cyst-2 (accessed 2026-08-06)
[^44^]: "Intramuscular Dissecting Baker's Cysts: A Case Series" (PMC10846661) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10846661/ (2024)
[^45^]: EPOS/ESSR 2019 poster P-0162, "Bone Contusion Patterns of the Knee at MRI" — https://epos.myesr.org/poster/esr/essr2019/P-0162/imaging%20findings%20or%20procedure (2019-05-24)
[^46^]: Radiopaedia, "MRI grading system for abnormal meniscal signal intensity" (rID-36617) — https://radiopaedia.org/articles/mri-grading-system-for-abnormal-meniscal-signal-intensity (accessed 2026-08-06)
[^47^]: pacs.de, "Outerbridge grading system" (mirrors Radiopaedia modified Outerbridge grading) — https://pacs.de/term/outerbridge-grading-system (2022-11-08)
[^48^]: "Grading of Knee Osteoarthritis Based on Kellgren-Lawrence Classification…" Cureus 2024 (PMC11624959) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11624959/ (2024)
[^49^]: "MRI of Internal Derangements and Other Knee Pathologies in Adult Nigerians" (PMC11214712) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11214712/ (2024)
[^50^]: Chinmay Gupte, "How do you read a knee MRI" — https://www.chinmaygupte.com/how-do-you-read-a-knee-mri (accessed 2026-08-10)
[^51^]: JACR, "Analysis of Different Levels of Structured Reporting in Knee MRI" — https://www.sciencedirect.com/science/article/abs/pii/S1076633220300131 (2020-10-01)
[^52^]: MusculoskeletalKey, "The Knee" — "BOX 1: The Structured Report: Knee" — https://musculoskeletalkey.com/the-knee-9/ (2016-12-21)
[^53^]: NCBI Bookshelf, "Interpretation of kappa (from Landis and Koch 1977)" — https://www.ncbi.nlm.nih.gov/books/NBK92287/table/executivesummary.t2/ (n.d.)
[^54^]: "Can a single isotropic 3D FSE sequence replace three-plane standard PD FS knee MRI at 1.5 T?" Br J Radiol 2015 (PMC4651376) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4651376/ (2015)
[^55^]: Springer, "Timing of MRI affects the accuracy and interobserver agreement of anterolateral ligament tears detection in ACL deficient knees" — https://link.springer.com/article/10.1186/s43019-020-00082-z (2020-11-27)
[^56^]: ABC Research journal, "Reliability of MRI vs arthroscopy" — https://abcresearch.net/pdf/0fdb9ffe-e838-45c0-b564-25a52c51df96/issues/2026-008-001.pdf (2026)
[^57^]: Hunter DJ, et al. "Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS." Osteoarthritis Cartilage 2011 (PMC4058435) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435/ (2011)
[^58^]: Springer J Supercomputing review (2025), summarizing Bien N, et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging (MRNet)," PLoS Medicine 2018 — https://link.springer.com/content/pdf/10.1007/s11227-025-07103-2.pdf (2025)
[^59^]: Kohn MD, et al. "Classifications in Brief: Kellgren-Lawrence Classification of Osteoarthritis." Clin Orthop Relat Res 2016 (PMC4925407) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4925407/ (2016)
[^60^]: Kaggle discussion, "'Not addressed' is a label too" (stevenleehans) — LLM vs regex label quality (0.8780 vs 0.8136 macro AUC vs the 58 gold); 25.4% of cells unaddressed; per-finding silence semantics; <0.02 macro deltas unmeasurable on 58 studies. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932; report-vs-image agreement 82.5% per discussion 733826 (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826, host-confirmed that image-derived labels are authoritative).
[^61^]: Kaggle Data page + discussion 734095 — test.csv has no Report column; reports are a train-only teacher. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734095
[^62^]: Cross-verification of CoPAS internal/external AUC (0.812 → 0.721/0.726) from the Nature Communications full text; architecture-cloning (not data) recommended while the external-data ruling is pending.
[^63^]: Kaggle discussion 734105 — community-reported +0.056 macro AUC for graded/soft targets over binary (title-level evidence; single source). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105
[^64^]: Community EDA (R. Rozen notebook; yuki16 notebook; discussion 733826) — attested languages: EN/ES/NL dominant, FR/DE/PT/IT/TR tail, BG/EL rare; RSNA's own pages say "nine" (challenge page) vs "a dozen" (press release) languages. https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline
[^65^]: CXR-LanIC paper — arXiv:2510.21464 — https://arxiv.org/html/2510.21464v4 (2026-05-19)
[^66^]: Qwen3 Technical Report — arXiv:2505.09388 — https://arxiv.org/html/2505.09388v1 (2025)
[^67^]: "Anatomy-aware prompting for radiology report classification with GPT-OSS-20B" — arXiv:2512.05537 — https://arxiv.org/pdf/2512.05537 (2025-12)
[^68^]: Thieme/Rofo — "German CheXpert Chest X-ray Radiology Report Labeler" (Wollek et al. 2023) — https://www.thieme-connect.com/products/ejournals/html/10.1055/a-2234-8268 (2024-01-31)
[^69^]: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R) — arXiv:1911.02116 — https://arxiv.org/pdf/1911.02116 (2020)
[^70^]: "Evaluation of GPT-4o for multilingual translation of radiology reports" (European Radiology experimental) — https://www.sciencedirect.com/science/article/pii/S0720048X25004279 (2025-08-29); NLLB-200 model card — https://www.modelscope.cn/models/facebook/nllb-200-1.3B/summary (2022)
[^71^]: Bien et al., "Deep-learning-assisted diagnosis for knee MRI: Development and retrospective validation of MRNet" (PLoS Medicine) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/ (2018-11-27)
[^72^]: Kaggle write-up — "1st Place Solution" (Qishen Ha), RSNA 2022 Cervical Spine Fracture Detection — https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution (2022-10-29)
[^73^]: Qiu et al., "Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities" (CoPAS), Nature Communications 15 — https://www.nature.com/articles/s41467-024-51888-4 ; code: https://github.com/zqiuak/CoPAS (2024-09-02)
[^74^]: Kaggle write-up — "1st place solution" (NANACHI), RSNA 2024 Lumbar Spine Degenerative Classification — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution (2024-10-28)
[^75^]: Raghu et al., "Transfusion: Understanding Transfer Learning for Medical Imaging" — arXiv:1902.07208 — https://arxiv.org/abs/1902.07208 (2019)
[^76^]: "Robust Weakly Supervised Learning for COVID-19 Recognition Using Multi-Center CT Images" — arXiv:2112.04984 — https://arxiv.org/pdf/2112.04984.pdf (2021)
[^77^]: Kaggle write-up — "1st Place Solution" (tomoon33), RSNA Intracranial Aneurysm Detection — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution (2025-10-15)
[^78^]: Yiu et al., "A Systematic Study of Deep Learning Models and xAI Methods for ROI Detection in MRI Scans" — arXiv:2508.14151 — https://arxiv.org/html/2508.14151v1 (2025-08-19)
[^79^]: Kaggle write-up — "7th place solution - 3D nnU-Net + blob regression (again)" (MIC-DKFZ), RSNA Intracranial Aneurysm Detection — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution (2025-10-16)
[^80^]: Kaggle write-up — "6th Place Solution" (NVSpine), RSNA 2024 Lumbar Spine Degenerative Classification — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/nvspine-6th-place-solution (2024-10)
[^81^]: Kaggle write-up — "1st Place Solution: Team Oxygen", RSNA 2023 Abdominal Trauma Detection — https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen (2023-10-22)
[^82^]: Kushol et al., cross-manufacturer MRI harmonization study, Scientific Reports 2023 — https://www.nature.com/articles/s41598-023-43715-5.pdf (2023)
[^83^]: "Mitigating Bias in Radiology Machine Learning: 1. Data" (Radiology: AI) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9533091/
[^84^]: Szymański & Kajdanowicz, "A Network Perspective on Stratification of Multi-Label Data" (PMLR v74) — http://proceedings.mlr.press/v74/szymański17a/szymański17a.pdf (2017)
[^85^]: "Class-Wise Ensemble" multi-label radiology label paper — arXiv:2308.08853 — https://arxiv.org/pdf/2308.08853
[^86^]: Ridnik et al., "Asymmetric Loss For Multi-Label Classification" — arXiv:2009.14119 — https://arxiv.org/pdf/2009.14119 (2021)
[^87^]: siwooyong, RSNA 2024 Lumbar Spine 5th-place solution repository — https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification (2024)
[^88^]: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017), via TUM lecture slides — https://dvl.in.tum.de/slides/cv3dst-ws19/3.ObjectDetection2.pdf
[^89^]: Kaggle discussion 734117 — "Weak labels for all 12 findings + how recoverable each one actually is" (Luka Duvanov) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734117 (2026-08-10)
[^90^]: Kaggle write-up — "5th place solution with code" (HoangHuyen), RSNA Intracranial Aneurysm Detection — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution (2025-10-15)
[^91^]: "I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image Classification" — arXiv:2604.09697 — https://arxiv.org/html/2604.09697v1 (2026-04-06)
[^92^]: Tsai et al., "Knee Injury Detection using MRI with Efficiently-Layered Network (ELNet)" (MIDL 2020) — arXiv:2005.02706 — https://arxiv.org/pdf/2005.02706 ; code: https://github.com/mxtsai/ELNet (2020-05-06)
[^93^]: "Guidance-Aware Quantization for Classifier-Free Diffusion" (Kaggle T4 TensorRT measurement) — arXiv:2607.08241 — https://arxiv.org/html/2607.08241v1 (2026-07-09)
[^94^]: 3D Slicer community discourse, "DICOM to voxel Python" worked code example — https://discourse.slicer.org/t/dicom-to-voxel-python-landmark-placement-issue-for-dl-dataset/44366 (2025-09-05)
[^95^]: pydicom official API docs, apply_voi_lut / apply_modality_lut — https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html (accessed 2026-08-10)
[^96^]: K-Dense-AI scientific-agent-skills, pydicom SKILL.md — https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/pydicom/SKILL.md (2025-10-19)
[^97^]: SimpleITK official Doxygen, ImageSeriesReader — https://simpleitk.org/doxygen/v2_3/html/classitk_1_1simple_1_1ImageSeriesReader.html (2023-09-12)
[^98^]: SimpleITK GitHub issue #2214 (2.4.0 direction change) — https://github.com/SimpleITK/SimpleITK/issues/2214 (2025-01-08)
[^99^]: dcm2niix BIDS README (rordenlab) — https://github.com/rordenlab/dcm2niix/blob/master/BIDS/README.md (accessed 2026-08-10)
[^100^]: Nature Scientific Reports, "Standardization of brain MR images across machines and protocols" — https://www.nature.com/articles/s41598-020-69298-z (2020-07-23)
[^101^]: DIPY official docs, bias correction guide (N4 cost / fast alternatives) — https://docs.dipy.org/dev/examples_built/preprocessing/bias_correction_dwi.html (accessed 2026-08-10)
[^102^]: GitHub Elzawawy/MRNet (slice-count interpolation to 24) — https://github.com/Elzawawy/MRNet (2019-05-18)
[^103^]: MONAI official docs, Modules Overview + fast model training guide (CacheDataset/PersistentDataset) — https://monai.readthedocs.io/en/0.9.0/highlights.html (2022-06-13)
[^104^]: webdataset official README — https://github.com/webdataset/webdataset/blob/main/README.md (accessed 2026-08-10)
[^105^]: Kaggle official Notebooks documentation — https://www.kaggle.com/docs/notebooks (accessed 2026-08-10)
[^106^]: The Kaggle Book (Bojan Tunguz) + Kaggle Q&A #306441 (staff confirmation of quotas/12 h cap) — https://www.alvinang.sg/s/The-Kaggle-Book-Bojan.pdf ; https://www.kaggle.com/questions-and-answers/306441 (Q&A 2025-04-12)
[^107^]: Kaggle Product Feedback #162754 (staff: 1000-file dataset limit; chaining notebook output) — https://www.kaggle.com/product-feedback/162754 (accessed 2026-08-10)
[^108^]: Kaggle Q&A #567059 (offline wheel installs) — https://www.kaggle.com/questions-and-answers/567059 (2025-03-07)
[^109^]: mljourney, "Mixed Precision Training with PyTorch AMP" (fp16 vs bf16; GradScaler API) — https://mljourney.com/mixed-precision-training-with-pytorch-amp-fp16-bf16-and-gradscaler/ (2026-05-17)
[^110^]: LevelUp, "Claude Code Best Practices: 12 Patterns Agentic Engineers Use" — https://levelup.gitconnected.com/claude-code-best-practices-12-patterns-agentic-engineers-use-65264e3eb919 (2026-04-15)
[^111^]: GitHub osushinekotan/ExpAgent (Kaggle agent scaffolding: project.yml + .env) — https://github.com/osushinekotan/ExpAgent (2026-03-14)
[^112^]: GitHub shimacos37/kaggle-rsna-2019-10th-solution (top-10 solution repo layout) — https://github.com/shimacos37/kaggle-rsna-2019-10th-solution (2019-11-24)
[^113^]: GitHub da-fr/arc-prize-2024 (thin Kaggle notebooks + offline wheel datasets) — https://github.com/da-fr/arc-prize-2024/ (2024-11-12)
[^114^]: RSNA 2024 Lumbar Spine rules page (external-data wording precedent) — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules (accessed 2026-08-10)
