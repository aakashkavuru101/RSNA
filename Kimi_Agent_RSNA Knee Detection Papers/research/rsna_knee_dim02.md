# RSNA Knee Abnormality Detection (Kaggle, 2026) — Dimension 02: Dataset Anatomy & EDA Knowledge

Research date: 2026-08-10. Researcher: sub-agent dim02.
Citation format: **Claim / Source / URL / Date / Excerpt (verbatim) / Confidence**.
Primary sources: Kaggle competition pages (Data, Overview/Evaluation, Discussion, Code), RSNA.org pages, RSNA press release.

---

## 0. Provenance / basic facts

**Claim:** Competition launched July 30, 2026 (start date on Kaggle timeline; RSNA public announcement Aug 5–6, 2026); final submission Oct 22, 2026; $77,000 total prizes; main metric = macro-averaged ROC AUC over 12 targets.
**Source:** Kaggle Overview/Evaluation page; RSNA press release.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview ; https://www.rsna.org/media/press/2026/2669?PdfExport=1
**Date:** accessed 2026-08-10.
**Excerpt:** "July 30, 2026 - Start Date. October 15, 2026 - Entry Deadline... October 22, 2026 - Final Submission Deadline." / "Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets... The final score is, in other words, the macro-averaged AUC ROC." / "The top performing teams will share a total of $77,000 in prize money, including for the first time awards for the most efficient models."
**Confidence:** high.

**Claim:** Participation (as of ~Aug 10, 2026): 1,011 participants, 961 teams, 4,052 submissions, 8,892 entrants.
**Source:** Kaggle Overview page sidebar.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview
**Date:** accessed 2026-08-10.
**Excerpt:** "Participation 8,892 Entrants 1,011 Participants 961 Teams 4,052 Submissions"
**Confidence:** high.

---

## 1. The 12 abnormality classes (exact names + official definitions)

**Claim:** The 12 targets are binary per-study labels, with these exact column names: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
**Source:** Kaggle Data page (Dataset Description).
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Twelve binary labels: ACL - anterior cruciate ligament injury (0/1). MCL - medial collateral ligament injury (0/1). Medial Meniscus - medial meniscus tear (0/1). Lateral Meniscus - lateral meniscus tear (0/1). Medial OA - osteoarthritis of the medial tibiofemoral compartment (0/1). Lateral OA - osteoarthritis of the lateral tibiofemoral compartment (0/1). PF OA - patellofemoral osteoarthritis (0/1). Effusion - joint effusion / excess fluid (0/1). Synovitis - inflammation of the joint lining (0/1). Baker's - Baker's cyst (0/1). Contusion - bone contusion / bone bruise (0/1). Fracture - fracture (0/1)."
Submission header (verbatim): "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture"
**Confidence:** high.

**Claim:** Official annotation rubric (used by the expert annotators; borderline/"on the fence" graded negative to favor specificity). Published in the pinned host discussion "Knee Abnormality Detection AI Challenge Overview" (courtesy of Dr. Jacob Kazam).
**Source:** Kaggle Discussion (Competition Host Po-Hao "Howard" Chen).
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
**Date:** posted ~2026-08-06; accessed 2026-08-10.
**Excerpts (verbatim):**
- "In each case, ambiguous or borderline findings ('on the fence') were graded as negative to favor specificity."
- "ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative."
- "MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative."
- "Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative." / "Lateral meniscus tear: The same criteria applied to the lateral meniscus."
- "Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes." / "Lateral compartment osteoarthritis: The same criteria applied to the lateral compartment." / "Patellofemoral compartment osteoarthritis: The same criteria applied to the patellofemoral compartment."
- "Joint effusion: A moderate or large amount of fluid distending the joint."
- "Synovitis: Inflammation and thickening of the synovial lining of the joint."
- "Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee."
- "Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line."
- "Acute fracture: An acute cortical break or fracture line."
- Annotation protocol: "Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee."
**Confidence:** high.

---

## 2. Data structure

**Claim:** Imaging is DICOM only (no PNG/NIfTI), one `.dcm` per slice, organized `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`; series typically 20–45 slices (median 30) with a long tail to a few hundred.
**Source:** Kaggle Data page.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Training DICOMs, organized as train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm. Each .dcm is a single image slice. Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred."
**Confidence:** high.

**Claim:** Each study = multiple series; per-series metadata is given in `train_series.csv` with three organizer-curated descriptors: `Fluid_Sensitive` (1 if T2/PD/STIR-like), `Fat_Suppression` (0/1), `Anatomical_Plane` (Sagittal/Coronal/Axial). Pulse-sequence names are NOT given; only these derived flags.
**Source:** Kaggle Data page.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Fluid_Sensitive - 1 if the sequence emphasizes fluid signal (T2, PD, STIR, and similar), 0 otherwise. Fat_Suppression - 1 if the sequence applies fat suppression, 0 otherwise. Anatomical_Plane - imaging plane: Sagittal, Coronal, or Axial."
**Confidence:** high.

**Claim:** Mixed transfer syntaxes; DICOMs stripped to an allowlist of 86 metadata tags; intensities/orientations/resolutions vary.
**Source:** Kaggle Data page ("DICOM Notes").
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Intensities, orientations, and resolutions vary across series and studies. Series come in a mix of transfer syntaxes (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian). Every DICOM has been stripped to an allowlisted set of 86 metadata tags."
**Confidence:** high.

**Claim:** Reports are provided as a free-text column `Report` inside `train.csv` (not separate txt files), alongside `StudyInstanceUID` and `PatientSex` (Male/Female, may be blank). Labels are per-exam (study-level) multi-label binary (12 columns of 0/1) — no severity grades, no localization.
**Source:** Kaggle Data page.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "train.csv One row per training study. StudyInstanceUID - unique identifier for the study; matches the folder name under train_series/. PatientSex - patient sex (Male or Female; may be blank). Report - the free-text radiology report. May be in any of several languages, depending on the reporting institution."
**Confidence:** high.

**Claim:** Total download = 819,640 files / 569.76 GB (train + example test). Training portion alone ≈ 4,407 studies, 24,371 series, ~730,000 slices, ~265 GB.
**Source:** Kaggle Data page; Kaggle notebook "RSNA Knee EDA the reports are train only" (Will / wguesdon).
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data ; https://www.kaggle.com/code/wguesdon/rsna-knee-eda-the-reports-are-train-only
**Date:** accessed 2026-08-10.
**Excerpt (verbatim, data page):** "Files 819640 files Size 569.76 GB Type dcm, csv"
**Excerpt (verbatim, notebook snippet):** "Twelve findings per knee MRI study, scored as the macro average of twelve ROC AUCs. 4,407 training studies, 24,371 series, roughly 730,000 slices, 265 GB."
**Confidence:** high.

**Claim:** Left/right laterality: volumes can be normalized to a "right knee" frame; 4 of the 12 targets are medial/lateral pairs; left knees mirror on coronal/axial planes and sagittal slice order reverses.
**Source:** Kaggle notebook "RSNA Knee | Data structure, EDA, baseline🔥" (Roman Rozen, ~88 upvotes, public score 0.809, best 0.894).
**URL:** https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline
**Date:** 2026-08-06 (v14 ~2026-08-09); accessed 2026-08-10.
**Excerpt (verbatim, search-index snippet):** "Volumes are normalised to a right knee. Four of the twelve targets are medial/lateral pairs. Left and right knees mirror on coronal and axial..." / "Laterality normalisation happens here: for a left knee, coronal and axial images are mirrored and the sagittal slice order is reversed, so ..."
**Confidence:** high (author EDA, not organizer statement).

**Claim:** Baseline EDA approach uses "six slots per study" — one series per {plane}×{Fluid/non-fluid} combination — as the canonical study representation.
**Source:** Same Roman Rozen notebook.
**URL:** https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline
**Date:** accessed 2026-08-10.
**Excerpt (verbatim, snippet):** "Part 3 — Sequence slots, laterality and DINOv2 features¶. Six slots per study, one series each: Slot, Selection score, Covers. {plane}_Fluid, 3 ..."
**Confidence:** medium-high (baseline design choice, not official structure).

---

## 3. Train/test split & supervision design

**Claim:** Only 58 of the 4,407 training studies (1.3%) carry expert per-condition labels; the other 4,349 have only the report. Reports exist for TRAIN ONLY — test.csv has no Report column, and the host confirmed test reports are never available.
**Source:** Kaggle Discussion 733932 (stevenleehans, "Not addressed is a label too"); Kaggle Discussion 733592.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592
**Date:** ~2026-08-08/09; accessed 2026-08-10.
**Excerpt (verbatim):** "train.csv has a Report column. test.csv does not. So text can only ever produce targets, never a model input. And only 58 of 4,407 studies carry real annotator labels for the twelve findings — 1.3%." / (733592, host reply) "The test set will not have 'Report' available, whether before or after notebook submission."
**Confidence:** high.

**Claim:** Data page frames this explicitly: "Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies."
**Source:** Kaggle Data page.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10.
**Confidence:** high.

**Claim:** Test set ≈ 1,300 studies; example test files (test.csv with 3 study IDs, test_series.csv, test_series/) are placeholders replaced by real data at scoring (code-competition / Kaggle evaluation API style). Public leaderboard = ~30% of test data; private = remainder.
**Source:** Kaggle Data page; Kaggle Leaderboard page.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Example test file with three study IDs from the public test set. During scoring, this example data will be replaced with the actual test data. There are about 1300 studies in the test set." / "This leaderboard is calculated with approximately 30% of the test data."
**Confidence:** high.

**Claim:** Test-set ground truth is expert-annotated (image review), same protocol as the 58 gold training studies; report text and image labels come from different processes and can legitimately disagree (reports over-call; annotators use stricter thresholds).
**Source:** RSNA press release; host reply in Discussion 733826 and 733491.
**URL:** https://www.rsna.org/media/press/2026/2669?PdfExport=1 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491
**Date:** 2026-08-05 / ~2026-08-08; accessed 2026-08-10.
**Excerpt (verbatim, press):** "The dataset used to assess model performance was annotated by expert radiologists."
**Excerpt (verbatim, host 733826):** "Were the labels assigned independently from the MRI images, rather than extracted from the reports? Yes. If image interpretation and report text disagree, should the image-derived label be considered authoritative? Yes." / "Discrepancies are plausible and expected because clinical reports typically involve one signing radiologist who created it for clinical care, and the image-based labels uses multiple readers with stricter image-based thresholds."
**Excerpt (verbatim, host 733491):** "The radiology reports in the training set are deidentified versions of the original contributor reports... The sample labels were assigned from image review using the challenge annotation rubric by two independent readers with a third adjudicating disagreements... The same process was used for the testing set."
**Confidence:** high.

**Claim:** Bilateral exams exist: both knees occasionally scanned under one StudyInstanceUID; organizers adjusted report text/metadata so participants can disambiguate.
**Source:** Host reply, Discussion 733826.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826
**Date:** ~2026-08-08; accessed 2026-08-10.
**Excerpt (verbatim):** "Yes. In clinical practice, both knees may occasionally be scanned under one StudyInstanceUID. For the challenge, each bilateral study or bilateral report was individually reviewed, and the released report text or DICOM metadata was adjusted as needed to provide sufficient information for participants to disambiguate the relevant study/studies."
**Confidence:** high.

**Claim:** Organizers warn prevalence is NOT matched across splits.
**Source:** Kaggle Data page ("Dataset Distribution Notice").
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets."
**Confidence:** high.

---

## 4. Class distribution / imbalance / label quality (EDA findings)

**Claim:** The 58 gold studies are enriched for pathology: every gold study has ≥1 positive finding, mean 4.14 findings per study — their prevalence is NOT representative of the test set.
**Source:** Roman Rozen EDA notebook; stevenleehans Discussion 733932.
**URL:** https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Mean findings per labelled study, 4.14, and no study with zero findings, The 58 are enriched for pathology — their prevalence is not the test ..." / "Gold prevalence is the annotator's sampling, not disease prevalence. Every gold study has at least one positive finding; mean 4.14 per study."
**Confidence:** high (two independent sources agree).

**Claim:** Within the 58 gold studies, synovitis is strikingly common (27/58) yet seldom written in reports; fracture has 18/58 positives.
**Source:** Discussion 733932; Roman Rozen notebook snippet (per-label stats table).
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 ; https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Yet 27 of the 58 gold studies have synovitis. It is common in the joint and seldom written down." / (Rozen per-label table row, verbatim snippet): "Fracture, 58, 18, 0.639, 0.564, 0.968, -0.006, 0.178, 0.850."
**Confidence:** high for 27/58 synovitis; medium for the Fracture row interpretation (table columns inferred).

**Claim:** Report-derived weak labels miss a lot: LLM extraction beats regex (macro AUC 0.8780 vs 0.8136 vs the 58 gold), and 25.4% of all report-label cells are "not addressed" (mapped to 0.5). Missingness is highly uneven: Synovitis 83.7% not-addressed, Baker's 48.2%, Fracture 42.9%, ACL 8.3%, Medial Meniscus 5.5%. "Silence is the label" for some findings (Baker's: 3% gold+ when silent vs 44% when mentioned) but uninformative for others (Synovitis: 34% vs 76%).
**Source:** Discussion 733932 (stevenleehans; labels released as dataset "RSNA Knee LLM Report Labels").
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932
**Date:** posted ~2026-08-09; accessed 2026-08-10.
**Excerpt (verbatim):** "An LLM does this measurably better than a regex (0.8780 vs 0.8136 against the 58 gold studies)... 25.4% of all label cells are 'the report does not address this'... finding / gold AUC / 'not addressed': Synovitis 0.678 83.7%; Baker's 0.946 48.2%; Fracture 0.793 42.9%; ACL 0.993 8.3%; Medial Meniscus 0.954 5.5%... When a radiologist does not mention a Baker's cyst, there almost certainly is not one — 3% versus 44%. The silence is the label... silence about synovitis still leaves a 34% chance it is present."
**Confidence:** high (community analysis; not organizer-published).

**Claim:** Report-vs-gold agreement measured independently: manual report-only re-read of 20 multilingual gold studies gave 82.5% overall agreement (198/240 cells), PPA 73.1%, positive recall 80.0% — confirming reports over-call relative to the rubric.
**Source:** Discussion 733826 (Nagoya Univ. Mori Lab Cho Royou).
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826
**Date:** ~2026-08-08; accessed 2026-08-10.
**Excerpt (verbatim):** "Across 240 decisions, the report-only labels matched the provided labels in 198 cases: Overall agreement: 82.5% Positive predictive agreement: 68/93 = 73.1% Positive recall: 68/85 = 80.0% TP: 68, FP: 25, FN: 17, TN: 130"
**Confidence:** high (community audit).

---

## 5. Report languages & multilingual handling

**Claim:** Official counts differ by page: RSNA.org challenge page says "nine languages"; the RSNA press release/news say "a dozen different languages"; the task framing says ~12 languages. 16 sites/institutions across 5 continents.
**Source:** RSNA.org challenge page; RSNA News/press release.
**URL:** https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge ; https://www.rsna.org/news/2026/august/ai-challenge-knee-mri
**Date:** accessed 2026-08-10 (press release dated 2026-08-05).
**Excerpt (verbatim):** "Your expertise helps create a one-of-a-kind dataset with over 5,000 knee MRI exams from 16 institutions worldwide, paired with corresponding MRI reports in nine languages!" / "The training dataset includes more than 5,000 knee MRI exams and the associated radiology reports in a dozen different languages from 16 sites worldwide."
**Confidence:** high that both statements exist; the exact language count is ambiguous (9 vs ~12) — flag discrepancy.

**Claim:** Community EDA of the actual reports: dominated by English, Spanish and Dutch, with French, German, Portuguese, Italian and Turkish in the tail; Bulgarian also observed in the gold set.
**Source:** Roman Rozen EDA notebook (snippet); Discussion 733826 examples (Spanish, German, Turkish, Bulgarian reports quoted).
**URL:** https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "Reports are dominated by English, Spanish and Dutch, with French, German, Portuguese, Italian and Turkish in the tail. Spanish and Dutch ..." / (733826, verbatim): "The German report explicitly states 'Baker-Zyste.'... The Turkish report states 'Lateral menisküs normal'... The Bulgarian report explicitly describes an osteochondral fracture."
**Confidence:** high for named languages; full 9–12-language list not yet published verbatim.

**Claim:** Organizer/community guidance on multilingual handling: commercially hosted LLMs and other external inference services ARE permitted (if rules-compliant); local LLMs are a common approach to derive labels from multilingual reports.
**Source:** Discussion 733965 ("Use of Commercially Hosted LLMs", host Po-Hao Chen); Discussion 733836.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733836
**Date:** ~2026-08-09/10; accessed 2026-08-10.
**Excerpt (verbatim):** "Use of commercially hosted LLMs and other external inference services is permitted, provided that the service and method of use otherwise comply with the ..." / (733836, k256.dev): "you could even do it by hand, although the reports are written in multiple languages, so in most cases that would be hard. As discussed in other threads, using a local LLM is one option here."
**Confidence:** high.

---

## 6. Public EDA notebooks & what early EDA revealed

Code page (sorted by hotness/votes, accessed 2026-08-10): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code

1. **"RSNA Knee Abnormalities - Efficiency LB" — Ryan Holbrook (Kaggle/RSNA staff), PINNED, ~100 upvotes, 2 comments.** Implements the public efficiency leaderboard referenced in the Evaluation tab. Confirms the dual-track scoring; a daily-updated public efficiency leaderboard showing team ranks only (not full scores). URL: https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb (accessed 2026-08-10). Excerpt (Overview, verbatim): "During the training period of the competition, you may see a leaderboard for the public test data in the following notebook, updated daily: Efficiency Leaderboard. After the competition ends, we will update this leaderboard with efficiency scores on the private data. During the training period, this leaderboard will show only the rank of each team, but not the complete score." Confidence: high.

2. **"RSNA Knee | Data structure, EDA, baseline🔥" — Roman Rozen, ~88–90 upvotes, public score 0.809 (best 0.894).** The reference EDA. Findings: 58/4,407 fully labelled studies ("58 is a test set, not a training set"); mean 4.14 findings per labelled study; report language distribution (English/Spanish/Dutch dominant); laterality normalization to right knee (mirror coronal/axial, reverse sagittal order for left knees); six-slot series selection (plane × fluid-sensitivity); frozen DINOv2 sequence features; report-teacher trained on reports, not on the 58. URL: https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline. Verbatim snippets: "Fully labelled studies, 58 (1.3%), Supervision must come from reports; 58 is a test set, not a training set" / "Frozen DINOv2 sequence features · laterality-normalised · report teacher trained on reports, not on 58 ..." Confidence: high.

3. **"RSNA Knee EDA the reports are train only" — Will/wguesdon.** Core counts: 4,407 training studies, 24,371 series, ~730,000 slices, 265 GB; emphasizes reports are train-only. URL: https://www.kaggle.com/code/wguesdon/rsna-knee-eda-the-reports-are-train-only. Confidence: high.

4. **"RSNA Knee: EDA to 2.5D" — Karnakbayev Artur** (pipeline EDA→2.5D baseline; referenced by 733932 as a measured pipeline). URL: https://www.kaggle.com/code/karnakbaevarthur/rsna-knee-eda-to-2-5d. Confidence: medium (page crashed on open; title/source verified via search index).

5. **"RSNA Knee Abnormality Detection - EDA" — mohammedkhamisamro** — DICOM tag-level EDA (extracts Rows/Columns etc. per slice). URL: https://www.kaggle.com/code/mohammedkhamisamro/rsna-knee-abnormality-detection-eda. Verbatim snippet: `"Rows": getattr(ds, "Rows", np.nan), "Columns": getattr(ds, "Columns", np.` Confidence: medium-high.

6. **"4407 Studies and 58 Labels"** (notebook on code page) — highlights the supervision gap in its title. Confidence: medium.

**Claim (EDA: DICOM metadata shortcut test):** DICOM header metadata alone yields macro AUC 0.6516 (random folds) vs 0.5981 (scanner-grouped folds); series composition alone (the 4 columns of train_series.csv) already gives 0.5954; 265 distinct scanner fingerprints (Manufacturer+Model+SoftwareVersions+ImagingFrequency+ReceiveCoilName), top 20 covering 45.5% of studies — i.e., substantial site/scanner heterogeneity but no exploitable metadata shortcut; LB scores >0.9 reflect real image reading.
**Source:** Discussion 733517 (Oleksii Zhukov, "0.932 LB within one day. Tested for DICOM metadata shortcut").
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517
**Date:** posted ~2026-08-07; accessed 2026-08-10.
**Excerpt (verbatim):** "Full header metadata, no pixels, reaches 0.6515 macro AUC under random folds and 0.5981 under scanner-grouped folds. The 0.053 increment is site memorization and it does not transfer to unseen scanners." / "Probe A: site identifiability... Result: 265 distinct fingerprints, top 20 covering 45.5% of studies." / "Series composition alone (no DICOM reads at all) gives 0.5954."
**Confidence:** high.

**Claim (EDA: label/encoder bottleneck):** Label quality, not image encoder, is the binding constraint; public LB passed 0.9 within ~a day (0.899–0.932 top notebooks by Aug 10, e.g. "[0.899] Let me Cook" with 72 upvotes, "exp-2" 0.899, "RSNA Knee Baseline V1" 0.899).
**Source:** Discussion 733932; Kaggle code page listing.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "That means almost everyone in this competition is training against labels they invented, and the quality of that invention is a ceiling nothing downstream can pass."
**Confidence:** high.

**Claim (derived-label datasets shared by community):** pilkwang "rsna-knee-llm-labels" (2026-08-06, first), barun2104 "Stratified Folds & LLM Soft Labels" (2026-08-07), lixin73 "LLM Report Labels (GPT-5.6-Sol)" (2026-08-08), stevenleehans "RSNA Knee LLM Report Labels".
**Source:** Discussion 733932 (credit list with links).
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932
**Confidence:** high.

---

## 7. Code requirements / efficiency track (dataset-relevant constraints)

**Claim:** Notebook-only submissions (CPU or GPU ≤ 9h), internet disabled, freely-public external data + pretrained models allowed, submission.csv; efficiency track scored by (runtime × normalized-score-gap) on private data.
**Source:** Kaggle Overview page.
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview
**Date:** accessed 2026-08-10.
**Excerpt (verbatim):** "CPU Notebook <= 9 hours run-time; GPU Notebook <= 9 hours run-time; Internet access disabled; Freely & publicly available external data is allowed, including pre-trained models; Submission file must be named submission.csv" / "we will evaluate submissions on both runtime and predictive performance... t is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score."
**Confidence:** high.

---

## 8. Gaps / unresolved items

- Exact image dimensions (Rows×Columns distribution) not yet captured verbatim from an EDA (mohammedkhamisamro's notebook extracts them but the notebook body would not render in automation; Kaggle notebook cells load via JS and were not readable via web_open_url or the browser text extractor).
- Complete authoritative list of all report languages (9 vs 12 discrepancy between RSNA pages) not resolved; only 9 languages named in community EDA (English, Spanish, Dutch, French, German, Portuguese, Italian, Turkish, Bulgarian).
- Full per-class prevalence table for the 58 gold studies exists in Roman Rozen's notebook (partially captured: Synovitis 27/58, Fracture 18/58) but the complete table was not extractable.
- "DICOM metadata findings: scanner-grouped CV and PatientSex priors" (morningduck) thread located in the discussion index but its URL/body was not retrieved before the step budget prioritized other sources.
- PatientSex priors and age distribution: age is NOT provided (only sex); no age field in train.csv schema.
