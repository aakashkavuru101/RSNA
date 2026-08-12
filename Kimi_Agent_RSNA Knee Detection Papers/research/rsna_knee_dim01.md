# RSNA Knee Abnormality Detection (Kaggle, 2026) — Dimension 01: Competition Mechanics & Strategy

Research date: 2026-08-10 (Day ~5–12 of competition; Kaggle header shows "Start 5 days ago", "2 months to go").
Competition URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
Type: "Research Code Competition" (Kaggle code/notebook competition).
Participation snapshot (2026-08-10): 8,892–8,899 Entrants, 1,011 Participants, 961 Teams, ~4,060 Submissions.

---

## 1. EVALUATION METRIC & SUBMISSION FORMAT

Claim: Main metric is macro-averaged AUC ROC over 12 binary targets (multi-label, NOT weighted — plain mean over the 12 per-class AUCs).
Source: Kaggle Evaluation page
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation
Date: accessed 2026-08-10
Excerpt: "Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets: ... The final score is, in other words, the macro-averaged AUC ROC."
(Rendered formula: "Final Score = 1/12 Σ_{i=0}^{11} AUC_i")
Confidence: high

Claim: Submission file is submission.csv with one row per test study and 12 confidence-score columns.
Source: Kaggle Evaluation page
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation
Date: accessed 2026-08-10
Excerpt: "For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format: StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture"
Confidence: high

Claim: The 12 target labels and their clinical definitions (from the Data page).
Source: Kaggle Data page
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
Date: accessed 2026-08-10
Excerpt: "Twelve binary labels: ACL - anterior cruciate ligament injury (0/1). MCL - medial collateral ligament injury (0/1). Medial Meniscus - medial meniscus tear (0/1). Lateral Meniscus - lateral meniscus tear (0/1). Medial OA - osteoarthritis of the medial tibiofemoral compartment (0/1). Lateral OA - osteoarthritis of the lateral tibiofemoral compartment (0/1). PF OA - patellofemoral osteoarthritis (0/1). Effusion - joint effusion / excess fluid (0/1). Synovitis - inflammation of the joint lining (0/1). Baker's - Baker's cyst (0/1). Contusion - bone contusion / bone bruise (0/1). Fracture - fracture (0/1)."
Confidence: high

Claim: No public scoring code repository was found; metric is standard sklearn macro ROC AUC. Competition tags confirm "Roc Auc Score".
Source: Kaggle competition header/tags
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation
Date: accessed 2026-08-10
Excerpt: "Tags: Image Classification, Image, Text, Computer Vision, Medicine, Roc Auc Score"
Confidence: high

## 2. RULES: EXTERNAL DATA, CODE COMP FORMAT, COMPUTE, INTERNET, TEAMS, LICENSE

Claim: This is a code competition — submissions via Kaggle Notebooks only; 9h CPU or 9h GPU runtime; internet disabled during submission; external data + pretrained models explicitly allowed; output must be submission.csv.
Source: Kaggle Overview — Code Requirements
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements
Date: accessed 2026-08-10
Excerpt: "Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met: CPU Notebook <= 9 hours run-time; GPU Notebook <= 9 hours run-time; Internet access disabled; Freely & publicly available external data is allowed, including pre-trained models; Submission file must be named submission.csv"
Confidence: high

Claim: External data is allowed if publicly available at no cost / "reasonably accessible to all" and of "minimal cost"; automated ML tools allowed with appropriate license.
Source: Kaggle Rules — Section 2.6 External Data and Tools
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
Date: accessed 2026-08-10
Excerpt: "You may use data other than the Competition Data ("External Data") to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria... The use of external data and models is acceptable unless specifically prohibited by the Host... their use must be 'reasonably accessible to all' and of 'minimal cost'... Purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition would not be considered reasonable."
Confidence: high

Claim: Host ruling (2026-08-09): sending report text to commercial LLM APIs (OpenAI/Anthropic/Google) for label extraction IS permitted; it is not "private sharing".
Source: Kaggle discussion — "Use of Commercially Hosted LLMs" (Po-Hao "Howard" Chen, Competition Host)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965
Date: posted ~2026-08-09 ("21 hours ago" on 2026-08-10)
Excerpt: "Use of commercially hosted LLMs and other external inference services is permitted, provided that the service and method of use otherwise comply with the Competition Rules, including requirements that external data, models, software, and associated tools be reasonably accessible to all participants and of minimal cost. In other words, for purposes of this competition, submitting Competition Data, including report text, to an external LLM or API for inference or other computational processing (for example, extracting labels from reports) will not, by itself, be considered prohibited PRIVATE SHARING of Competition Data outside the Team... The Competition Host reserves the right to determine whether a particular service, model, or configuration is reasonably accessible, is prohibitively costly, or otherwise creates an unfair competitive advantage."
Confidence: high

Claim: Max team size 5; 5 submissions/day; up to 2 final submissions for judging.
Source: Kaggle Rules — Competition-Specific Rules
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
Date: accessed 2026-08-10
Excerpt: "a. The maximum Team size is five (5)... a. You may submit a maximum of five (5) Submissions per day. b. You may select up to two (2) Final Submissions for judging."
Confidence: high

Claim: Winner license is CC-BY-NC 4.0; competition data governed by RSNA's MIRA license (commercial + academic research use allowed per rules text).
Source: Kaggle Rules — Competition-Specific Terms
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
Date: accessed 2026-08-10
Excerpt: "6. WINNER LICENSE TYPE: CC-BY-NC 4.0; 7. DATA ACCESS AND USE: Commercial and Academic Research - MIRA license... Open Source: You hereby license and will license your winning Submission and the source code used to generate the Submission under CC-BY-NC 4.0... In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s)."
(Note: the rules' license paragraph calls CC-BY-NC 4.0 "an Open Source Initiative-approved license... that in no event limits commercial use" — boilerplate inconsistent with CC-BY-NC's actual non-commercial clause; flagged, not resolved.)
MIRA terms: http://rsna.org/mira-license
Confidence: high (license type); medium (wording anomaly)

Claim: Winners' obligations: deliver training + inference code, model weights (as a public Kaggle dataset), environment (pip requirements.txt + Kaggle image or Dockerfile), plus a short video, public code/weights link on forum, and publicly distributable final model.
Source: Kaggle Rules Section 2.8 + Overview Prizes section
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/prizes
Date: accessed 2026-08-10
Excerpt (rules): "delivered software code should include training code, inference code, model weights, and a description of the required computational environment. The computational environment can take the form of a pip requirements.txt with one of: 1) the corresponding kaggle image or a 2) a dockerfile used to build the image. The model weights should be provided as a public kaggle dataset so it is both publicly accessible and linked to the inference/submission code."
Excerpt (overview): "the host team also asks that you: (i) create a short video presenting your approach and solution, and (ii) publish a link to your open sourced code and the weights on the competition forum (iii) Share final version of model as publicly available for open distribution and validation. Please see https://www.kaggle.com/models/tom99763/9th-place-models-rsna-iad/PyTorch/default as an example."
Confidence: high

Claim: Eligibility excludes residents of Crimea, so-called DNR/LNR, Cuba, Iran, North Korea, and sanctioned persons; employees of Sponsor/Kaggle may participate but cannot win prizes.
Source: Kaggle Rules — Eligibility
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
Date: accessed 2026-08-10
Excerpt: "not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, or North Korea; and not a person or representative of an entity under U.S. export controls or sanctions"
Confidence: high

## 3. PRIZE STRUCTURE & EFFICIENCY AWARDS

Claim: Total prize pool $77,000. Main leaderboard pays 10 places; Efficiency Track pays 3 places. A submission may win both.
Source: Kaggle Overview — Prizes / Rules Section 1.5
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/prizes
Date: accessed 2026-08-10
Excerpt: "Main Leaderboard: First Prize: $9,000; Second Prize: $7,000; Third Prize: $6,500; Fourth Prize: $6,000; Fifth Prize: $5,500; Sixth Prize: $5,000; Seventh Prize: $5,000; Eighth Prize: $5,000; Ninth Prize: $5,000; Tenth Prize: $5,000. Efficiency Track: First Efficiency Prize: $7,000; Second Efficiency Prize: $6,000; Third Efficiency Prize: $5,000."
(Arithmetic check: main = $59,000; efficiency = $18,000; total $77,000. ✔)
Confidence: high

Claim: Efficiency awards are measured by a combined AUC-and-runtime score — NOT model size/VRAM/energy. Only wall-clock runtime of the evaluation notebook and predictive performance count.
Source: Kaggle Overview — Efficiency Prize Evaluation
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation
Date: accessed 2026-08-10
Excerpt: "For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance. To be eligible for an Efficiency Prize, a submission: Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions automatically selected under the conditions described in the My Submissions tab. Must be ranked on the Private Leaderboard higher than the sample_submission.csv benchmark. All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize."
Rendered formula (MathML flattening): "Efficiency = AUC Benchmark − max AUC + RuntimeSeconds 32400" — interpreted as:
  Efficiency = (Benchmark − AUC) / (Benchmark − max AUC) + RuntimeSeconds / 32400
"where [AUC] is the submission's score on the main competition metric, [Benchmark] is the score of the benchmark sample_submission.csv, [max AUC] is the maximum [AUC] of all submissions on the Private Leaderboard, and [RuntimeSeconds] is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score."
(32,400 s = 9 h = the notebook runtime cap, i.e., runtime normalized to the limit. Fraction orientation of the AUC term is inferred from the minimization objective and the flattened MathML; exact numerator/denominator order should be verified against the rendered page.)
Confidence: high (runtime+AUC combined, minimize, eligibility rules); medium (exact algebraic arrangement of the AUC term)

Claim: GPU notebooks ARE eligible for the Efficiency Prize; RuntimeSeconds = full notebook wall time including package installs, model loading, and DICOM reading.
Source: Kaggle discussion 733475 — answer by Ryan Holbrook (Kaggle Staff)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475
Date: posted ~2026-08-07
Excerpt: "GPU notebooks are allowed. RuntimeSeconds is the time from when execution of the submitted notebook begins to the time when execution ends. If your notebook is installing packages, loading models, and reading DICOMs, then all of that is included in RuntimeSeconds."
Confidence: high

Claim: A public Efficiency Leaderboard notebook (by Ryan Holbrook, Kaggle staff) is updated daily during training phase; shows ranks only, not full scores; private efficiency scores computed after the end.
Source: Kaggle Overview — Efficiency Prize Evaluation + pinned notebook
URL: https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb
Date: accessed 2026-08-10 (notebook "Updated 8h ago", scheduled runs)
Excerpt: "During the training period of the competition, you may see a leaderboard for the public test data in the following notebook, updated daily: Efficiency Leaderboard. After the competition ends, we will update this leaderboard with efficiency scores on the private data. During the training period, this leaderboard will show only the rank of each team, but not the complete score."
Confidence: high

## 4. TIMELINE & LEADERBOARD STRUCTURE

Claim: Key dates (all 11:59 PM UTC).
Source: Kaggle Overview — Timeline
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline
Date: accessed 2026-08-10
Excerpt: "July 30, 2026 - Start Date. October 15, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete. October 15, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams. October 22, 2026 - Final Submission Deadline. November 5, 2026 - Winners' Requirement Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video and method description."
(Note: the Kaggle timeline lists July 30, 2026 as Start Date, while RSNA press/announcements date the public launch to ~Aug 5, 2026 and the Kaggle header showed "Start 5 days ago" on Aug 10. Minor discrepancy, likely soft-open vs announcement.)
Confidence: high

Claim: Two-stage leaderboard: public LB computed on ~30% of test data; final results on the other 70%. ~1,300 studies in the full test set.
Source: Kaggle Leaderboard page + Data page
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
Date: accessed 2026-08-10
Excerpt (LB): "This leaderboard is calculated with approximately 30% of the test data. The final results will be based on the other 70%, so the final standings may be different."
Excerpt (data): "There are about 1300 studies in the test set."
Confidence: high

Claim: Winners announced November 2026 and recognized in the AI Theater at RSNA 2026 (Nov 29–Dec 3, McCormick Place, Chicago); winners invited to AI Challenge Recognition Event with waived fee.
Source: RSNA News
URL: https://www.rsna.org/news/2026/august/ai-challenge-knee-mri
Date: 2026-08-06
Excerpt: "The competition will run through October 22, 2026. Winners will be announced in November, and winning teams will be recognized in the AI Theater during RSNA's 112th Scientific Assembly and Annual Meeting (RSNA 2026), held Nov. 29–Dec. 3 at McCormick Place in Chicago."
Confidence: high

## 5. DATA SPECIFICS (mechanics-relevant)

Claim: Train set = 4,407 studies; only 58 carry the 12 per-condition labels (all-or-nothing); all 4,407 have reports. ~1,300 test studies. Reports exist ONLY for training — test.csv has no Report column, so text is a label-source, never a model input at inference.
Source: Kaggle discussion 734055 (maximo lorenzo y losada) and discussion 733932 (stevenleehans)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932
Date: posted 2026-08-09/10
Excerpt (734055): "Of 4,407 studies, 58 carry labels. The other 4,349 have every label column empty. It is all-or-nothing per study: the 58 labelled studies have all twelve findings filled, and the rest have none... Every one of the 4,407 studies has a Report, including all 4,349 unlabelled ones... The file is 58,556 lines but only 4,407 rows. Reports contain newlines... Empty is not zero. The unlabelled cells are empty strings, not 0."
Excerpt (733932): "train.csv has a Report column. test.csv does not. So text can only ever produce targets, never a model input. And only 58 of 4,407 studies carry real annotator labels for the twelve findings — 1.3%."
Confidence: high

Claim: Data page (official) — labels scarce by design; DICOM details; prevalence not guaranteed consistent across splits.
Source: Kaggle Data page
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
Date: accessed 2026-08-10
Excerpt: "Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies... Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred... Series come in a mix of transfer syntaxes (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian). Every DICOM has been stripped to an allowlisted set of 86 metadata tags... Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets."
Files: 819,640 files, 569.76 GB (dcm, csv). train_series.csv provides Fluid_Sensitive, Fat_Suppression, Anatomical_Plane per series.
Confidence: high

Claim: Prevalence within the 58 labelled studies (enriched, not population prevalence): Effusion 60.3%, Synovitis 46.6%, Medial Meniscus 44.8%, ACL 41.4%, Lateral Meniscus 39.7%, PF OA 36.2%, Contusion 32.8%, Fracture 31.0%, Medial OA 25.9%, Baker's 20.7%, Lateral OA 19.0%, MCL 15.5%; mean 4.1 findings/labelled study.
Source: Kaggle discussion 734055
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055
Date: posted 2026-08-10
Excerpt: "Prevalence within the 58 labelled studies, which is emphatically not population prevalence — it looks like a deliberately enriched annotation set: Effusion 60.3%, Synovitis 46.6%, Medial Meniscus 44.8%, ACL 41.4%, Lateral Meniscus 39.7%, PF OA 36.2%, Contusion 32.8%, Fracture 31.0%, Medial OA 25.9%, Baker's 20.7%, Lateral OA 19.0%, MCL 15.5%. Mean 4.1 findings per labelled study, max 9 — heavily multi-label and co-occurring."
Confidence: high (community-computed, unchallenged)

Claim: Reports are multilingual and multi-script: ~97.6% contain Latin script, 7.3% Greek, 5.0% Cyrillic (~12% non-Latin; a report may contain >1 script). English+Spanish dominate Latin. Keyword/regex pipelines silently return confident negatives on non-Latin reports.
Source: Kaggle discussion 734055
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055
Date: posted 2026-08-10
Excerpt: "Over all 4,407 reports, and a report can contain more than one script: Latin 4,301 (97.6%), Greek 321 (7.3%), Cyrillic 220 (5.0%)... Roughly 12% of reports contain non-Latin script... a keyword or regex pipeline built on English or Spanish terms does not fail loudly on those reports. It extracts nothing and returns a confident negative for every finding."
(Note: RSNA's own pages variously say reports in "nine languages" (challenge page) and "a dozen/12 languages" (press release). Unresolved discrepancy.)
Confidence: high (script stats); medium (language count)

## 6. ANNOTATION PROTOCOL & LABEL DEFINITIONS (host-published)

Claim: Reference/test labels: double-read by subspecialty MSK radiologists with third-reader adjudication; borderline findings graded negative (specificity-favoring); exam-level labels for a single knee.
Source: Kaggle discussion 733343 — "Knee Abnormality Detection AI Challenge Overview" (pinned host post by Po-Hao "Howard" Chen, courtesy of Dr. Jacob Kazam)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
Date: posted ~2026-08-06
Excerpt: "Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee."
Excerpt: "In each case, ambiguous or borderline findings ('on the fence') were graded as negative to favor specificity."
Label criteria (verbatim): "ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted... MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament... Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment... Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness... Joint effusion: A moderate or large amount of fluid distending the joint. Synovitis: Inflammation and thickening of the synovial lining of the joint. Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee. Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line. Acute fracture: An acute cortical break or fracture line."
Also: "a 'fluid sensitive' sequence refers to one in which edema, hemorrhage, and other types of fluid appear bright and fat is suppressed in some way."
Confidence: high

Claim: Host confirmation — the 58 training labels are IMAGE-derived (not report-derived); where report and image labels disagree, the image-derived label is authoritative; bilateral knees can share a StudyInstanceUID and were manually disambiguated.
Source: Kaggle discussion 733826 — reply by Po-Hao "Howard" Chen (Competition Host)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826
Date: posted ~2026-08-08
Excerpt: "Were the labels assigned independently from the MRI images, rather than extracted from the reports? Yes. If image interpretation and report text disagree, should the image-derived label be considered authoritative? Yes. Note that only a small sample of provided data contains both. It is intended to help participants surface this conclusion... Do negative labels mean that the finding was confirmed absent...? It means the finding was annotated as absent using methods outlined in Overview discussion post under Label Description section... Yes. In clinical practice, both knees may occasionally be scanned under one StudyInstanceUID. For the challenge, each bilateral study or bilateral report was individually reviewed... Discrepancies are plausible and expected because clinical reports typically involve one signing radiologist who created it for clinical care, and the image-based labels uses multiple readers with stricter image-based thresholds."
(Community audit in same thread: report-only review of 20 studies vs provided labels: "Overall agreement: 82.5%; Positive predictive agreement: 68/93 = 73.1%; Positive recall: 68/85 = 80.0%" — i.e., reports are a noisy label source.)
Confidence: high

## 7. DISCUSSION / COMMUNITY INSIGHTS & EARLY PITFALLS

Claim: Public LB reached 0.9+ within ~1 day of launch; top public score 0.942 as of 2026-08-10. DICOM-metadata shortcut probe came back negative (no leak): metadata-only macro AUC 0.6516 (random folds) / 0.5981 (scanner-grouped).
Source: Kaggle discussion 733517 (Oleksii Zhukov) + Leaderboard page
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517
Date: posted ~2026-08-07
Excerpt: "TL;DR: DICOM metadata alone reaches 0.6515 macro AUC under random folds but only 0.598 across unseen scanners. Safe to say no meaningful shortcuts found. Leaderboard scores seem to reflect image reading... 265 distinct fingerprints, top 20 covering 45.5% of studies... Series composition alone (no DICOM reads at all) gives 0.5954."
Leaderboard excerpt: "1 Brandon Low 0.942 ... 2 Pizza Boy 0.940 ... 3 JOLEE 0.938" (accessed 2026-08-10; public LB, 30% of test data)
Confidence: high

Claim: The dominant early strategy is LLM-derived training labels from reports; LLM labelers beat regex (0.8780 vs 0.8136 macro AUC vs the 58 gold studies); ~25.4% of label cells are "not addressed" by the report; targeted gap-filling (e.g., synovitis from effusion) helps, blanket imputation hurts.
Source: Kaggle discussion 733932 (stevenleehans)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932
Date: posted ~2026-08-09
Excerpt: "An LLM does this measurably better than a regex (0.8780 vs 0.8136 against the 58 gold studies)... 25.4% of all label cells are 'the report does not address this'... LLM Synovitis field vs gold Synovitis: 0.6780; LLM Effusion field vs gold Synovitis: 0.7115... Filling only the undecided synovitis cells from effusion — never overriding an explicit statement — moves that column 0.678 → 0.790 and the whole key 0.8780 → 0.8873... The blanket version is worse than the targeted one... 58 studies is a small ruler. On it, differences below roughly 0.02 macro are not measurable."
Public label datasets cited: pilkwang "RSNA Knee LLM-read report labels" (rsna-knee-llm-labels, 2026-08-06, first), barun2104 "Stratified Folds & LLM Soft Labels" (2026-08-07), lixin73 "LLM Report Labels (GPT-5.6-Sol)" (2026-08-08).
Confidence: high

Claim: Early pitfalls catalogued by the community: (a) wc -l overstates train rows ~13x (embedded newlines in reports); (b) empty label cells ≠ 0 — filling with 0 invents ~4,349 false negatives per class; (c) English-only keyword extraction silently fails on Greek/Cyrillic reports; (d) U+03BC GREEK SMALL LETTER MU vs U+00B5 MICRO SIGN look identical but don't match; (e) scanner/site memorization does not transfer (group CV by scanner advised); (f) DICOM decode path is CPU-bound, matters for the 9h limit and efficiency track.
Source: Kaggle discussions 734055, 733517, 733475
URLs: as above
Date: 2026-08-07..10
Excerpt (734055): "Empty is not zero. The unlabelled cells are empty strings, not 0. Filling them with 0 invents roughly 4,349 unasserted negatives per class — my first pass did exactly that and produced a nonsense prevalence of 0.5% per finding."
Confidence: high

Claim: DINOv2/DINOv3 self-supervised backbones are the popular image encoders; one participant reported DINOv2 0.775 vs DINOv3 0.763 (V3 slightly worse in their setup). DINOv3 is not registered in Kaggle Models, so users ship weights via datasets (allowed).
Source: Kaggle discussion 733313 ("Using Dino3")
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733313
Date: posted ~2026-08-06..09
Excerpt: "From what I can tell, DINOv3 doesn't seem to be registered. Am I correct in assuming that we can't use it?" / "Dino v3 has been used in other competitions this year - so I think its good!" / "My V2 model scored 0.775 while pretty much the same V3 scored 0.763 - I was expecting some gain, for sure not a small loss." / "I am using it now for this competition, but training local and including the files in my dataset so I have no need to get to the model on kaggle."
Confidence: high

Claim: Top public/starter notebooks (by votes, 2026-08-10): pilkwang "RSNA Knee baseline v1" (256 votes, public 0.891, GPU T4 x2, ~6.5 min inference, uses DINOv2 + own LLM-read report labels + public weights); prvsiyan "RSNA Knee: read the report, then the knee" (106 votes, 0.899); romanrozen "RSNA Knee | Data structure, EDA, baseline" (88 votes, 0.894, includes the report extractor others reuse); wguesdon "RSNA Knee DINOv2 at meniscus resolution" (83 votes, 0.815); "RSNA Knee | DINOsaur V2" (68 votes, 0.808). Forks of pilkwang's baseline reach ~0.897–0.899. Simple EDA-to-2.5D baselines score ~0.63; pseudo-label baselines ~0.625.
Source: Kaggle Code page (sorted by votes)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code?sortBy=voteCount
Date: accessed 2026-08-10
Excerpt: "RSNA Knee baseline v1 ... Score: 0.891 ... 255 [votes]" ; "RSNA Knee: read the report, then the knee ... Score: 0.899 ... 106" ; "RSNA Knee | Data structure, EDA, baseline ... Score: 0.894 ... 88" ; "RSNA Knee DINOv2 at meniscus resolution ... Score: 0.815 ... 83" ; "RSNA Knee: EDA to 2.5D ... Score: 0.63 ... 28"
Confidence: high

Claim: Official getting-started resources + official competition Discord (discord.gg/kaggle); Discord channels are public but NOT staff-monitored — rulings only on the forum.
Source: Kaggle discussion 730709 (María Cruz, Kaggle Staff)
URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/730709
Date: posted ~2026-07-30
Excerpt: "In addition to this competition forum, you can continue the discussion in our official Kaggle Discord Server here: discord.gg/kaggle... Kaggle Staff and Hosts running competitions will not monitor Discord or be available to answer questions in Discord... Please keep important questions, insights, writeups, and other valuable conversation on the Kaggle forums."
Confidence: high

## 8. ORGANIZERS & METHODOLOGY BACKGROUND

Claim: Challenge co-leaders: Po-Hao "Howard" Chen, MD, MBA (Vice Chair for AI, Diagnostics Institute, Cleveland Clinic; MSK radiologist; chairs RSNA Informatics Policy Committee; Kaggle handle javaduke95) and Naveen Subhas, MD, MPH (Vice Chair of Clinical Operations, Cleveland Clinic Enterprise Imaging; MSK radiologist; MRI acceleration/MSK AI researcher). Full author/organizer list in the official citation.
Source: Kaggle Overview — Citation; RSNA News; howardpchen.com; Cleveland Clinic pages
URLs: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/citation ; https://www.rsna.org/news/2026/august/ai-challenge-knee-mri ; https://howardpchen.com/ ; https://my.clevelandclinic.org/departments/imaging/medical-professionals/fellowships/musculoskeletal-imaging-fellowship
Date: accessed 2026-08-10
Excerpt (citation): "Po-Hao “Howard” Chen, Naveen Subhas, Robyn Ball, Pieter Baeyens, Errol Colak, Ali Emami, Hillary Garner, Jacob Kazam, Hui-Ming Lin, Luciano Prevedello, Daniel Schneider, Jason Sho, Ryan Holbrook, and María Cruz. RSNA Knee Abnormality Detection. https://kaggle.com/competitions/rsna-knee-abnormality-detection, 2026. Kaggle."
Excerpt (RSNA News): "Po-Hao “Howard” Chen, MD, MBA, challenge co-leader, member of the RSNA Artificial Intelligence Committee and vice chair for Artificial Intelligence at the Cleveland Clinic Diagnostics Institute" ; "Naveen Subhas, MD, MPH, challenge co-leader and vice chair of Clinical Operations for Cleveland Clinic Enterprise Imaging"
Confidence: high

Claim: No methodology/annotation-protocol paper for THIS challenge has been published yet (as of 2026-08-10). RSNA recruited volunteer MSK radiologist annotators Feb–Mar 2026 with recognition "in associated research publications" promised. Precedent: prior RSNA challenges (e.g., 2024 LumbarDISC) published dataset papers later in Radiology: Artificial Intelligence; expect the same here.
Source: RSNA challenge page; themoonlight review of LumbarDISC paper
URL: https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge ; https://www.themoonlight.io/en/review/the-rsna-lumbar-degenerative-imaging-spine-classification-lumbardisc-dataset
Date: accessed 2026-08-10
Excerpt: "RSNA invites trained musculoskeletal radiologists to participate in the challenge by annotating knee MRI studies and reviewing associated reports. The work requires approximately 10 hours of commitment between February and March 2026. Those who complete the assignment will be recognized in associated research publications... Your expertise helps create a one-of-a-kind dataset with over 5,000 knee MRI exams from 16 institutions worldwide, paired with corresponding MRI reports in nine languages!"
(Note: the public clinical/annotation overview posted by the host — discussion 733343, "courtesy of Dr. Jacob Kazam" — is currently the closest thing to a published annotation protocol.)
Confidence: high (no paper found); medium (expectation of later publication)

Claim: Relevant prior art with the same 12-abnormality knee MRI formulation: Qiu et al., Nature Communications 2024, co-plane attention across MRI sequences, 1,748 subjects, 12 abnormality types, overall AUC 0.812.
Source: Nature Communications
URL: https://www.nature.com/articles/s41467-024-51888-4.pdf
Date: published 2024-09-02
Excerpt: "we collected the largest multi-sequence knee magnetic resonance imaging dataset involving the most comprehensive range of abnormalities, comprising 1748 subjects and 12 types of abnormalities. Our model achieved an overall area under the receiver operating characteristic curve score of 0.812."
Confidence: high

Claim: Press coverage confirms "firsts": first RSNA challenge on MSK MRI, first to combine images + multilingual report text, first with efficiency awards.
Source: RSNA press release / RSNA News / Radiology Business / Axis Imaging News
URLs: https://www.rsna.org/media/press/2026/2669 ; https://radiologybusiness.com/topics/artificial-intelligence/rsna-launches-new-knee-mri-ai-challenge ; https://axisimagingnews.com/imaging-news/associations/rsna-launches-2026-knee-mri-challenge
Date: 2026-08-03..06
Excerpt: "It is the first challenge to focus on musculoskeletal MRI, with knee MRI—the gold standard for diagnosing internal derangement of the knee—as its centerpiece... The challenge is also the first to combine medical images with multilingual radiology reports, creating a unique multimodal dataset." (Subhas)
Excerpt: "The top performing teams will share in a total of $77,000 in prize money, including for the first time awards for the most efficient models."
Confidence: high

## 9. STRATEGY SYNTHESIS (dimension-01 view)

1. Metric = unweighted macro AUC over 12 labels → rare/hard labels (Synovitis, Baker's, Fracture per community AUC-vs-gold table) contribute as much as easy ones; per-label performance matters equally.
2. The real competition is label engineering: only 58/4,407 train studies are image-labeled; reports are train-only and must be mined (LLM labelers permitted, incl. commercial APIs per host ruling of 2026-08-09). Image-derived labels trump report text where they conflict.
3. Report silence is informative per-finding (Baker's silence ≈ negative; synovitis silence ≈ uninformative) — treat "not addressed" as a first-class value.
4. Code competition: notebook submission, internet OFF at submission, 9h CPU/GPU cap; pretrained weights must ship via Kaggle datasets/models. 5 subs/day, 2 final selections, team ≤5.
5. Efficiency track ($18k) = minimize (normalized AUC gap to best) + (RuntimeSeconds/32,400); GPU allowed; full notebook wall time counts (installs, model load, DICOM decode — CPU-bound decode path is the lever); must beat the all-0.5 sample_submission benchmark on the private LB to be eligible.
6. Public LB = 30% of ~1,300 test studies; private = 70%; prevalence shift across splits explicitly warned by organizers → expect shakeup; grouped CV by scanner/site recommended (site memorization shown not to transfer).
7. Baselines: public notebooks ~0.89–0.90 within days (DINOv2 + LLM labels); LB top 0.942 on day 5; DICOM-metadata shortcut ruled out.
8. Winners: CC-BY-NC 4.0 license on code+weights, public weights as Kaggle dataset, short video, method description by Nov 5, 2026.

## 10. OPEN ITEMS / UNRESOLVED
- Exact algebraic form of the efficiency formula's AUC term (MathML flattens ambiguously): "(Benchmark − AUC)/(Benchmark − max AUC) + RuntimeSeconds/32400" is the best-supported reading; verify against rendered page.
- Report language count: RSNA pages say both "nine languages" and "a dozen different languages".
- Kaggle timeline start (July 30) vs public launch (~Aug 5) discrepancy.
- MIRA license full text (http://rsna.org/mira-license) not fetched; a "Clarification on MIRA Section 6" thread (hangglider5) was unanswered at research time.
- Gated click-through datasets (MRNet, fastMRI+, OAI, SKM-TEA, KneeCoT): question raised (threads 733652, KneeCoT thread by Tiago Mazzutti); host answered the LLM half but had NOT ruled on gated research-agreement datasets as of 2026-08-10.
