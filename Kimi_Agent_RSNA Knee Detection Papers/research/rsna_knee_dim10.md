# Dimension 10 Research: Multilingual Radiology Report NLP & RSNA Annotation Protocol
## Kaggle "RSNA Knee Abnormality Detection" (2026)
**Research date:** 2026-08-10 | **Agent:** research sub-agent (dimension 10)
**Citation format:** Claim / Source / URL / Date / Excerpt (verbatim) / Confidence

---

## 1. RSNA CHALLENGE ANNOTATION PROTOCOL

### 1.1 Challenge identity, scale, leadership

**Claim:** The 2026 RSNA Knee Abnormality Detection AI Challenge launched Aug 5, 2026, runs through Oct 22, 2026, is hosted on Kaggle, offers $77,000 (including first-ever efficiency awards), and is co-led by Po-Hao "Howard" Chen (Cleveland Clinic) and Naveen Subhas (Cleveland Clinic).
**Source:** RSNA Press Release
**URL:** https://www.rsna.org/media/press/2026/2669
**Date:** 2026-08-05
**Excerpt:** "OAK BROOK, Ill. (August 5, 2026) – The Radiological Society of North America (RSNA) has launched the 2026 RSNA Knee Abnormality Detection AI Challenge... The competition will be conducted on a platform provided by Kaggle, Inc. The top performing teams will share in a total of $77,000 in prize money, including for the first time awards for the most efficient models. The competition will run through October 22, 2026."
**Confidence:** High

**Claim:** Training data = >5,000 knee MRI exams with reports in ~12 languages from 16 sites across five continents; test/evaluation set expert-annotated.
**Source:** RSNA Press Release 2669
**URL:** https://www.rsna.org/media/press/2026/2669
**Date:** 2026-08-05
**Excerpt:** "The training dataset includes more than 5,000 knee MRI exams and the associated radiology reports in a dozen different languages from 16 sites worldwide. The dataset used to assess model performance was annotated by expert radiologists."
**Confidence:** High

**Claim:** Co-leader Po-Hao Chen explicitly frames report-derived learning as the core design: participants must learn from real-world reports, not tidy tables.
**Source:** RSNA Press Release 2669
**URL:** https://www.rsna.org/media/press/2026/2669
**Date:** 2026-08-05
**Excerpt:** "Participants must learn from real-world diagnostic radiology reports, where findings are complex and answers are not neatly organized in a table. This brings the challenge closer to how clinical AI must actually be developed: by confronting the nuance and variability of real radiologic interpretation."
**Confidence:** High

### 1.2 The 12 labels (verbatim definitions from Kaggle Data page)

**Claim:** Twelve binary study-level labels: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
**Source:** Kaggle competition Data page
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10
**Excerpt:** "This dataset contains knee MRI studies annotated for twelve common findings: ligament and meniscus injuries, three compartments of osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and fracture... Twelve binary labels: ACL - anterior cruciate ligament injury (0/1). MCL - medial collateral ligament injury (0/1). Medial Meniscus - medial meniscus tear (0/1). Lateral Meniscus - lateral meniscus tear (0/1). Medial OA - osteoarthritis of the medial tibiofemoral compartment (0/1). Lateral OA - osteoarthritis of the lateral tibiofemoral compartment (0/1). PF OA - patellofemoral osteoarthritis (0/1). Effusion - joint effusion / excess fluid (0/1). Synovitis - inflammation of the joint lining (0/1). Baker's - Baker's cyst (0/1). Contusion - bone contusion / bone bruise (0/1). Fracture - fracture (0/1)."
**Confidence:** High

### 1.3 Label definitions & annotator criteria (pinned Overview post)

**Claim:** Each label has explicit positivity thresholds; borderline ("on the fence") findings were graded NEGATIVE to favor specificity.
**Source:** Kaggle pinned discussion "Knee Abnormality Detection AI Challenge Overview" (host Po-Hao Chen, courtesy of Dr. Jacob Kazam)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
**Date:** posted ~2026-08-06
**Excerpt (verbatim criteria):**
- "In each case, ambiguous or borderline findings ('on the fence') were graded as negative to favor specificity."
- "ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted... Mild signal change, degeneration, or thickening without discontinuity is graded negative."
- "MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament... Low-grade sprains and chronic or remote stress changes are graded negative."
- "Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment... Intrasubstance degeneration that does not reach the surface is negative."
- "Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment..."
- "Joint effusion: A moderate or large amount of fluid distending the joint."
- "Synovitis: Inflammation and thickening of the synovial lining of the joint."
- "Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee."
- "Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line."
- "Acute fracture: An acute cortical break or fracture line."
**Confidence:** High

### 1.4 Reference-set annotation protocol (2 readers + adjudication)

**Claim:** Annotated reference set: each study independently labeled by TWO subspecialty-trained MSK radiologists, disagreements adjudicated by a THIRD radiologist; labels are exam-level for a single knee.
**Source:** Kaggle pinned Overview discussion 733343
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
**Date:** ~2026-08-06
**Excerpt:** "Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee."
**Confidence:** High

**Claim:** RSNA ran a volunteer annotator program: trained MSK radiologists annotating knee MRI studies + reviewing reports, ~10 hours during February–March 2026, recognized in publications.
**Source:** RSNA challenge page (Knee Abnormality Detection AI Challenge)
**URL:** https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
**Date:** accessed 2026-08-10 (page live since Spring 2026)
**Excerpt:** "RSNA invites trained musculoskeletal radiologists to participate in the challenge by annotating knee MRI studies and reviewing associated reports. The work requires approximately 10 hours of commitment between February and March 2026. Those who complete the assignment will be recognized in associated research publications and acknowledged in AI Challenge communications."
**Confidence:** High

### 1.5 CRITICAL: Training labels are IMAGE-derived, NOT report-derived

**Claim:** Only 58 of 4,407 training studies carry expert labels; the other 4,349 have reports only. Host confirmed labels were assigned from images independently of reports, and image-derived labels are authoritative when they disagree with report text.
**Source:** Kaggle discussion "Possible inconsistencies between MRI reports and provided labels" (host reply by Po-Hao Chen)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826
**Date:** posted ~2026-08-08
**Excerpt (participant measurement):** "I performed a preliminary report-only review on 20 multilingual studies selected from the 58 studies with complete labels... Across 240 decisions, the report-only labels matched the provided labels in 198 cases: Overall agreement: 82.5%... Positive predictive agreement: 68/93 = 73.1%... Positive recall: 68/85 = 80.0%"
**Excerpt (host answers, verbatim):** "Were the labels assigned independently from the MRI images, rather than extracted from the reports? — Yes. ... If image interpretation and report text disagree, should the image-derived label be considered authoritative? — Yes. Note that only a small sample of provided data contains both. It is intended to help participants surface this conclusion. ... Discrepancies are plausible and expected because clinical reports typically involve one signing radiologist who created it for clinical care, and the image-based labels uses multiple readers with stricter image-based thresholds."
**Excerpt (bilateral handling):** "Yes. In clinical practice, both knees may occasionally be scanned under one StudyInstanceUID. For the challenge, each bilateral study or bilateral report was individually reviewed, and the released report text or DICOM metadata was adjusted as needed to provide sufficient information for participants to disambiguate the relevant study/studies."
**Confidence:** High

**Claim:** RSNA did NOT produce report-derived labels. Producing labels from the reports is the participants' task ("you may wish to derive the labels for the remaining studies"). test.csv has NO Report column — text is a train-only teacher.
**Source:** Kaggle Data page + discussion 734095
**URLs:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734095
**Date:** 2026-08
**Excerpt (Data page):** "Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies." / "Report - the free-text radiology report. May be in any of several languages, depending on the reporting institution."
**Excerpt (734095):** "train.csv has 4,407 studies but only 58 carry the 12 labels. The other 4,349 have a radiology report and nothing else, so the report is a train-only teacher. test.csv is study ids only, so whatever labels you derive, the model that actually scores has to read pixels."
**Confidence:** High

**Claim:** Test set ~1,300 studies; dataset 569.76 GB DICOM; class prevalence may differ across train/public/final evaluation.
**Source:** Kaggle Data page
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
**Date:** accessed 2026-08-10
**Excerpt:** "There are about 1300 studies in the test set." / "Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets."
**Confidence:** High

**Claim:** No arXiv/Radiology:AI methodology paper for this challenge exists as of 2026-08-10 (challenge runs to Oct 22; winners announced Nov 2026 at RSNA 2026). Past RSNA challenges publish dataset papers post hoc (e.g., RSNA Cervical Spine Fracture CT Dataset, Radiology:AI). Methodology detail lives on RSNA pages + pinned Kaggle posts.
**Source:** search results (absence of hits) + RSNA press release
**URL:** https://www.rsna.org/media/press/2026/2669 ; https://www.researchgate.net/publication/373539136 (prior-challenge precedent)
**Date:** 2026-08-10
**Excerpt:** "Winners will be announced in November, and winning teams will be recognized in the AI Theater during RSNA's 112th Scientific Assembly and Annual Meeting (RSNA 2026)"
**Confidence:** Medium-High (absence of evidence; paper likely forthcoming)

### 1.6 Rules: commercial LLM APIs ARE permitted (official ruling)

**Claim:** Host officially permits sending report text to commercial LLM APIs for label extraction, subject to accessibility/minimal-cost rules; submission notebooks remain offline.
**Source:** Kaggle discussion "Use of Commercially Hosted LLMs" (Competition Host)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965
**Date:** ~2026-08-09
**Excerpt:** "Use of commercially hosted LLMs and other external inference services is permitted, provided that the service and method of use otherwise comply with the Competition Rules, including requirements that external data, models, software, and associated tools be reasonably accessible to all participants and of minimal cost. In other words, for purposes of this competition, submitting Competition Data, including report text, to an external LLM or API for inference or other computational processing (for example, extracting labels from reports) will not, by itself, be considered prohibited PRIVATE SHARING of Competition Data outside the Team."
**Corollary (participant reading, discussion 733652):** "with open weights run locally or in a Kaggle notebook the text never leaves your own environment, and internet-off only applies to the submission notebook, so offline label generation during development isn't restricted. Small multilingual models are great for this task."
**Confidence:** High

---

## 2. THE ~12 REPORT LANGUAGES ("nine" vs "a dozen" RESOLVED)

**Claim:** RSNA's own pages disagree: challenge page volunteer section says "nine languages"; press release says "a dozen different languages." Community EDA identifies ~9 clearly attested languages, with additional rare ones (Bulgarian, Greek) explaining the "dozen" rounding.
**Source 1:** RSNA challenge page
**URL:** https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
**Date:** accessed 2026-08-10
**Excerpt:** "Your expertise helps create a one-of-a-kind dataset with over 5,000 knee MRI exams from 16 institutions worldwide, paired with corresponding MRI reports in nine languages!"
**Source 2:** RSNA Press Release 2669
**URL:** https://www.rsna.org/media/press/2026/2669
**Excerpt:** "the associated radiology reports in a dozen different languages from 16 sites worldwide."
**Confidence:** High (discrepancy documented verbatim)

**Claim:** Language distribution (participant EDA): English, Spanish, Dutch dominant; French, German, Portuguese, Italian, Turkish in the tail; Bulgarian and Greek additionally observed in labeled-set reports. Attested list ≈ {English, Spanish, Dutch, French, German, Portuguese, Italian, Turkish, Bulgarian, Greek} (~10, "nine" if counting only the languages RSNA had confirmed at volunteer-call time; "a dozen" as a rounded figure).
**Source 1:** Roman Rozen, Kaggle notebook "RSNA Knee | Data structure, EDA, baseline" (LB 0.809–0.894)
**URL:** https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline
**Date:** 2026-08-06
**Excerpt:** "Reports are dominated by English, Spanish and Dutch, with French, German, Portuguese, Italian and Turkish in the tail."
**Source 2:** yuki16, Kaggle notebook "RSNA_Knee_2D_CNN_StudyLevel baseline"
**URL:** https://www.kaggle.com/code/yuki16/rsna-knee-2d-cnn-studylevel-baseline
**Date:** 2026-08-08
**Excerpt:** "Rough language-hint distribution over a random sample of 300 reports: Report english 101 spanish 50 unknown/other 49 turkish 38 german 21 ..."
**Source 3:** Kaggle discussion 733826 (report audit) — verbatim mentions: a Spanish report ("Leve derrame articular con mínima sinovitis"), a German report ("Baker-Zyste"), a Turkish report ("Lateral menisküs normal"), a Bulgarian report describing an osteochondral fracture.
**Source 4:** Kaggle discussion 734095 (Busya PRIME rule labeler)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734095
**Date:** 2026-08-10
**Excerpt:** "It reads each report in its own language (a crude detector finds at least eight, and reading them I hit English, Spanish, German, Dutch, French, plus Greek and Turkish), fires a per-finding vocabulary, and runs a negation and uncertainty check on every mention so that 'no meniscal tear', 'sin rotura', 'geen scheur' and 'kein' count as negative instead of positive."
**Confidence:** Medium-High for the attested set (participant-derived, not official); note 16 institutions vs ~9–12 languages implies several institutions share languages (e.g., multiple Spanish- or English-language sites across continents).

---

## 3. MULTILINGUAL NLP OPTIONS — EVIDENCE

### 3.1 XLM-RoBERTa (multilingual encoder baseline)

**Claim:** XLM-R: single encoder pretrained on 100 languages (2.5TB CommonCrawl), 250k SentencePiece vocab; strong zero-shot cross-lingual transfer (XNLI 80.9% avg trained English-only; 83.6% translate-train-all).
**Source:** Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (arXiv:1911.02116)
**URL:** https://arxiv.org/pdf/1911.02116
**Date:** 2020 (v2 Apr 2020)
**Excerpt:** "On cross-lingual transfer, XLM-R obtains 80.9% accuracy, outperforming the XLM-100 and mBERT open-source models by 10.2% and 14.6% average accuracy... Using the multilingual training of translate-train-all, XLM-R further improves performance and reaches 83.6% accuracy, a new overall state of the art for XNLI."
**Confidence:** High

### 3.2 Multilingual E5 embeddings

**Claim:** multilingual-e5-large / -instruct: XLM-R-large backbone, ~100 languages, 1024-dim, 512-token context; contrastive pretraining on ~1B multilingual pairs + supervised fine-tuning. mE5-large-instruct was the best model in the initial MMTEB release.
**Source:** AfriMTEB/AfriE5 paper baseline descriptions (arXiv:2510.23896) + COMPASS paper (arXiv:2604.20720)
**URLs:** https://arxiv.org/html/2510.23896v1 ; https://arxiv.org/html/2604.20720v1
**Date:** 2025–2026
**Excerpt:** "mE5-large and mE5-large-instruct (Wang et al., 2024c) are multilingual members of the E5 family built on XLM-RoBERTa-large (Conneau et al., 2020), trained with a two-stage recipe that first performs weakly supervised contrastive pre-training on roughly one billion multilingual text pairs... Both use a 24-layer encoder that produces 1,024-dimensional vectors and inherit broad (~100 language) coverage from the XLM-RoBERTa backbone."
**Confidence:** High

### 3.3 BGE-M3 (and 2026 alternatives: Qwen3-Embedding)

**Claim:** BGE-M3: 100+ languages, 8192-token context, dense+sparse+ColBERT multi-function, MIT license, ~568M params, ~63 MTEB-multilingual; Qwen3-Embedding-8B tops MTEB multilingual at 70.58 (Apache 2.0, sizes 0.6B/4B/8B).
**Source:** BGE-M3 paper description (arXiv:2510.23896 App. A.2); lelabdev embedding benchmark (GitHub)
**URLs:** https://arxiv.org/html/2510.23896v1 ; https://github.com/lelabdev/embedding-benchmark
**Date:** 2026-06-30 (benchmark)
**Excerpt:** "BGE-M3 (BAAI) is a versatile embedding model unifying three capabilities in a single encoder: Multi-Functionality (dense, multi-vector, and sparse retrieval), Multi-Linguality (100+ languages), and Multi-Granularity (robust from short queries to long documents, up to ~8,192 tokens)." / Benchmark table: "Qwen3-Embedding-8B 70.58 (#1)... BGE-M3 ~63.0 — Multi-function (dense + sparse + multi-vector), 100+ languages."
**Confidence:** High

### 3.4 Translate-to-English pipelines (NLLB and caveats)

**Claim:** NLLB-200 (distilled 600M/1.3B) covers 200 languages, runs offline on Kaggle (CTranslate2 int8 variants exist), CC-BY-NC 4.0 license; BUT its own model card explicitly warns against medical-domain use and >512-token inputs.
**Source:** NLLB-200 model card (via modelscope mirror) + picovoice open-source translation guide
**URLs:** https://www.modelscope.cn/models/facebook/nllb-200-1.3B/summary ; https://picovoice.ai/blog/open-source-translation/
**Date:** model card 2022; blog 2025-12-11
**Excerpt (model card, verbatim):** "NLLB-200 is trained on general domain text data and is not intended to be used with domain specific texts, such as medical domain or legal domain. The model is not intended to be used for document translation. The model was trained with input lengths not exceeding 512 tokens, therefore translating longer sequences might result in quality degradation." / Blog: "Meta's NLLB family spans from 54.5 billion to 600 million parameters. The smallest variant, NLLB-200-distilled-600M, ranks as Hugging Face's most popular translation model."
**Confidence:** High

**Claim:** GPT-4o translation of radiology reports: high but imperfect fidelity — factual correctness averaged 79% (English 84%, French 83%, Russian 69%); potentially harmful errors in 4% of translations. Lesson: translation preserves ranking signal but mangles negation/hedges often enough to matter for label extraction.
**Source:** Rofo/European Radiology experimental two-center study, "Evaluation of GPT-4o for multilingual translation of radiology reports across imaging modalities"
**URL:** https://www.sciencedirect.com/science/article/pii/S0720048X25004279
**Date:** 2025-08-29
**Excerpt:** "Factual correctness averaged 79 %, with English (84 %) and French (83 %) outperforming Russian (69 %) (each p < 0.05). Potentially harmful errors were identified in 4 % of translations, primarily in Russian (9 %)."
**Confidence:** High

**Claim:** A Seoul Asan Medical Center study deliberately did NOT translate mixed-language (Korean/English) radiology reports, citing degraded translation quality in domain-specific mixed-language contexts, and their pipeline stayed robust.
**Source:** PMC, "Two stage large language model approach enhancing entity classification and relationship mapping in radiology reports"
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12391506/
**Date:** 2025
**Excerpt:** "We did not perform any translation or language normalization, as most key medical terms were already written in English, and translation quality between English and non-English languages can degrade significantly in domain-specific, mixed-language contexts."
**Confidence:** High

**Claim:** Single multilingual models beat per-language models for radiology report tasks (summarization across EN/PT/DE).
**Source:** arXiv:2310.00100, "Multilingual Natural Language Processing Model for Radiology Reports — The Summary is all you need!"
**URL:** https://arxiv.org/html/2310.00100v4
**Date:** 2024-01-13
**Excerpt:** "Training a single model such as M^{rr-1000}_{EN,PT,GE} to summarize radiology reports in multiple languages produces better results than training separate models for each language... the model is also able to apply language transfer learning in the summarization task, meaning that knowledge acquired in one language can improve performance in another."
**Confidence:** High

### 3.5 LLM structured extraction (JSON labels) — accuracy numbers

**Claim:** Fine-tuned open-weight LLMs (Llama/DeepSeek, QLoRA + CoT distillation from GPT-4o) reach 97–98% feature-extraction accuracy on abdominal MRI/CT reports, matching GPT-4o (97%), with radiologist-level agreement (Fleiss' κ 0.89–0.90 vs human κ 0.888).
**Source:** Rasromani et al., "Leveraging Fine-Tuned Large Language Models for Interpretable Pancreatic Cystic Lesion Feature Extraction and Risk Categorization" (arXiv:2507.19973; Radiology 2026)
**URL:** https://arxiv.org/abs/2507.19973
**Date:** 2025-07-26
**Excerpt:** "CoT fine-tuning improved feature extraction accuracy for LLaMA (80% to 97%) and DeepSeek (79% to 98%), matching GPT-4o (97%). Risk categorization F1 scores also improved (LLaMA: 0.95; DeepSeek: 0.94)... Radiologist inter-reader agreement was high (Fleiss' Kappa = 0.888) and showed no statistically significant difference with the addition of DeepSeek-FT-CoT (Fleiss' Kappa = 0.893)."
**Confidence:** High

**Claim:** Prompt-only small open LLM (GPT-OSS-20B) with anatomy-aware prompting beat supervised encoder baselines (BioClinicalModernBERT/ModernBERT) on German CT incidentaloma classification: incidentaloma macro-F1 0.79 vs 0.70; anatomy-aware prompting adds up to +0.08 macro-F1; prompt-only GPT-4o (with anatomy) 0.77 — at/above inter-annotator agreement macro-F1 0.76.
**Source:** arXiv:2512.05537
**URL:** https://arxiv.org/pdf/2512.05537
**Date:** 2025-12
**Excerpt:** "the best overall performance was achieved by the GPT-OSS-20b (With Anatomy) model, which obtained the highest F1-scores across all labels and an incidentaloma-positive macro-F1 of 0.79... GPT-4o (Base) already matched the strongest non-LLM baselines, and adding anatomical context further improved performance, yielding up to a Δ+0.08 increase in macro-F1 (classes 1-2)."
**Confidence:** High

**Claim:** Cautionary low-resource evidence: Llama-3.1-8B on Hebrew radiology reports achieved only F1 0.397 baseline (0.479 after uncertainty filtering) — small LLMs can fail on low-resource languages; prompt-ensemble uncertainty (BayesPE) helps.
**Source:** arXiv:2502.01691, "Agent-Based Uncertainty Awareness Improves Automated Radiology Report Labeling with an Open-Source Large Language Model"
**URL:** https://arxiv.org/abs/2502.01691
**Date:** 2025-02-02
**Excerpt:** "Structured data extraction was performed using Llama 3.1 (Llama 3-8b-instruct) with Bayesian Prompt Ensembles (BayesPE)... The agent-based model outperformed the baseline across all metrics, achieving an F1 score of 0.3967... After filtering high-uncertainty cases (≥0.5), the F1 score improved to 0.4787, and Kappa increased to 0.4258."
**Confidence:** High

**Claim:** LLM entity extraction on MIMIC-CXR RadGraph schema: GPT-4.1 entity F1 0.826, Llama-4-Maverick 0.841 (5-shot); zero-shot drops 8–12 points; both overconfident on hedged language ("cannot exclude," "possibly representing") — the uncertainty category is hardest (ECE > 0.40).
**Source:** arXiv:2603.00924v2, "Conformal Prediction for Risk-Controlled Medical Entity Extraction Across Clinical Domains"
**URL:** https://arxiv.org/html/2603.00924v2
**Date:** 2026-03-09
**Excerpt:** "With 5-shot prompting, GPT-4.1 achieves entity F1 of 0.826 (precision 0.798, recall 0.856)... Llama-4-Maverick achieves entity F1 of 0.841... OBS-U (uncertain observations) remains the most challenging category for both models, with ECE exceeding 0.40, reflecting the inherent ambiguity of hedging language in radiology reports (e.g., 'cannot exclude,' 'possibly representing')."
**Confidence:** High

**Claim:** Fine-tuned small medical LLM (MediPhi-Instruct 4B) reached micro-F1 87.8% for multi-label billing-code classification from 500k German radiology reports; on cleaned data GPT-5 F1 89.5% beat fine-tuned models.
**Source:** European Radiology (Springer), "Comparison of proprietary and fine-tuned large language models for multi-label classification of billing codes from radiology reports"
**URL:** https://link.springer.com/article/10.1007/s00330-026-12445-3
**Date:** 2026-03-14
**Excerpt:** "The fine-tuned model achieved an accuracy of 77.15% ± 0.47% and a micro-average F1-score of 87.79% ± 0.31% on the hold-out test set... For the cleaned dataset of 350 samples, GPT-5 achieved the best F1-score of 89.51 ± 1.52%."
**Confidence:** High

**Claim:** LLM multilingual coverage for offline Kaggle use: Qwen2.5 supports 29+ languages (128k context, strong JSON output); Qwen3 trained on 119 languages/36T tokens; Llama 3.1/3.3 officially only 8 languages; Gemma 3 140+ languages; Mistral Small 3.2 "dozens of languages."
**Source:** Qwen3 Technical Report (arXiv:2505.09388); South Slavic benchmarking paper (arXiv:2511.07989) model notes
**URLs:** https://arxiv.org/html/2505.09388v1 ; https://arxiv.org/html/2511.07989v1
**Date:** 2025
**Excerpt:** "All Qwen3 models are trained on a large and diverse dataset consisting of 119 languages and dialects, with a total of 36 trillion tokens." / "LLaMA 3.3 model... is reported to support only 8 languages, namely, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai." / "Gemma 3 model... is reported to support over 140 languages."
**Confidence:** High

**Claim:** Medical multilingual LLM/benchmark: MMedC (25.5B tokens, 6 languages EN/ZH/JA/FR/RU/ES) and MMedBench (8,518 test QA pairs, 6 languages); MMedLM 2 (7B) rivaled GPT-4 on MMedBench.
**Source:** Qiu et al., "Towards Building Multilingual Language Model for Medicine" (arXiv:2402.13963)
**URL:** https://arxiv.org/pdf/2402.13963.pdf
**Date:** 2024-02
**Excerpt:** "we construct a new multilingual medical corpus, that contains approximately 25.5B tokens encompassing 6 main languages, termed as MMedC... our final model, termed as MMedLM 2, with only 7B parameters, achieves superior performance compared to all other open-source models, even rivaling GPT-4 on MMedBench."
**Confidence:** High

---

## 4. REPORT STRUCTURE MINING, NEGATION & UNCERTAINTY (MULTILINGUAL)

### 4.1 CheXpert labeler (the canonical 3-stage design + numbers)

**Claim:** CheXpert labeler = mention extraction → mention classification (negated/uncertain) → mention aggregation (positive > uncertain > negative); performance micro-F1: mention 0.969, negation 0.952, uncertainty 0.848. Uncertainty ("u") is a first-class label.
**Source:** Emergent Mind topic summaries of Irvin et al. 2019 + McDermott et al. 2020
**URL:** https://www.emergentmind.com/topics/chexpert-labeler ; https://www.emergentmind.com/topics/uncertainty-aware-chexpert-style-labels
**Date:** updated 2026-01-08 / 2026-02-05
**Excerpt:** "Validation on manually annotated datasets has demonstrated the CheXpert labeler's effectiveness: Mention micro-F1 0.969 (macro 0.948); Negation micro-F1 0.952 (macro 0.899); Uncertainty micro-F1 0.848 (macro 0.770)." / "An explicit aggregation rule gives precedence to positive > uncertain > negative, ensuring that 'cannot exclude pneumonia' yields 'uncertain', while an outright negative phrase yields 'negative'."
**Confidence:** High (numbers consistent across sources; primary: Irvin et al., AAAI 2019)

### 4.2 Impression vs Findings sections

**Claim:** Standard practice: extract labels from the Impression section (CheXpert labeler applied to Impression in MIMIC-CXR pipelines); Findings carries detail, Impression carries clinically significant conclusions. In this competition reports may lack clean section headers in all languages, so section splitting itself is a per-language task.
**Source:** CXR-LanIC paper (arXiv:2510.21464) + ContrastConnect explainer
**URLs:** https://arxiv.org/html/2510.21464v4 ; https://www.contrast-connect.com/blog-post/radiology-report-findings-vs-impression-whats-the-difference
**Date:** 2026-05-19 / 2025-12-31
**Excerpt:** "We extract structured diagnostic labels from the 'Impression' section of radiology reports using the CheXpert labeler, a rule-based natural language processing system that identifies 14 common radiological observations." / "The Findings section of a radiology report contains detailed observations of all structures visible in the imaging study, while the Impression section provides a concise summary and interpretation of clinically significant findings."
**Confidence:** Medium-High (CheXpert-Impression practice well attested; note knee MRI reports here are heterogeneous across 16 sites)

### 4.3 NegEx multilingual extensions (rule-based negation)

**Claim:** NegEx lexicon extended to French, German, Swedish with performance comparable to English: Swedish recall 82%/precision 75%; French recall 85%/precision 89%.
**Source:** Chapman et al., "Extending the NegEx Lexicon for Multiple Languages" (PMC3923890, MEDINFO 2013)
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3923890/
**Date:** 2013
**Excerpt:** "NegEx has been recently ported and evaluated on clinical texts in Swedish [6] and French [7] and has shown good performance (recall 82%; precision 75%) for Swedish assessment sections of the Stockholm EPR corpus and better performance (recall 85%; precision 89%) for French cardiology notes... Although not evaluated yet, the NegEx lexicon has also been translated to German."
**Confidence:** High

**Claim:** Spanish NegEx adaptation for radiology (SpRadNeg): precision 0.87, recall 0.49 on ultrasound reports (recall is the weak point of naive trigger porting; ML-boosted annotation reached 0.91 P / 0.89 R).
**Source:** Stricker, Iacobacci, Cotik, "Negated Findings Detection in Radiology Reports in Spanish: an Adaptation of NegEx to Spanish" (IJCAI WS)
**URL:** https://staff.dc.uba.ar/vcotik/docs/papers/NegatedFindingsDetectionIJCAISWS.pdf
**Date:** 2015/2016
**Excerpt:** "In this article we present SpRadNeg, which is an adaptation of NegEx to the Spanish language... We have tested SpRadNeg with radiology reports, obtaining a precision of 0.87 and a recall of 0.49. We also propose a method to automatize text annotation based on Machine Learning techniques with 0.91 precision and 0.89 recall."
**Confidence:** High

### 4.4 German CheXpert (rule labeler port with negation + uncertainty phrase files)

**Claim:** German CheXpert labeler (Wollek et al. 2023): two negation types in German ("nicht/kein" NegEx-resolvable + implicit-negation terms like "Herz normal groß"); architecture extended with per-observation positive/negative/uncertain phrase files; mention F1 up to 0.995; BERT fine-tune on rule labels adds +3–10 F1. CheXpert has also been ported to Brazilian Portuguese and Vietnamese.
**Source:** Thieme/Rofo, "German CheXpert Chest X-ray Radiology Report Labeler" + Emergent Mind CheXpert topic
**URLs:** https://www.thieme-connect.com/products/ejournals/html/10.1055/a-2234-8268 ; https://www.emergentmind.com/topics/chexpert-labeler
**Date:** 2024-01-31
**Excerpt:** "In German radiology reports, two distinct types of negations were identified: expressions that contain phrases like 'nicht' or 'kein' ('no', 'not') and are observation-independent, which can be resolved by the German NegEx algorithm, and medical terms that lack any negations but convey the lack of an observation, for example, 'Herz normal groß' ('regular heart size'). As the CheXpert architecture addresses only negated observations, we extended the architecture by using multiple phrase files (positive, negative, uncertain) per observation." / "this labeler has been adapted and ported to process reports in other languages, such as Brazilian [18] and Vietnamese [19]." / Emergent Mind: "achieving mention F1 up to 0.995... Deep learning extensions (BERT-based) trained with weak supervision from rule-based labels and then fine-tuned on limited manual annotations improve F1 by 3–10 points over purely rule-based systems."
**Confidence:** High

### 4.5 Cross-lingual CheXbert (Spanish)

**Claim:** Multilingual CheXbert (M-BERT fine-tuned on translated/English CheXpert labels, tested on Spanish PadChest): zero-shot cross-lingual report labeling is feasible; PadChest (Spanish) provides 20,281 physician-annotated reports; binary mapping discards "uncertain" scores.
**Source:** Stanford CS224n report, "Multilingual CheXbert: Radiology Report Labeling in Spanish"
**URL:** https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report075.pdf
**Date:** 2021
**Excerpt:** "As our Spanish-language dataset, we use PadChest, which has 84,170 radiology reports. Of those reports, 63,889 were automatically-labeled using a supervised method based on a recurrent neural network with attention mechanisms, and 20,281 were manually annotated by trained physicians."
**Confidence:** Medium (student report; directional evidence)

### 4.6 Hedging/uncertainty phrases

**Claim:** Hedging is the hardest category across methods and languages: CheXpert uncertainty micro-F1 0.848 (vs 0.952 negation); LLMs poorly calibrated on OBS-U ("cannot exclude," "possibly representing", ECE > 0.40); German uncertainty triggers like "unwahrscheinlich" ("unlikely") and pseudo-negations "kann ausgeschlossen werden" ("can be excluded") require explicit phrase files.
**Source:** Emergent Mind CheXpert topic; arXiv:2603.00924; German CheXpert paper
**URLs:** https://www.emergentmind.com/topics/uncertainty-aware-chexpert-style-labels ; https://arxiv.org/html/2603.00924v2 ; https://www.thieme-connect.com/products/ejournals/html/10.1055/a-2234-8268
**Date:** 2023–2026
**Excerpt (German CheXpert):** "the labeling algorithm identifies negation phrases such as 'kann ausgeschlossen werden' ('can be excluded'), and uncertainty phrases, such as 'unwahrscheinlich' ('unlikely'), based on a set of rules and marks them as pre- or post-negation/uncertainty phrases."
**Confidence:** High

---

## 5. BENCHMARKS / ACCURACY NUMBERS FOR THIS COMPETITION'S ACTUAL SETUP (Kaggle evidence)

**Claim:** On the 58 gold studies, LLM-derived report labels beat regex/lexicon labels: macro AUC 0.8780 vs 0.8136. 25.4% of all report-label cells are "not addressed"; per-finding silence is informative differently (synovitis silent→34% positive; Baker's silent→3% positive). Targeted imputation (fill silent synovitis from effusion) lifted synovitis AUC 0.678→0.790 and macro 0.8780→0.8873; blanket learned imputation was WORSE (0.8805).
**Source:** Kaggle discussion "'Not addressed' is a label too — what we learned reading 4,407 knee reports with an LLM" (stevenleehans)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932
**Date:** ~2026-08-09
**Excerpt:** "Measured against those 58 gold studies: regex / lexicon extraction 0.8136; LLM reading the same reports 0.8780... 25.4% of all cells came back at exactly 0.5... Synovitis [gold AUC] 0.678 [not addressed] 83.7%; Baker's 0.946 / 48.2%; Fracture 0.793 / 42.9%; ACL 0.993 / 8.3%; Medial Meniscus 0.954 / 5.5%... When a radiologist does not mention a Baker's cyst, there almost certainly is not one — 3% versus 44%. The silence is the label... silence about synovitis still leaves a 34% chance it is present." / "58 studies is a small ruler. On it, differences below roughly 0.02 macro are not measurable... Gold prevalence is the annotator's sampling, not disease prevalence. Every gold study has at least one positive finding; mean 4.14 per study."
**Confidence:** High (verbatim; community measurement, in-sample caveats noted by author)

**Claim:** Pure rule-based multilingual labeler (no LLM, no GPU): ablation on 58 gold — naive keyword 0.638 → +sentence-scope negation 0.667 → +OA-consequence vocabulary 0.727 macro AUC. OA is "almost never written as the word osteoarthritis" — mining osteophytes/joint-space narrowing/chondral loss/gonarthrose + "tricompartmental" rule took Lateral OA 0.47→0.83, Medial OA 0.59→0.75. Negation buys precision (fracture 0.53→0.80; Baker's 0.42→0.62). Synovitis (0.61) and Effusion (0.63) are hardest even for rules.
**Source:** Kaggle discussion "Labeling the 4,349 report-only studies without an LLM, and what it scores on the 58 gold" (Busya PRIME)
**URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734095
**Date:** 2026-08-10
**Excerpt:** "The ablation, macro AUC over the 12 on the 58 gold: naive keyword presence 0.638, add sentence-scope negation 0.667, add the OA consequence vocabulary 0.727... Osteoarthritis is almost never written as the word osteoarthritis... That vocabulary takes Lateral OA to 0.83 and Medial OA to 0.75... Negation mostly buys precision (fracture precision 0.53 to 0.80, Baker's 0.42 to 0.62) rather than ranking."
**Confidence:** High

**Claim:** Report-vs-image label agreement is only ~82.5% (strict textual reading vs provided labels), with positive recall 80% — i.e., report-derived labels are SILVER labels; the image-annotated subset and the hidden test set use stricter, multi-reader image-based thresholds. Graded/soft targets reportedly beat binary ones by +0.056 macro AUC (FHZ982 thread title; community claim).
**Source:** Kaggle discussions 733826 and 734105
**URLs:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826 ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105
**Date:** 2026-08
**Excerpt (733826):** "Across 240 decisions, the report-only labels matched the provided labels in 198 cases: Overall agreement: 82.5%" / (734105 title): "Graded targets beat binary ones by +0.056 macro AUC — and we can say which half comes from where"
**Confidence:** High for 82.5% figure; Medium for +0.056 (title-only evidence)

**Claim (prior literature):** NLP label imperfections are offset by training-set size — imperfect report labels can still train strong image classifiers (Shih et al. / Radiology:AI challenges review); weakly supervised CT classification with NLP labels achieved 91–99% manual-validation label accuracy (Dufumier et al., Radiol:AI 2022).
**Source:** "Challenges Related to Artificial Intelligence Research in Medical Imaging..." (Radiol:AI 2019, PMC8017381); "Classification of Multiple Diseases on Body CT Scans Using Weakly Supervised Deep Learning" (PMC8823458)
**URLs:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8017381/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8823458/
**Date:** 2019 / 2022
**Excerpt:** "Recent reports show promising results in these methodologies to label image datasets created for classification tasks and seem to indicate that imperfections of the natural language processing system can be counterbalanced by increasing the number of images in the training set." / "Manual validation of the extracted labels confirmed 91%–99% accuracy across the 15 different labels."
**Confidence:** High

---

## 6. PRACTICAL RECIPE (Kaggle-feasible, offline-capable) — RECOMMENDATION

Synthesis of the above evidence into a concrete pipeline for turning ~10–12-language reports into 12-class training labels:

**Step 0 — Language ID.** Run fastText `lid.176` (or langdetect) on each report; per-institution priors are strong (16 sites ≈ 1 language each). Expect EN/ES/NL dominant; FR/DE/PT/IT/TR tail; rare BG/EL.

**Step 1 — Section split.** Per-language regex for Impression/Conclusión/Beurteilung/Conclusie headers; keep both Findings + Impression but weight Impression mentions higher (CheXpert-on-Impression practice, §4.2). For headerless reports, use whole text.

**Step 2 — LLM-first label extraction (best measured quality).** Per host ruling (§1.6), commercial APIs are allowed during development; for zero-cost/reproducible/offline: run an open-weight multilingual instruct LLM in a Kaggle GPU notebook (Qwen2.5-14B/32B-Instruct or Qwen3-14B/32B, or Gemma-3-27B for widest language coverage incl. Bulgarian/Greek; GPT-OSS-20B is a strong single-GPU option with proven radiology extraction macro-F1 0.79, §3.5). Prompt for a strict JSON schema with one field per finding and a 5-way value {present, absent, uncertain, not_addressed, laterality-ambiguous} + verbatim evidence span + English gloss. Few-shot with 2–3 examples per language; include the official label thresholds (§1.3) in the system prompt (anatomy/guideline-aware prompting gave +0.08 macro-F1, §3.5). Expect ~0.88 macro AUC vs gold (Kaggle-measured, §5).

**Step 3 — Explicit "not addressed" channel (critical).** Map not_addressed → soft label 0.5, never coerce to negative; 25.4% of cells are silent and silence semantics differ per finding (§5: Baker's silence ≈ negative; synovitis silence ≈ uninformative). Fill only per-finding-justified cells from co-occurrence (effusion→synovitis worked: +0.11 AUC on that column); do NOT blanket-impute.

**Step 4 — Rule-based fallback/ensemble + audit.** Multilingual CheXpert-style labeler (per-language phrase files: positive/negative/uncertain per finding; NegEx-style triggers per language — "no/sin/kein/geen/pas de/yok/няма/δεν"; hedges — "possible/posible/möglicherweise/eventuell/olası") gives ~0.73 macro alone, adds precision on fracture/Baker's, and provides deterministic audit + a second opinion for ensemble disagreement flagging (§5). Mine OA consequences (osteophytes, JSN, chondral loss, gonarthrose/Gonarthrose, "tricompartmental") — biggest single lever for rules.

**Step 5 — Translate-to-English as a THIRD voter only.** NLLB-200-distilled-1.3B (offline, CC-BY-NC) per sentence; its model card warns against medical domain and >512 tokens (§3.4) and GPT-4o-level translation still loses ~16–21% factual fidelity (§3.4), so use translated text only for cross-checking/ensemble, or fine-tune XLM-R-base/large classifier on the LLM silver labels (translate-train-all style, §3.1) as the scalable student.

**Step 6 — Distill to a multilingual student + calibrate on the 58 gold.** Fine-tune XLM-R-large (or mE5/BGE-M3 embeddings + logistic heads) on the LLM soft labels; validate exclusively against the 58 image-derived gold studies (the ONLY labels matching test-time truth); treat <0.02 macro deltas as noise (§5). Output per-finding probabilities (graded/soft targets beat hard binary by ~+0.056 macro, community-reported).

**Step 7 — Image training uses soft silver labels with per-finding confidence weights** (weight = labeller confidence; downweight not_addressed-filled cells). Report text is NOT available at test, so the text pipeline is purely a teacher. Label noise is tolerable — literature shows NLP-label imperfections are counterbalanced by dataset size (§5), and here you have 4,349 silver + 58 gold studies.

**Key pitfalls (evidence-backed):** (a) coercing "not addressed" to 0 — destroys synovitis (84% silent) and misreads Baker's (silence = negative); (b) trusting report labels over image labels where both exist — host says image wins (§1.5); (c) naive keyword matching without negation/uncertainty per language (−0.03 to −0.09 macro, §5; SpRadNeg recall 0.49 shows naive porting fails, §4.3); (d) treating OA as a keyword — mine consequences (§5); (e) bilateral studies under one UID (host-adjusted, §1.5); (f) low-resource-language LLM extraction without verification (Hebrew F1 0.40 cautionary, §3.5); (g) test-time distribution shift in prevalence (Data page notice, §1.5).

---

## SEARCH LOG (25+ independent searches, 2026-08-10)
1. RSNA Knee Abnormality Detection challenge 2026
2. RSNA knee abnormality detection challenge annotation protocol rsna.org
3. RSNA 2026 knee volunteer annotators musculoskeletal radiologists
4. RSNA knee Po-Hao Chen Naveen Subhas
5. Kaggle rsna-knee discussion languages reports
6. RSNA knee 12 abnormalities list labels meniscus ACL
7. "rsna-knee-abnormality-detection" PCL/LCL/Baker twelve binary
8. RSNA knee competition evaluation labels effusion/fracture/BME
9. arxiv 2026 RSNA knee dataset multilingual Chen Subhas
10. RSNA knee discussion languages identified EN/ES/PT/DE
11. NegEx multilingual Spanish German French radiology extension
12. CheXpert labeler extension other languages negation uncertainty
13. Kaggle RSNA knee fastText langdetect language identification
14. RSNA knee how many studies labeled training subset
15. multilingual-e5-large model card 100 languages
16. XLM-RoBERTa 100 languages cross-lingual classification
17. NLLB 200 languages distilled offline Hugging Face
18. translate-to-English multilingual clinical text radiology study
19. Kaggle "4,407 studies"/"58 labels" maximo thread
20. Kaggle "Not addressed" LLM reports thread
21. Kaggle "Weak labels for all 12 findings" thread
22. LLM structured extraction radiology JSON F1 GPT-4 Llama
23. findings vs impression section NLP label extraction accuracy
24. Kaggle RSNA knee EDA language distribution (→ Rozen, yuki16)
25. CheXpert labeler F1 0.969/0.952/0.848 Irvin 2019
26. arxiv RSNA knee dataset paper 2026 (negative result)
27. multilingual clinical NLP cross-lingual XLM-R medical classification
28. BGE-M3 model card 100 languages MTEB
29. MMedC/MMed-Llama multilingual medical benchmark (→ arXiv:2402.13963)
30. Qwen2.5/Qwen3 multilingual model card languages
31. CheXbert impression section labeler (Smit 2020)
32. impression-only label extraction MIMIC CheXpert practice
(plus direct browser reads of Kaggle data page, discussion threads 733343/733652/733826/733932/733965/734095, discussion listings pp.1–2)
