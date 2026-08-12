# Dimension 05 — Annotated Bibliography: Multimodal Learning from Medical Images + Radiology Reports
**Target competition:** RSNA Knee Abnormality Detection (Kaggle, 2026) — knee MRI + training labels derived from real-world radiology reports (complex, unstructured, multilingual).
**Compiled:** 2026-08-10 by research sub-agent (dimension 05).
**Method note:** 22 independent web searches executed (PubMed/PMC/arXiv/ACL Anthology/NeurIPS/PMLR prioritized). Every entry uses the required claim format: Claim / Source / URL / Date / Excerpt (verbatim) / Confidence. Excerpts are verbatim from the cited source.

---

## SECTION 1 — Automated Label Extraction from Radiology Reports

### 1.1 CheXpert labeler (Irvin et al. 2019)
- **Claim:** The CheXpert labeler is a three-stage rule-based NLP pipeline (mention extraction, mention classification into positive/negative/uncertain, mention aggregation) that extracts 14 observation labels from free-text chest radiology reports and provides "silver-standard" labels for large-scale image model training; it reaches high mention/negation F1 but markedly lower uncertainty F1.
- **Source:** Irvin J, Rajpurkar P, Ko M, et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI 2019 (arXiv:1901.07031); performance figures cross-checked via the CheXpert Labeler Overview (EmergentMind).
- **URL:** https://arxiv.org/abs/1901.07031 ; https://www.emergentmind.com/topics/chexpert-labeler
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Validation on manually annotated datasets has demonstrated the CheXpert labeler's effectiveness: |Task|Micro-F1|Macro-F1| |Mention|0.969|0.948| |Negation|0.952|0.899| |Uncertainty|0.848|0.770|" and "Uncertain labels are assigned explicitly instead of being collapsed into positive or negative, thus capturing the spectrum of radiologist interpretations."
- **Confidence:** High (primary paper well established; table reproduced from secondary overview).
- **Why it matters:** This is the canonical template for turning free-text reports into training labels — directly analogous to what RSNA 2026 does for knee MRI. Its explicit *uncertain* class and the U-Ignore / U-Zeros / U-Ones / U-MultiClass / U-SelfTrained policies are the standard playbook for handling ambiguous report labels.
- **Key methods:** radiologist-curated keyword/regex mention lists; ordered pre-negation-uncertainty → negation → post-negation-uncertainty rules; prioritized report-level aggregation.
- **Key results:** Mention F1 0.969, negation F1 0.952, uncertainty F1 0.848 (micro); downstream pathology-dependent gains from uncertainty policies (e.g., U-Ones helps Atelectasis, U-MultiClass helps Cardiomegaly).
- **Transferable trick:** Keep a 3- or 4-state label scheme (positive/negative/uncertain/unmentioned) when re-deriving or auditing the competition's report-derived labels; treat "uncertain" as a distinct class or soft label rather than forcing binary, and evaluate per-pathology which uncertainty policy maximizes validation AUC.

### 1.2 NegBio (Peng et al. 2018)
- **Claim:** NegBio detects negation and uncertainty scopes in radiology reports using patterns defined on universal-dependency graphs instead of surface regular expressions, outperforming NegEx by ~9.5% precision and ~5.1% F1 on average.
- **Source:** Peng Y, Wang X, Lu L, et al. "NegBio: a high-performance tool for negation and uncertainty detection in radiology reports." AMIA Jt Summits Transl Sci Proc. 2018:188–196 (PMC5961822).
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC5961822/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Evaluation on these datasets demonstrates that NegBio is highly accurate for detecting negative and uncertain findings and compares favorably to a widely-used state-of-the-art system NegEx (an average of 9.5% improvement in precision and 5.1% in F1–score)." Also: "Overall, NegBio achieved a higher precision of 89.8%, recall of 85.0%, and Fl-score of 87.3% on OpenI."
- **Confidence:** High.
- **Why it matters:** Negation/uncertainty scoping is the single largest error source in report-derived labels. NegBio's dependency-graph approach generalizes better than windowed regex when reports use long noun phrases — a hallmark of non-English and telegraphic report styles likely present in the multilingual knee dataset.
- **Key methods:** MetaMap UMLS concept recognition + Semgrex dependency-graph patterns + subgraph matching; scope determined by syntactic context, not word distance.
- **Key results:** OpenI precision 89.8% / F1 87.3% for positive-finding extraction; on BioScope negation detection, +25.5% precision and +13.6% F1 over NegEx.
- **Transferable trick:** If building/validating a labeler for non-English knee reports, define negation/uncertainty triggers over dependency parses (e.g., spaCy/Stanza) rather than fixed word windows; watch for the known failure modes NegBio documents (long verb-less noun phrases, double negation).

### 1.3 CheXbert (Smit et al. 2020)
- **Claim:** A biomedically-pretrained BERT first trained on rule-based labeler outputs, then fine-tuned on a small expert-annotated set augmented with back-translation, beats the CheXpert rule labeler and nearly matches radiologist-level labeling.
- **Source:** Smit A, Jain S, Rajpurkar P, Pareek A, Ng A, Lungren M. "Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT." EMNLP 2020, pp. 1500–1519.
- **URL:** https://aclanthology.org/2020.emnlp-main.117/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Fourth, we find that CheXbert outperforms the previous best labeler, CheXpert (which was rules-based), with an improvement of 0.055 (95%CI 0.039,0.070) on the F1 metric ... Fifth, we find that CheXbert is 0.007 F1 points from the radiologist performance benchmark, suggesting that the gap to ceiling performance is narrow."
- **Confidence:** High.
- **Why it matters:** This is the practical recipe for the competition's central problem: combine cheap, noisy rule/LLM labels at scale with a tiny set of expert labels. It quantifies exactly how much expert annotation buys you (you need surprisingly little if you pretrain on weak labels).
- **Key methods:** two-stage training (weak-label pretraining → expert fine-tuning); back-translation augmentation of the expert set; BioClinicalBERT initialization.
- **Key results:** +0.055 F1 over CheXpert; within 0.007 F1 of radiologist benchmark; BERT pretrained only on expert labels underperforms the two-stage recipe.
- **Transferable trick:** For knee reports: (1) write a quick rule/keyword labeler or use an LLM to label all training reports; (2) pretrain a domain BERT on those weak labels; (3) fine-tune on any small hand-checked subset (even a few hundred reports); (4) use back-translation to augment the expert subset — especially valuable across the dataset's multiple languages.

### 1.4 CheXpert++ (McDermott et al. 2020)
- **Claim:** A BERT classifier trained to imitate CheXpert outputs achieves 99.81% parity while being differentiable, probabilistic, and ~1.8x faster; clinicians preferred its labels in the majority of disagreements with the rule-based original.
- **Source:** McDermott MBA, Hsu TMH, Weng WH, et al. "CheXpert++: Approximating the CheXpert labeler for Speed, Differentiability, and Probabilistic Output." MLHC 2020; details via EmergentMind CheXpert Labeler Overview.
- **URL:** https://www.emergentmind.com/topics/chexpert-labeler
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "99.81% overall parity, with per-task parity >99.7%, and labeling speed improvement of 1.8× (1.53 vs 2.75 hours) over the original ... In expert-blinded comparisons, CheXpert++ labels were preferred by clinicians in 59% of disagreements, versus 28% preferring CheXpert."
- **Confidence:** Medium-High (secondary source summarizing the primary paper).
- **Why it matters:** Shows a neural labeler can be a drop-in replacement for a fragile rule pipeline while emitting *probabilities* — soft labels you can feed straight into image-model training, which is exactly what noisy-label learning needs.
- **Key methods:** 14 independent softmax heads over clinical BERT; trained on 602,855 MIMIC-CXR sentences with CheXpert silver labels; entropy-based active learning round.
- **Key results:** 99.81% parity; ~+8% accuracy after one active-learning relabeling round.
- **Transferable trick:** Train your text labeler to output calibrated per-class probabilities and train the image model on soft targets (e.g., via BCE on probabilities); use the labeler's entropy to select which reports to hand-verify.

### 1.5 RadBERT (Yan et al. 2022)
- **Claim:** BERT-family models continued-pretrained on millions of VA radiology reports outperform general/biomedical/clinical BERT baselines on radiology NLP tasks, with the largest gains in low-data regimes.
- **Source:** Yan A, McAuley J, Lu X, et al. "RadBERT: Adapting Transformer-based Language Models to Radiology." Radiology: Artificial Intelligence 2022;4(4):e210258 (PMC9344353).
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9344353/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "For abnormal sentence classification, all models performed well (accuracies above 97.5 and F1 scores above 95.0). RadBERT variants achieved significantly higher scores than corresponding baselines when given only 10% or less of 12 458 annotated training sentences. For report coding, all variants outperformed baselines significantly for all five coding systems."
- **Confidence:** High.
- **Why it matters:** Domain-adaptive pretraining on *radiology* text (not just clinical text) is what moves the needle when labeled reports are scarce — directly applicable to a knee-specific, multilingual corpus.
- **Key methods:** MLM continued pretraining on 2.16M/4.42M VA reports over four initializations; fine-tuning for abnormal-sentence classification, report coding, summarization.
- **Key results:** RadBERT-RoBERTa-4m with 5% training data matched BioBERT with 100% training data; best summarization ROUGE-1 16.18 vs 15.27 baseline.
- **Transferable trick:** Before fine-tuning any text encoder for knee-report label extraction, run cheap continued MLM pretraining on the *unlabeled competition reports themselves* (in their native languages) — a free, competition-legal domain-adaptation step.

### 1.6 MedCAT (Kraljevic et al. 2021)
- **Claim:** MedCAT is an open-source *unsupervised* NER+linking toolkit that beat supervised baselines for disease detection (F1 0.848 vs 0.691) and supports self-supervised disambiguation of biomedical concepts in EHR text.
- **Source:** Kraljevic Z, Searle T, Shek A, et al. "Multi-domain clinical natural language processing with MedCAT: the Medical Concept Annotation Toolkit." Artificial Intelligence in Medicine 2021;117:102083 (arXiv:1912.10166).
- **URL:** https://arxiv.org/abs/1912.10166
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "In case of NER+L, the comparison with existing tools shows that MedCAT improves the previous best with only unsupervised learning (F1=0.848 vs 0.691 for disease detection; F1=0.710 vs. 0.222 for general concept detection)."
- **Confidence:** High.
- **Why it matters:** Provides an ontology-anchored alternative to regex/LLM labeling: map report spans to UMLS/SNOMED concepts, then derive abnormality labels from concept IDs — robust to surface-form and language variation if a multilingual terminology is used.
- **Key methods:** unsupervised concept detection + VITERBI-style linking + embedding-based disambiguation; optional supervised/active learning via MedCATtrainer.
- **Key results:** See excerpt; validated on MIMIC-III and MedMentions.
- **Transferable trick:** Use concept IDs (e.g., "tear of meniscus", "ACL rupture") rather than surface strings as labeling targets for the multilingual reports; concept-based aggregation collapses synonyms and translations into one label space.

### 1.7 RadGraph (Jain et al. 2021)
- **Claim:** RadGraph defines a schema of clinical entities and relations in radiology reports and provides a DYGIE++-based benchmark model reaching micro-F1 0.94 (MIMIC-CXR) for entity recognition and 0.82 for relation extraction, close to but below human benchmarks.
- **Source:** Jain S, Agrawal A, Saporta A, et al. "RadGraph: Extracting Clinical Entities and Relations from Radiology Reports." CHIL 2021 (arXiv:2106.14463).
- **URL:** https://ar5iv.labs.arxiv.org/html/2106.14463
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "RadGraph Benchmark achieves a micro F1 of 0.94/0.91 on named entity recognition and a micro F1 of 0.82/0.73 on relation extraction. The human benchmark achieves a micro F1 of 0.99/0.93 on named entity recognition and a micro F1 of 0.95/0.75 on relation extraction."
- **Confidence:** High.
- **Why it matters:** Structured (anatomy, observation, certainty) tuples are a richer supervision signal than flat labels; graph-level report representations have been shown to improve image-text contrastive learning (IGCL outperforms ConVIRT/GLoRIA in the CheXzero lineage).
- **Key methods:** radiologist-designed entity/relation schema; DYGIE++/PURE information-extraction models initialized from PubMedBERT; RadGraph F1 metric later used to reward factuality in report generation.
- **Key results:** See excerpt; entity F1 highest for Anatomy and "Observation: Definitely Absent", lowest for "Observation: Uncertain".
- **Transferable trick:** Parse knee reports into (anatomy, finding, certainty) triples — e.g., (medial meniscus, tear, definitely present) — and use them both as multi-task auxiliary targets and as a clean way to resolve contradictory/uncertain mentions before image-label assignment.

### 1.8 Commercial vs open-source LLMs for report labeling (Dorfner et al. 2024)
- **Claim:** GPT-4 sets the zero-shot report-labeling benchmark (micro-F1 ≈ 0.975–0.984), but open-source 70B-class models with few-shot prompting reach parity, enabling privacy-preserving, free, at-scale label extraction.
- **Source:** Dorfner FJ, Jürgensen L, Donle L, et al. "Is Open-Source There Yet? A Comparative Study on Commercial and Open-Source LLMs in Their Ability to Label Chest X-Ray Reports." arXiv:2402.12298 (2024); published in J Digit Imaging 2024.
- **URL:** https://arxiv.org/abs/2402.12298
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "On the ImaGenome dataset, the best performing open-source model was Llama2-70B with micro F1-scores of 0.972 and 0.970 for zero- and few-shot prompts, respectively. GPT-4 achieved micro F1-scores of 0.975 and 0.984, respectively. On the institutional dataset, the best performing open-source model was QWEN1.5-72B with micro F1-scores of 0.952 and 0.965 for zero- and few-shot prompting, respectively. GPT-4 achieved micro F1-scores of 0.975 and 0.973, respectively."
- **Confidence:** High.
- **Why it matters:** If the competition's report labels look noisy or the reports are in languages where rule labelers don't exist, an offline open-source LLM can regenerate or audit labels at micro-F1 ≈ 0.95+ without any PHI/API concerns.
- **Key methods:** zero-/few-shot structured prompting for presence of multiple findings; comparison across 540 MGH + 500 ImaGenome reports against CheXbert/CheXpert.
- **Key results:** See excerpt; few-shot prompting closes the open-source vs GPT-4 gap.
- **Transferable trick:** Ensemble 2–3 open-source LLMs (e.g., Llama-3-70B, Qwen-72B, Mixtral) with few-shot prompts in each report language and take majority vote per finding — a cheap proxy "human-level" labeler for relabeling or confidence-scoring training samples.

### 1.9 Open-source LLM label extraction at scale; classifier resilience to label noise (2025)
- **Claim:** Open-source LLMs (Llama-3, Phi-3, Zephyr) beat the CheXpert rule labeler against human annotations (e.g., 95% vs 51% sensitivity for rib fracture), and — critically — image classifiers trained on noisier labels retained most of their performance when evaluated against clean labels.
- **Source:** "Role of Model Size and Prompting Strategies in Extracting Labels from Free-Text Radiology Reports with Open-Source Large Language Models." Journal of Digital Imaging 2025 (PMC12920854).
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12920854/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "LLM-based labeling outperformed the CheXpert labeler, with the best LLM achieving 95% sensitivity for fracture detection versus CheXpert's 51%." ... "Image classifiers showed resilience to labeling noise when tested externally." ... "a classifier trained on Llama-3 with chain-of-thought labels achieved AUCs of 0.96 and 0.84 for pneumothorax and fracture detection respectively when evaluated against human annotations, compared to 0.91 and 0.73 when evaluated on CheXpert labels."
- **Confidence:** High.
- **Why it matters:** Two transferable facts: (1) LLM labels are substantially better than rule labels for nuanced findings; (2) the *evaluation* label schema matters more than the training-label schema — models are fairly robust to training-label noise.
- **Key methods:** 227,827 MIMIC-CXR reports labeled by three open LLMs; 2,000-report human-annotated validation; downstream pneumothorax/rib-fracture classifiers tested on CANDID-PTX.
- **Key results:** See excerpt; CoT prompting hurt small models (Phi-3 F1 0.91 → 0.81).
- **Transferable trick:** Don't over-engineer label cleaning for training; instead invest in a small, high-quality (possibly self-annotated) *validation* split, and expect image-model ranking on noisy-label training to transfer. Skip chain-of-thought prompting for smaller local LLMs.

### 1.10 LLM pseudo-label distillation into small discriminative labelers (DeBERTa-RAD, 2025)
- **Claim:** High-fidelity pseudo-labels generated by an LLM can be distilled into a lightweight DeBERTa classifier that extracts Present/Absent/Uncertain labels for 13 findings from chest reports, combining LLM quality with deployable speed.
- **Source:** "High-Fidelity Pseudo-label Generation by Large Language Models for Training Robust Radiology Report Classifiers." arXiv:2505.01693 (2025).
- **URL:** https://arxiv.org/html/2505.01693v1
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Our methodology proceeds in two distinct stages: first, leveraging a powerful large language model (LLM) to generate high-quality pseudo-labels for a large corpus of reports, and second, training the DeBERTa-RAD model using these pseudo-labels via a knowledge distillation strategy."
- **Confidence:** High (for method description; performance claims not independently verified).
- **Why it matters:** In Kaggle you cannot call an LLM at inference for huge unlabeled corpora cheaply — distillation gives near-LLM label quality at BERT speed and reproducibility.
- **Key methods:** LLM pseudo-labeling → knowledge-distilled multi-label DeBERTa.
- **Key results:** Noted in abstract-level detail only here.
- **Transferable trick:** Use GPT-4-class or 70B local LLM to label a few thousand knee reports per language, distill into a small multilingual student (e.g., XLM-R/DeBERTa), and use the student to label/confidence-score the full training set.

---

## SECTION 2 — Vision-Language Pretraining in Radiology (weights availability flagged)

### 2.1 ConVIRT (Zhang et al. 2020)
- **Claim:** Bidirectional image–text contrastive learning on naturally paired radiographs and report sentences yields visual representations far superior to ImageNet or image-only self-supervision, especially in low-label regimes.
- **Source:** Zhang Y, Jiang H, Miura Y, Manning CD, Langlotz CP. "Contrastive Learning of Medical Visual Representations from Paired Images and Text." MLHC 2022 / arXiv:2010.00747 (2020).
- **URL:** https://ar5iv.labs.arxiv.org/html/2010.00747
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "We present Contrastive VIsual Representation Learning from Text (ConVIRT), a framework for learning visual representations by exploiting the naturally occurring pairing of images and textual data." Results table: "ImageNet 82.8 / SimCLR 86.3 / MoCo v2 86.6 / ConVIRT 90.7 (RSNA Linear, 1% data AUC); CheXpert 75.7 / 77.4 / 81.3 / 85.9; Image-Image Prec@10 14.4 / 17.6 / 20.6 / 42.9."
- **Confidence:** High.
- **Why it matters:** The foundational proof that *reports are supervision*. With ~1% labels, ConVIRT beat ImageNet by ~8 AUC points on an RSNA task — exactly the regime of a new competition with imperfect labels.
- **Key methods:** ResNet-50 + BERT encoders; random sentence sampling from reports; bidirectional InfoNCE with non-linear projections.
- **Key results:** See table; also outperformed SimCLR/MoCo on all downstream tasks and produced better-localized saliency.
- **Transferable trick:** Pretrain the knee MRI encoder with image–report contrastive loss on the competition's own training pairs (reports are given), using sentence-level sampling and report text in any language; even a modest compute budget pretraining beat generic ImageNet init in low-label settings. **Weights:** no official public weights; reproducible PyTorch reimplementations exist (e.g., edreisMD/ConVIRT-pytorch).

### 2.2 GLoRIA (Huang et al. 2021) — weights public
- **Claim:** Adding global–local (image-region ↔ word) attention alignment to contrastive image–report learning gives label-efficient recognition and zero-shot segmentation ability.
- **Source:** Huang SC, Shen L, Lungren MP, Yeung S. "GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-Efficient Medical Image Recognition." ICCV 2021, pp. 3942–3951.
- **URL:** https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_ICCV_2021_paper.html ; code+weights: https://github.com/marshuang80/gloria
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "GLoRIA (Global-Local Representations for Images using Attenion) is a multimodal representation learning framework for label-efficient medical image recognition. Our results demonstrate high-performance and label-efficiency for image-text retrieval, classification (finetuning and zeros-shot settings), and segmentation on different medical imaging datasets."
- **Confidence:** High.
- **Why it matters:** Local alignment is what you want for focal knee lesions (meniscal/ACL tears occupy few slices/regions): word-level attention gives free localization cues that a global-only model lacks.
- **Key methods:** attention-weighted sub-region/word similarity + global contrastive loss; label-efficient fine-tuning at 1%/10%/100% data.
- **Key results:** SOTA label-efficient classification/retrieval/segmentation on CheXpert, RSNA Pneumonia, SIIM pneumothorax.
- **Transferable trick:** Use local cross-attention between report tokens and MRI slice regions as an auxiliary objective or as a weak localizer for the abnormality. **Weights:** official pretrained ResNet-50 image encoder + BioClinicalBERT text encoder released on GitHub — usable as a Kaggle offline resource (verify license/competition rules).

### 2.3 BioViL / CXR-BERT (Boecking et al. 2022) — weights public
- **Claim:** Better *text-side* modeling (radiology-specific vocabulary, semantic/discourse-aware pretraining in CXR-BERT) substantially improves joint vision–language pretraining; BioViL sets SOTA on classification, phrase grounding, and even segmentation using only a global alignment objective.
- **Source:** Boecking B, Usuyama N, Bannur S, et al. "Making the Most of Text Semantics to Improve Biomedical Vision–Language Processing." ECCV 2022 (arXiv:2204.09817).
- **URL:** https://arxiv.org/abs/2204.09817 ; weights: HuggingFace `microsoft/BiomedVLP-CXR-BERT-specialized`, `microsoft/BiomedVLP-BioViL-B` (via hi-ml-multimodal)
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "We release a language model that achieves state-of-the-art results in radiology natural language inference through its improved vocabulary and novel language pretraining objective leveraging semantics and discourse characteristics in radiology reports. Further, we propose a self-supervised joint vision--language approach with a focus on better text modelling. It establishes new state of the art results on a wide range of publicly available benchmarks, in part by leveraging our new domain-specific language model."
- **Confidence:** High.
- **Why it matters:** Demonstrates the text encoder is not a commodity: in-domain vocabulary and report-discourse modeling are where the gains are. For multilingual knee reports, the analogous move is a report-domain-adapted multilingual text encoder.
- **Key methods:** CXR-BERT (custom radiology vocab; MLM + radiology NLI objective over report-section structure); joint image–text contrastive pretraining on MIMIC-CXR; MS-CXR phrase-grounding benchmark release.
- **Key results:** SOTA on RadNLI, phrase grounding, zero/few-shot CXR classification; beats prior methods on segmentation with only global alignment.
- **Transferable trick:** **Weights:** CXR-BERT and BioViL image/text encoders are public on HuggingFace — the text encoder is English-only, but the recipe (custom vocab + section-aware NLI pretext task) transfers: train a small multilingual CXR-BERT analog on the competition reports.

### 2.4 BioViL-T (Bannur et al. 2023) — weights public
- **Claim:** Exploiting the temporal structure of prior–current exam pairs during vision–language pretraining yields SOTA progression classification, phrase grounding, and report generation.
- **Source:** Bannur S, Hyland S, Liu Q, et al. "Learning to Exploit Temporal Structure for Biomedical Vision–Language Processing." CVPR 2023 (poster abstract, CVPR virtual site).
- **URL:** https://cvpr.thecvf.com/virtual/2023/poster/21900 ; weights: HuggingFace `microsoft/BiomedVLP-BioViL-T`
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "BioViL-T, uses a CNN--Transformer hybrid multi-image encoder trained jointly with a text model. It is designed to be versatile to arising challenges such as pose variations and missing input images across time. The resulting model excels on downstream tasks both in single- and multi-image setups, achieving state-of-the-art (SOTA) performance on (I) progression classification, (II) phrase grounding, and (III) report generation, whilst offering consistent improvements on disease classification and sentence-similarity tasks."
- **Confidence:** High.
- **Why it matters:** Its CNN–Transformer hybrid multi-image encoder is directly relevant to multi-series/multi-plane knee MRI: it shows how to fuse multiple image inputs with a text model while being robust to missing inputs.
- **Key methods:** multi-image encoder with temporal self-supervision; missing-modality handling; MS-CXR-T temporal benchmark.
- **Key results:** SOTA on progression classification, phrase grounding, report generation; gains on single-image tasks too.
- **Transferable trick:** Treat the multiple MRI sequences/planes of each knee exam like BioViL-T's multi-image input: shared CNN encoder + transformer across series with missing-series dropout for robustness. **Weights:** public (Microsoft HiML) — candidate Kaggle external model.

### 2.5 MedCLIP (Wang et al. 2022) — weights/code public
- **Claim:** Decoupling image and text corpora (combinatorially expanding pairs) and replacing InfoNCE with a medical-knowledge semantic-matching loss eliminates false negatives and gives extreme data efficiency.
- **Source:** Wang Z, Wu Z, Agarwal D, Sun J. "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text." EMNLP 2022, pp. 3876–3887 (arXiv:2210.10163).
- **URL:** https://ar5iv.labs.arxiv.org/html/2210.10163 ; code: https://github.com/RyanWangZf/MedCLIP
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Surprisingly, we observe that with only 20K pre-training data, MedCLIP wins over the state-of-the-art method (using ≈200K data)." And: "It wins over baselines by a large margin: over 10% improvement of prediction ACC for zero-shot prediction and supervised image classification tasks on average; over 2% improvement of retrieval precision."
- **Confidence:** High.
- **Why it matters:** Medical contrastive learning's hidden flaw is *false negatives* (other patients' reports describing the same finding). The competition's report labels have the same structure — many knees share findings — so pair-based InfoNCE wastes supervision.
- **Key methods:** UMLS/entity-based semantic similarity between image labels and report sentences as soft targets; use of image-only and text-only datasets.
- **Key results:** Beats GLoRIA with ~1/10 the pretraining data; >10% average ACC gains on zero-shot and supervised tasks.
- **Transferable trick:** Replace hard one-to-one image↔report contrastive pairing with soft semantic-similarity targets derived from extracted finding labels (or LLM-parsed findings) — this both scales and denoises pretraining on the knee corpus. **Weights:** code and pretrained checkpoints public on GitHub.

### 2.6 CheXzero (Tiu et al. 2022) — weights public
- **Claim:** CLIP-style contrastive training on unannotated chest X-rays + raw report impressions achieves zero-shot pathology detection not significantly different from board-certified radiologists on CheXpert, and transfers cross-country to PadChest.
- **Source:** Tiu E, Talius E, Patel P, Langlotz CP, Ng AY, Rajpurkar P. "Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning." Nature Biomedical Engineering 2022;6:1399–1406.
- **URL:** https://pubmed.ncbi.nlm.nih.gov/36109605/ ; code+checkpoints: https://github.com/rajpurkarlab/CheXzero
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Using no labels, we outperformed a fully supervised approach (100% of labels) on 3 out of the 8 selected pathologies on a dataset (PadChest) collected in a different country. We further demonstrated high performance (AUC > 0.9) on 14 findings and at least 0.700 on 53 findings out of 107 radiographic findings that the method had not seen during training." Also: "Model checkpoints of CheXzero pre-trained on MIMIC-CXR are publicly available."
- **Confidence:** High.
- **Why it matters:** The strongest evidence that *raw reports alone* (no extracted labels at all) can supervise an image model to expert level — a direct template for learning from the competition's noisy multilingual reports.
- **Key methods:** CLIP architecture initialized from CLIP weights; impression-section text; positive/negative prompt pairs for zero-shot inference; prompt ensembling.
- **Key results:** CheXpert zero-shot mean AUC 0.889 (vs supervised DenseNet-121 0.902); MCC not significantly different from radiologists on 5 pathologies.
- **Transferable trick:** If a report-text encoder for the report languages is available, train a knee CheXzero: image encoder ↔ impression text contrastive; then either zero-shot with per-pathology prompt pairs as a baseline/pseudo-labeler, or fine-tune the image encoder with the noisy labels. **Weights:** public MIMIC-CXR checkpoints (English CXR domain — distribution shift to knee MRI is large; use mainly for the recipe).

### 2.7 BiomedCLIP (Zhang et al. 2023) — weights public, permissive license
- **Claim:** A CLIP-style foundation model pretrained on PMC-15M (15M biomedical image–text pairs from 4.4M PMC articles) with PubMedBERT text encoder achieves SOTA across retrieval, classification, and VQA, even beating radiology-specific models on radiology tasks.
- **Source:** Zhang S, Xu Y, Usuyama N, et al. "BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs." arXiv:2303.00915 (2023; v3 Jan 2025).
- **URL:** https://arxiv.org/pdf/2303.00915 ; weights: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (Apache 2.0, aka.ms/biomedclip)
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "BiomedCLIP achieved new state-of-the-art results in a wide range of standard datasets, including retrieval, image classification, and visual question answering, substantially outperforming prior approaches. Intriguingly, by large-scale pretraining on diverse biomedical image types, BiomedCLIP even outperforms state-of-the-art radiology-specific models such as BioViL in radiology-specific tasks such as RSNA pneumonia detection."
- **Confidence:** High.
- **Why it matters:** The most practical off-the-shelf, permissively-licensed biomedical image–text backbone: includes MRI-like biomedical figures in pretraining, and its text encoder handles biomedical English well.
- **Key methods:** ViT-B/16 + PubMedBERT; PMC-15M scale-up; domain-specific tokenization and adaptations.
- **Key results:** 56%/77% top-1/top-5 retrieval on 725K held-out pairs; SOTA zero/few-shot across radiology and pathology benchmarks.
- **Transferable trick:** **Weights:** public, Apache 2.0 — strong candidate as the image-encoder initialization and/or text-embedding backbone for fusing report features in a Kaggle pipeline (check competition external-data rules). Fine-tune lightly rather than from scratch.

### 2.8 LumbarCLIP (2025) — musculoskeletal-specific VLP
- **Claim:** A CLIP-style framework aligning lumbar spine MRI with radiology report text achieves 95.0% accuracy / 94.75% F1 on downstream classification — the closest musculoskeletal analog of report-supervised MRI learning.
- **Source:** Mai Tan H, et al. "Revolutionizing Precise Low Back Pain Diagnosis via Contrastive Learning" (LumbarCLIP). arXiv:2509.20813 (2025).
- **URL:** https://arxiv.org/abs/2509.20813
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "We present LumbarCLIP, a novel multimodal framework that leverages contrastive language-image pretraining to align lumbar spine MRI scans with corresponding radiological descriptions. ... Our model achieves state-of-the-art performance on downstream classification, reaching up to 95.00% accuracy and 94.75% F1-score on the test set, despite inherent class imbalance. Extensive ablation studies demonstrate that linear projection heads yield more effective cross-modal alignment than non-linear variants."
- **Confidence:** High (abstract); preprint, not peer-reviewed.
- **Why it matters:** Proof-of-concept that MRI+report contrastive pretraining works specifically in MSK imaging; its ablation that *linear* projection heads beat non-linear ones is a cheap, directly copyable design choice.
- **Key methods:** ResNet-50/ViT/Swin encoders + BERT text encoder; shared embedding space; soft CLIP loss.
- **Key results:** See excerpt.
- **Transferable trick:** For a knee CLIP: use linear (not MLP) projection heads, and align at the *axial-view/series* level with report sentences.

### 2.9 OrthoFoundation (2025–2026) — knee-specific vision foundation model, weights public
- **Claim:** A large self-supervised vision foundation model pretrained on 1.25M knee images (357,670 radiographs + 893,985 MRI slices) transfers across 17 MSK tasks including knee MRI diagnosis and cross-joint transfer.
- **Source:** Yu K, Wang D, Yuan Z, et al. "OrthoFoundation: A large-scale multimodal vision foundation model for generalizable musculoskeletal diagnosis and prognosis." GitHub repository (2026); paper associated.
- **URL:** https://github.com/ytrsk/OrthoFoundation
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "The study pretrained OrthoFoundation on 1,251,655 knee images, including 357,670 radiographs and 893,985 MRI slices from public repositories and a private multicenter cohort. The model was then fully fine-tuned and evaluated on 17 downstream tasks covering knee X-ray, knee MRI, long-term osteoarthritis prognosis, and cross-joint transfer to the hip, shoulder, and ankle." and "Training scripts, configuration files, and model weights are released through this project repository."
- **Confidence:** High for repository contents; Medium for paper details (preprint status unclear).
- **Why it matters:** This is the single most on-target pretrained backbone for the competition: knee MRI is its home domain. Vision-only (DINO-style) but complements report-text supervision.
- **Key methods:** DINOv2/DINOv3 student–teacher pretraining on ViT backbones; multi-crop.
- **Key results:** Evaluated on 17 clinical tasks (per repo overview).
- **Transferable trick:** **Weights:** released (OrthoFoundation-L checkpoint) — initialize the knee MRI encoder from OrthoFoundation-L instead of ImageNet; then apply report-supervised fine-tuning or multimodal fusion on top. Verify license and external-model rules.

### 2.10 Decipher-MR (2026) — 3D MRI vision–language foundation model
- **Claim:** A vision–language foundation model for 3D MRI representations demonstrates the modality-general direction of MRI+report pretraining across anatomical regions including musculoskeletal.
- **Source:** "Decipher-MR: a vision-language foundation model for 3D MRI representations." npj Digital Medicine / Nature portfolio (2026).
- **URL:** https://www.nature.com/articles/s41746-026-02596-4
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Magnetic Resonance Imaging (MRI) is a cornerstone of modern medical imaging... It plays a crucial role in diagnosing and monitoring conditions in neurology, cardiology, musculoskeletal health, etc." and "Extensive research has explored both vision-only and vision-language pretraining for specific modalities–such as X-ray, CT, and pathology images–demonstrating the effectiveness of both approaches in capturing modality-specific features, with the latter excelling at capturing cross-modal representations."
- **Confidence:** Medium (introduction verified; specific results not extracted).
- **Why it matters:** Signals that 3D MRI+report VLP is now feasible at foundation scale; volumetric (3D) alignment matters for MRI where abnormality evidence spans slices.
- **Key methods:** 3D vision–language pretraining on MRI volumes + reports (details beyond scope of excerpt).
- **Key results:** Not extracted.
- **Transferable trick:** Consider 3D/series-level (not just slice-level) alignment when pairing knee MRI with reports; check whether Decipher-MR weights are released for use as an encoder.

---

## SECTION 3 — Learning from Noisy Report-Derived Labels

### 3.1 Snorkel / data programming (Ratner et al. 2017/2018)
- **Claim:** Weak-supervision "data programming" lets users combine multiple noisy labeling heuristics (labeling functions) into probabilistic training labels via a generative model that estimates each source's accuracy and correlations without ground truth, coming within ~3.6% of models trained on large hand-labeled sets.
- **Source:** Ratner A, Bach SH, Ehrenberg H, Fries J, Wu S, Ré C. "Snorkel: Rapid Training Data Creation with Weak Supervision." PVLDB 11(3):269–282, 2018 (arXiv:1711.10160; PMC5951191).
- **URL:** https://arxiv.org/abs/1711.10160 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC5951191/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "In a user study, subject matter experts build models 2.8x faster and increase predictive performance an average 45.5% versus seven hours of hand labeling. ... Snorkel provides 132% average improvements to predictive performance over prior heuristic approaches and comes within an average 3.60% of the predictive performance of large hand-curated training sets." Also: "Snorkel automatically learns a generative model over the labeling functions, which allows it to estimate their accuracies and correlations. This step uses no ground-truth data, learning instead from the agreements and disagreements of the labeling functions."
- **Confidence:** High.
- **Why it matters:** The competition gives one noisy label source (reports). Snorkel theory says: *create more* weak sources (rule labeler, LLM labeler, zero-shot VLP model, keyword lists per language) and learn their reliability jointly — the probabilistic consensus beats any single source.
- **Key methods:** labeling functions → generative label model → probabilistic labels → discriminative end model.
- **Key results:** See excerpt; generative weighting adds 5.81% over unweighted majority vote.
- **Transferable trick:** Build a label-model layer over multiple knee-report labelers (regex rules + multilingual BERT + LLM + CheXzero-style zero-shot image model) and train the final image model on the resulting soft probabilities instead of the raw binary labels.

### 3.2 Co-teaching (Han et al. 2018)
- **Claim:** Two networks trained simultaneously, each selecting small-loss (likely clean) samples to teach its peer, are far more robust to extreme label noise than standard training.
- **Source:** Han B, Yao Q, Yu X, Niu G, Xu M, Hu W, Tsang I, Sugiyama M. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." NeurIPS 2018.
- **URL:** https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "recent studies on the memorization effects of deep neural networks show that they would first memorize training data of clean labels and then those of noisy labels. Therefore in this paper, we propose a new deep learning paradigm called ''Co-teaching'' for combating with noisy labels. Namely, we train two deep neural networks simultaneously, and let them teach each other given every mini-batch ... Empirical results on noisy versions of MNIST, CIFAR-10 and CIFAR-100 demonstrate that Co-teaching is much superior to the state-of-the-art methods in the robustness of trained deep models."
- **Confidence:** High.
- **Why it matters:** Report-derived labels are classically noisy (CheXpert-label disagreement with humans can be 10–30% on some findings). Small-loss selection exploits the fact that the image model learns clean knee-abnormality patterns before overfitting label errors.
- **Key methods:** dual-network peer sample selection by per-batch small-loss criterion; decaying keep-ratio schedule.
- **Key results:** Superior robustness to symmetric/asymmetric noise up to extreme rates on MNIST/CIFAR.
- **Transferable trick:** Train two knee classifiers (different seeds/architectures); each epoch, exchange the lowest-loss ~80–90% of samples. Nearly free to implement and a strong baseline against report-label noise; combine with early stopping.

### 3.3 Loss correction via noise transition matrix (Patrini et al. 2017)
- **Claim:** Forward/backward loss correction using an estimated label-noise transition matrix makes deep networks provably robust to label noise without architecture changes.
- **Source:** Patrini G, Rozza A, Menon AK, Nock R, Qu L. "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach." CVPR 2017, pp. 1944–1952 (2233–2241 in IEEE proceedings listing).
- **URL:** https://doi.org/10.1109/CVPR.2017.240 (citation verified across multiple arXiv reference lists)
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim, bibliographic):** "Patrini, G.; Rozza, A.; Menon, A. K.; Nock, R.; and Qu, L. 2017. Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, 2233–2241. IEEE Computer Society."
- **Confidence:** High (citation); Medium for method details (not directly excerpted here).
- **Why it matters:** If you can *estimate* the noise rate of report-derived labels per class (e.g., from a small hand-checked subset), loss correction is a one-line change with theoretical backing.
- **Key methods:** estimate class-conditional flip probabilities T; apply T (forward) or T⁻¹ (backward) to network outputs in the loss.
- **Key results:** In DivideMix's reproduction table (below), F-correction improves CIFAR-10 40% asymmetric noise accuracy 85.0 → 87.2 over cross-entropy.
- **Transferable trick:** Hand-verify a few hundred knee reports to estimate per-pathology sensitivity/specificity of the provided labels; plug the flip matrix into a forward-corrected BCE.

### 3.4 DivideMix (Li et al. 2020)
- **Claim:** Modeling per-sample loss distributions with a two-component Gaussian mixture to split clean/noisy data, then treating noisy samples as unlabeled in a semi-supervised MixMatch loop with co-divided training, gives ~10% accuracy gains at high noise.
- **Source:** Li J, Socher R, Hoi SCH. "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." ICLR 2020 (arXiv:2002.07394).
- **URL:** https://ar5iv.labs.arxiv.org/html/2002.07394
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Different from most existing LNL approaches, DivideMix discards the sample labels that are highly likely to be noisy, and leverages the noisy samples as unlabeled data to regularize the model from overfitting." Table 2 (CIFAR-10, 40% asymmetric noise): "Cross-Entropy 85.0/72.3; F-correction 87.2/83.1; ... DivideMix 93.4/92.1."
- **Confidence:** High.
- **Why it matters:** The SOTA-anchoring recipe for exactly this competition's situation: some unknown fraction of training labels are wrong, and unlabeled-image regularization recovers most of the loss.
- **Key methods:** GMM on per-sample loss; co-guessing label sharpening; MixMatch with two networks.
- **Key results:** >12% top-1 improvement on WebVision; ~10% on CIFAR-100 high-noise.
- **Transferable trick:** Fit a GMM on training losses mid-training for each knee pathology; down-weight/unlabel the "noisy" component (reports with LLM/image-model disagreement are prime suspects) and apply consistency regularization on those MRIs.

### 3.5 Learning robust CXR classifiers from NLP-derived noisy labels with radiologist-measured priors (Gündel et al. 2021)
- **Claim:** Report-NLP labels carry substantial noise; measuring per-class label-error probabilities on a radiologist re-read subset and injecting them as priors into the loss (plus comorbidity and anatomy side-tasks) yields SOTA performance (avg AUC 0.880 over 17 abnormalities on 297,541 radiographs).
- **Source:** Gündel S, et al. "Robust Classification from Noisy Labels: Integrating Additional Knowledge for Chest Radiography Abnormality Assessment." arXiv:2104.05261 (2021; ML4H/journal version).
- **URL:** https://arxiv.org/abs/2104.05261
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "However, the labels of these datasets were obtained using natural language processed medical reports, yielding a large degree of label noise that can impact the performance. In this study, we propose novel training strategies that handle label noise from such suboptimal data. Prior label probabilities were measured on a subset of training data re-read by 4 board-certified radiologists and were used during training to increase the robustness of the training model to the label noise. ... With an average AUC score of 0.880 across all abnormalities, our proposed training strategies can be used to significantly improve performance scores."
- **Confidence:** High.
- **Why it matters:** The most directly transferable methodology paper: it quantifies report-label noise with expert re-reads and converts that estimate into a noise-robust training loss.
- **Key methods:** noise-transition priors from re-reads; label-correlation modeling; auxiliary segmentation/anatomy tasks; multi-dataset normalization.
- **Key results:** See excerpt.
- **Transferable trick:** Even without radiologists, approximate the "re-read" with a strong LLM ensemble or zero-shot image model to estimate per-class noise rates, then train with those priors. Auxiliary anatomy prediction (e.g., meniscus/ACL segmentation if available) further regularizes.

### 3.6 Evidence review: how sensitive are image models to report-label noise? (Wei et al. 2024; Karimi et al. 2020; Yang et al. 2023)
- **Claim A:** Even small label-error fractions measurably hurt chest X-ray classifiers: "Jang et al found that even a 2% random flip of labels significantly affects chest X-ray prediction accuracy."
- **Source A:** Wei Y, et al. "Deep learning with noisy labels in medical prediction problems: scoping review." JAMIA 2024;31(7):1596 (academic.oup.com).
- **URL A:** https://academic.oup.com/jamia/article/31/7/1596/7685298
- **Confidence A:** High (review citing primary study).
- **Claim B:** Label smoothing on report-derived CXR labels improved AUC by up to 0.08 versus ignoring noisy samples.
- **Source B:** Karimi D, et al. "Deep learning with noisy labels: exploring techniques and remedies in medical image analysis." IEEE TMI 2020 (arXiv:1912.02911), citing Pham et al.
- **URL B:** https://arxiv.org/pdf/1912.02911v4
- **Excerpt B (verbatim):** "For classification of thoracic diseases from chest x-ray scans, [71] used label smoothing to handle noisy labels. They compared their label smoothing method with simple methods such as ignoring data samples with noisy labels. They found that label smoothing can lead to improvements of up to 0.08 in the area under the receiver operating characteristic curve (AUC)."
- **Confidence B:** High.
- **Claim C:** Detecting and removing noisy-label CXRs (validated by radiologists) significantly improved 8 of 14 abnormality classifiers, reaching AUC 0.827 on ChestX-ray14.
- **Source C:** Yang M, et al. "Performance improvement in multi-label thoracic abnormality classification of chest X-rays with noisy labels." Int J CARS 2023;18(1):181–189 (PMID 35616775).
- **URL C:** https://pubmed.ncbi.nlm.nih.gov/35616775/
- **Excerpt C (verbatim):** "Report from the radiologists indicated that detected noisy labels had high possibility to be true positives. ... After removing the CXRs with detected noisy labels, 8 out of 14 abnormalities improved significantly on CXR14. The suggested framework achieved AUC score of 0.827 on CXR14."
- **Confidence C:** High.
- **Why it matters (all three):** Quantifies the stakes: report labels are noisy, the noise measurably caps performance, and cheap remedies (label smoothing, noise detection, priors) recover real AUC.
- **Transferable trick:** Start with label smoothing ε≈0.05–0.1 on the provided report labels (cheapest, evidence-backed); escalate to noise detection (high-loss + LLM/image-model disagreement) and sample down-weighting rather than deletion.

---

## SECTION 4 — Direct Image + Report-Text Fusion Classifiers

### 4.1 Multimodal medical tensor fusion network (Shetty/Mahale et al. 2023)
- **Claim:** Fusing CXR image features with report text embeddings via bilinear-style tensor fusion (CBP/DHP) outperforms both unimodal models and plain concatenation fusion by wide margins (image-only 79.2% / text-only 90.4% / fused 97.4% accuracy on Indiana University data).
- **Source:** Shetty S, Mahale A, et al. "Multimodal medical tensor fusion network-based DL framework for abnormality prediction from the radiology CXRs and clinical text reports." Multimedia Tools and Applications 2023;82:44431–44478 (PMC10119019; PMID 37362656).
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10119019/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "The proposed multimodal models have given superior results compared to the unimodal models." Results table (Indiana University): "Image Only UM-VES 79.22% acc / 0.8572 AUROC; Text Only UM-TES 90.40% / 0.9555; Image + Text CBP-MMFN 97.35% / 0.9876." Also: "Both the proposed multimodal models have shown 14-15% more accuracy compared to the existing model fusion technique [5], where they have used concatenation for fusing the tensors."
- **Confidence:** High.
- **Why it matters:** Direct evidence for the "fuse report text with image features at inference" design: text carries diagnostic context images lack, and interaction-aware fusion (bilinear/Hadamard) beats naive concatenation.
- **Key methods:** unimodal encoders (UM-VES, UM-TES) + Compact Bilinear Pooling / Deep Hadamard Product fusion + DNN head.
- **Key results:** See table; consistent gains on private KMC cohort too (82.3/94.1 → 96.9%).
- **Transferable trick:** If the competition provides test-time reports, a fused head wins; if reports are train-only, use them to build a better text embedding target. Either way, prefer a bilinear/Hadamard interaction term over plain concat when fusing MRI and report embeddings.

### 4.2 Joint multimodal + self-supervised pretraining from reports (Huang et al., MIDL 2023)
- **Claim:** Combining multimodal (image–report) learning with self-supervised learning outperforms SSL alone at 1% and 10% label budgets and is more robust out-of-distribution.
- **Source:** Huang H, Rawlekar S, Chopra S, Deniz CM. "Radiology Reports Improve Visual Representations Learned from Radiographs." MIDL 2023, PMLR 227:1385–1405.
- **URL:** https://proceedings.mlr.press/v227/huang24a.html ; code: https://github.com/denizlab/MIMICCXR-MultiModal-SelfSupervision
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Our experiments indicated that in limited labeled data settings with 1% and 10% labeled data, the joint learning with multi-modal and self-supervised models outperforms self-supervised learning and is at par with multi-modal learning. Additionally, we found that multi-modal learning is generally more robust on out-of-distribution datasets."
- **Confidence:** High.
- **Why it matters:** Practical recipe: don't choose between report-supervision and image-only SSL — stack them. OOD robustness matters when the hidden test set differs from public training data.
- **Key methods:** benchmark of multimodal vs SSL vs joint objectives on MIMIC-CXR with downstream CheXpert/ChestX-ray14 evaluation.
- **Key results:** See excerpt.
- **Transferable trick:** Pretrain the knee encoder with *both* an image-only SSL loss (DINO/MAE on slices) and an image–report contrastive loss; expect better few-label and OOD behavior than either alone.

### 4.3 Geometric multimodal foundation model fusing bp-MRI + clinical reports for prostate cancer (2026)
- **Claim:** Fusing BiomedCLIP image embeddings with report text through a geometric (SPD-manifold) head and a combined BCE + InfoNCE fine-tuning loss improves clinically-graded prostate cancer classification over unimodal and ImageNet baselines.
- **Source:** "A Geometric Multimodal Foundation Model Integrating Bp-MRI and Clinical Reports in Prostate Cancer Classification." arXiv:2602.00214 (2026).
- **URL:** https://arxiv.org/html/2602.00214v1
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "We employed BiomedCLIP FM, pretrained on 15 million image–text pairs extracted from biomedical articles in PubMed Central. The image encoder is a ViT-B/16. The text encoder is PubMedBERT... For fine-tuning, the loss function is the sum of the binary cross-entropy and contrastive (InfoNCE) loss."
- **Confidence:** Medium-High (preprint; methods verified, results section not fully extracted).
- **Why it matters:** A 2026 MRI + report fusion template: freeze a biomedical FM encoder, fuse with report text embeddings, and keep a contrastive term in the fine-tuning loss to retain cross-modal alignment.
- **Key methods:** frozen BiomedCLIP encoders; SPD/geometric fusion blocks; BCE + InfoNCE objective; 5-fold CV on PI-CAI-style cohort.
- **Key results:** Ablations compare unimodal FM, ImageNet init, and multimodal geometric model (metrics via AUC-PR, FPR95).
- **Transferable trick:** Keep an InfoNCE term between MRI and report embeddings *during supervised fine-tuning* — it regularizes the image encoder and aligns fusion; works with frozen BiomedCLIP backbones.

### 4.4 VisualCheXbert — vision+text fusion for report labeling (Jain et al. 2021)
- **Claim:** Fusing image features with report text improves the *labeler itself* (F1 ≈ 0.73, AUROC ≈ 0.87 on benchmark tasks), because images resolve report ambiguity.
- **Source:** Jain S, Pareek A, et al. "VisualCheXbert: Addressing the Discrepancy Between Clinical Experience and Computer Vision Report Labels" (MICCAI 2021 workshop lineage); figures via EmergentMind CheXpert overview table.
- **URL:** https://www.emergentmind.com/topics/chexpert-labeler
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim, from comparison table):** "|VisualCheXbert|Vision+text fusion|Discrete label|Yes|No|—|F₁: 0.73; AUROC: 0.87|"
- **Confidence:** Medium (secondary tabulation; recommend pulling the primary paper for exact numbers).
- **Why it matters:** Inverts the usual direction: images can clean report labels. In this competition, image-model predictions can audit report-derived labels — a virtuous cycle.
- **Transferable trick:** Use your best image classifier's predictions as an extra labeling function (à la Snorkel) when re-estimating the reliability of report labels.

---

## SECTION 5 — Multilingual Radiology NLP

### 5.1 German CheXpert rule labeler (Wollek et al. 2024)
- **Claim:** A rule-based CheXpert adaptation for German thoracic reports produces labels good enough that a pneumothorax classifier trained on them beat one trained on public data and matched manual-label training — without annotation time.
- **Source:** Wollek A, Hyska S, Sedlmeyr T, et al. "German CheXpert Chest X-ray Radiology Report Labeler." Fortschr Röntgenstr 2024;196:956–965 (PMID 38295825).
- **URL:** https://pubmed.ncbi.nlm.nih.gov/38295825/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Automatic label extraction from German thoracic radiology reports is a promising substitute for manual labeling. By reducing the time required for data annotation, larger training data sets can be created, resulting in improved overall modeling performance. Our results demonstrated that a pneumothorax classifier trained on automatically extracted labels strongly outperformed the model trained on publicly available data, without the need for additional annotation time and performed competitively compared to manually labeled data."
- **Confidence:** High.
- **Why it matters:** Proof the CheXpert paradigm ports to non-English reports — the same holds for whatever languages appear in the knee corpus.
- **Key methods:** translated/expanded trigger and phrase lists; web multi-reader annotation interface for iterative rule improvement; 1,086-report reference standard.
- **Key results:** See excerpt (downstream AUCs in companion paper below).
- **Transferable trick:** For each report language: build language-specific mention/negation/uncertainty cue lists with a fast human-in-the-loop review of ~1k reports; rules beat zero-effort translation in structured label extraction.

### 5.2 German BERT labeler with weak supervision (Wollek et al. 2025)
- **Claim:** A German-BERT labeler pretrained on 66k rule-labeled reports and fine-tuned on ~1k manual labels beats the rule labeler on mention/negation/uncertainty F1, and a DenseNet-121 trained on its labels (AUC 0.939) slightly beats one trained on manual labels (0.934).
- **Source:** Wollek A, Haitzer P, Sedlmeyr T, et al. "Language model-based labeling of German thoracic radiology reports." Fortschr Röntgenstr 2025;197:55–64 (PMID 38663428; DOI 10.1055/a-2287-5054). Code: gitlab.lrz.de/IP/german-lm-radiology-report-labeler.
- **URL:** https://pubmed.ncbi.nlm.nih.gov/38663428/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "The proposed deep learning-based labeler (DL) performed on average considerably stronger than the rule-based labeler (RB) for all three tasks on DS 1 with F1 scores of 0.938 vs. 0.844 for mention extraction, 0.891 vs. 0.821 for negation detection, and 0.624 vs. 0.518 for uncertainty detection. ... Chest X-ray pneumo-thorax classification results (DS 2) were highest when trained with DL labels with an area under the receiver operat-ing curve (AUC) of 0.939 compared to R8 labels with an AUC of 0.858. Training with manual labels performed slightly worse than training with DL labels with an AUC of 0.934. In contrast, training with a public data set resulted in an AUC of 0.720."
- **Confidence:** High.
- **Why it matters:** The single most relevant multilingual pipeline: rules → weak pretraining → tiny manual fine-tune, in a non-English language, *with downstream image-model validation*. Note the remarkable finding that DL labels beat manual labels downstream (consistency > individual accuracy).
- **Key methods:** bert-base-German-cased + 14 linear heads; weak-supervision pretraining; 1,091 manual reports.
- **Key results:** See excerpt.
- **Transferable trick:** Per-language recipe for the knee reports: (1) seed rules, (2) weak-pretrain a language-specific BERT, (3) fine-tune on ≤1k hand-checked reports, (4) use model labels at scale. Consistent automatic labels can outperform sparse manual ones for training image models.

### 5.3 Med-UniC — cross-lingual medical VLP (Wan et al., NeurIPS 2023)
- **Claim:** Cross-lingual Text Alignment Regularization (CTR) unifies English and Spanish report semantics in medical VLP, mitigating language-community bias; reducing that bias improves even uni-modal visual tasks. SOTA across 5 tasks / 10 datasets / 30+ diseases.
- **Source:** Wan Z, Liu C, Zhang M, et al. "Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias." NeurIPS 2023 (arXiv:2305.19894). Code: github.com/SUSTechBruce/Med-UniC.
- **URL:** https://openreview.net/forum?id=4vpsQdRBlK
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "This paper presents a novel framework named Unifying Cross-Lingual Medical Vision-Language Pre-Training (Med-UniC), designed to integrate multi-modal medical data from the two most prevalent languages, English and Spanish. ... CTR is optimized through latent language disentanglement, rendering our optimization objective to not depend on negative samples ... Med-UniC reaches superior performance across 5 medical image tasks and 10 datasets encompassing over 30 diseases ... The experimental outcomes highlight the presence of community bias in cross-lingual VLP. Reducing this bias enhances the performance not only in vision-language tasks but also in uni-modal visual tasks."
- **Confidence:** High.
- **Why it matters:** The key multilingual-VLP insight: reports in different languages describe the same findings, but naive multilingual training lets language identity contaminate embeddings. For a multilingual knee corpus this bias is a first-order concern.
- **Key methods:** latent language disentanglement; negative-free CTR objective; mixed EN/ES vocabulary and post-pretraining of CXR-BERT.
- **Key results:** See excerpt.
- **Transferable trick:** Add a language-adversarial/disentanglement term (or simply balanced per-language sampling + shared finding-label targets) so the text encoder maps equivalent findings across languages to nearby embeddings; audit zero-shot performance per language to detect community bias.

### 5.4 Multilingual CheXbert in Spanish (Stanford CS224n project)
- **Claim:** The CheXbert recipe (weak rule labels + small expert set + back-translation) transfers to Spanish reports; back-translation is highlighted as the cross-language enabler.
- **Source:** "Multilingual CheXbert: Radiology Report Labeling in Spanish." Stanford CS224n final report (course project; lower authority).
- **URL:** https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report075.pdf
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "The success of CheXbert demonstrated that the overall approach of training on a combination of automatically- and manually-labeled data (and the use of backtranslation as a dataset augmentation technique) could be worth applying across other medical domains-and other languages. Where high-quality expert-labeled data is not so widely available or easy to obtain, training in conjunction with automatically-labeled data can yield promising results."
- **Confidence:** Medium (student project; directional evidence only).
- **Transferable trick:** Round-trip machine translation (EN↔report language) both augments scarce manual labels and provides a cross-lingual consistency check on extracted labels.

### 5.5 Vietnamese PET/CT report–volume VLM dataset (NeurIPS 2025)
- **Claim:** First Vietnamese whole-body PET/CT + full-report multimodal dataset (2,757 volumes), showing medical VLM training in low-resource languages improves downstream report generation and VQA, though diagnostic gaps remain.
- **Source:** "Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation." NeurIPS 2025 poster.
- **URL:** https://neurips.cc/virtual/2025/poster/121676
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "Most existing medical VLMs are trained on a subset of imaging modalities and focus primarily on high-resource languages, thus limiting their generalizability and clinical utility. ... The experimental results show that incorporating our dataset significantly improves the performance of existing VLMs. However, despite these advancements, the models still underperform on clinically critical criteria, particularly the diagnosis of lung cancer, indicating substantial room for future improvement."
- **Confidence:** High (poster abstract).
- **Why it matters:** Confirms (a) low-resource-language medical VLP is feasible and beneficial, and (b) diagnostic classification is the hardest transfer target — so don't rely on a generic multilingual VLM for knee abnormality detection without task-specific fine-tuning.
- **Transferable trick:** If report languages include lower-resource ones, fine-tune/continue-pretrain a multilingual encoder (XLM-R, mDeBERTa) on the in-corpus reports rather than trusting zero-shot multilingual medical coverage.

---

## SECTION 6 — Knee-MRI Anchors & Competition Synthesis

### 6.1 MRNet (Bien et al. 2018) — the knee-MRI benchmark itself
- **Claim:** MRNet (per-series CNN + logistic-regression stacking across 3 planes), trained on labels extracted from clinical reports, reached AUCs 0.937 (abnormality) / 0.965 (ACL tear) / 0.847 (meniscal tear) against musculoskeletal-radiologist reference standards, with no significant difference from general radiologists on abnormality detection.
- **Source:** Bien N, Rajpurkar P, Ball RL, et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet." PLoS Medicine 2018;15(11):e1002699 (PMC6258509).
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "In detecting abnormalities, ACL tears, and meniscal tears, this model achieved area under the receiver operating characteristic curve (AUC) values of 0.937 (95% CI 0.895, 0.980), 0.965 (95% CI 0.938, 0.993), and 0.847 (95% CI 0.780, 0.914), respectively, on the internal validation set." And: "Labels were prospectively obtained through manual extraction from clinical reports." Also: "the MRNet trained on Stanford sagittal T2-weighted series achieved an AUC of 0.824 (95% CI 0.757, 0.892) in the detection of ACL injuries with no additional training, while an MRNet trained on the rest of the external data achieved an AUC of 0.911 (95% CI 0.864, 0.958)."
- **Confidence:** High.
- **Why it matters:** This competition is essentially "MRNet with report-NLP labels at RSNA scale." MRNet's numbers are the performance envelope reference, and its external-validation drop (0.965 → 0.824 zero-transfer) warns about domain shift.
- **Key methods:** AlexNet per-slice features + max-pool over slices per series; one model per (task × plane); logistic regression stack of 3 planes; ImageNet init; rotation/shift/flip augmentation.
- **Key results:** See excerpt; model assistance significantly increased experts' ACL specificity.
- **Transferable trick:** The per-series-model + learned-stacking design remains a strong, compute-cheap baseline for multi-sequence knee MRI; max-pooling over slices (not averaging) preserves focal tear signal.

### 6.2 Knee-MRI deep learning systematic review (European Radiology 2024)
- **Claim:** Across knee-MRI DL studies, average sensitivity/specificity/AUC/accuracy are 88.65%/90.12%/92.05%/88.30%; pathology-specific training beats general-abnormality training by up to ~4.5 points.
- **Source:** "MRI deep learning models for assisted diagnosis of knee pathologies: a systematic review." European Radiology 2024 (Springer).
- **URL:** https://link.springer.com/article/10.1007/s00330-024-11105-8
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "DL model performance outcomes were averaged across knee pathologies for sensitivity, specificity, AUC-ROC, and classification accuracy, reporting 88.65%, 90.12%, 92.05%, and 88.30%, respectively. Specific knee abnormality training improved outcomes, with average specificity, AUC-ROC, accuracy, and sensitivity of 90.57%, 92.72%, 88.91%, and 88.67%, respectively."
- **Confidence:** High.
- **Why it matters:** Sets realistic target metrics and supports per-pathology heads/models rather than one general "abnormal" model.
- **Transferable trick:** Train separate heads (or models) per knee pathology; pooled "any abnormality" training dilutes performance.

### 6.3 SCOPE-MRI (2025) — MRNet-pretraining transfers across joints
- **Claim:** Pretraining on the public MRNet knee dataset and fine-tuning transfers to shoulder MRI (Bankart lesion), with AlexNet/ViT/Swin selected by MRNet validation AUC.
- **Source:** "SCOPE-MRI: Bankart lesion detection as a case study in data curation and deep learning for challenging diagnoses." npj Digital Medicine 2025 (PMC12668287).
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12668287/
- **Date accessed:** 2026-08-10
- **Excerpt (verbatim):** "For pretraining, we used the MRNet dataset, a publicly available knee MRI dataset chosen for its anatomical and contextual relevance to musculoskeletal imaging. Pretraining for the initial architecture search was conducted on the sagittal view of MRNet using the 'abnormal' label as the target. The top-performing architectures—AlexNet, Vision Transformer, and Swin Transformer V1—were identified based on their AUC on the MRNet validation set."
- **Confidence:** High.
- **Why it matters:** MSK-MRI representations transfer across joints; public MRNet is a legal, on-domain pretraining resource for this competition (subject to rules).
- **Transferable trick:** Pretrain on MRNet (all planes, "abnormal" label) before fine-tuning on competition data; use MRNet validation AUC for architecture selection.

---

## CROSS-CUTTING SYNTHESIS FOR THE COMPETITION

1. **Label trust hierarchy (evidence-backed):** LLM-ensemble labels > neural labeler (BERT) > rule labeler > single heuristic; but image classifiers are *resilient* to moderate training-label noise (Sec 1.9, 3.6). Spend effort on a clean validation set, not on perfectly clean training labels.
2. **Cheapest high-yield tricks:** label smoothing (up to +0.08 AUC, Sec 3.6); small-loss selection / co-teaching (Sec 3.2); noise-prior loss correction from a small audited subset (Sec 3.3, 3.5); soft/probabilistic labels instead of hard binaries (Sec 1.4, 3.1).
3. **Text-side upgrades:** continued MLM pretraining on the competition's own multilingual reports (Sec 1.5); per-language BERT labelers with ≤1k manual reports each (Sec 5.2); concept/entity-anchored labels (Sec 1.6, 1.7); language-disentangled text embeddings (Sec 5.3).
4. **Vision-side upgrades:** image–report contrastive pretraining on the competition pairs (Sec 2.1, 2.6); joint SSL + multimodal objectives (Sec 4.2); MSK-domain initializations (OrthoFoundation knee weights, MRNet pretraining — Sec 2.9, 6.3).
5. **Fusion:** if reports are available at inference, bilinear/interaction fusion beats concatenation (Sec 4.1); keep an InfoNCE image↔report term during fine-tuning (Sec 4.3).
6. **Reusable public weights for Kaggle (verify rules/licenses):** BiomedCLIP (Apache 2.0), BioViL/BioViL-T & CXR-BERT (Microsoft HiML, HF), GLoRIA (GitHub), MedCLIP (GitHub), CheXzero (GitHub), OrthoFoundation-L (GitHub), Med-UniC (GitHub), German LM labeler (LRZ GitLab).

## SEARCH LOG (22 independent queries, 2026-08-10)
1. CheXpert labeler Irvin 2019 accuracy · 2. NegBio negation/uncertainty · 3. CheXbert Smit 2020 · 4. ConVIRT Zhang 2020 · 5. BioViL-T Microsoft · 6. MedCLIP unpaired · 7. Snorkel weak supervision · 8. Co-teaching NeurIPS 2018 · 9. CheXzero Nat Biomed Eng · 10. BiomedCLIP PMC-15M · 11. RadGraph · 12. GLoRIA ICCV 2021 · 13. MedCAT Kraljevic · 14. RadBERT · 15. GPT-4 label extraction 2024 · 16. German CheXpert Wollek · 17. Med-UniC cross-lingual · 18. label-noise CXR quantification · 19. knee MRNet musculoskeletal · 20. DivideMix / Patrini loss correction · 21. multimodal tensor fusion CXR+reports · 22. MIDL 2023 reports-improve-radiographs.
