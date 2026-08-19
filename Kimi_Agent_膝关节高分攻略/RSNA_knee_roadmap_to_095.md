# RSNA Knee Abnormality Detection — The Path from ~0.92 to 0.95–0.96

**Date of analysis: 2026-08-19 · Competition closes 2026-10-22 · Metric: macro-averaged ROC AUC over 12 labels**

---

## 1. Where the leaderboard actually stands

Verified against the live public leaderboard on 2026-08-19:

| Rank | Team | Score | Entries |
|---|---|---|---|
| 1 | Brandon Low | **0.951** | 51 |
| 2 | MKhlystun | 0.947 | 13 |
| 3 | CloseAI | 0.946 | 14 |
| 4 | Lukas Nissen Molvær | 0.945 | 8 |
| 5–8 | (several) | 0.942 | — |
| ~50 | — | 0.930 | — |

- ~1,900 teams; the public leaderboard is computed on **~30% of the test data**; the final standings come from the other 70%. [Source: Kaggle leaderboard page, pulled 2026-08-19]
- Publicly shared notebooks/baselines cluster around **0.92** [discussion/735767]; a fully documented rank-18 pipeline scored **0.903** on Aug 9 (already outside the top 50 ten days later — the field is moving fast). [huggingface.co/blog/bishnoiyash]

**So your target 0.95–0.96 is, today, the #1 slot.** The question "knowledge gap or model gap?" has a clear evidence-based answer.

---

## 2. Verdict: it is primarily a knowledge (supervision) gap — not a model gap

Three facts decide this:

1. **Only 58 of 4,407 training studies carry expert image labels.** The other 4,349 must be labeled by mining free-text radiology reports in ~12 languages. The hidden test set, by contrast, was **annotated by expert MSK radiologists** (double-read with adjudication). [Competition data page; RSNA press release; discussion/733343]
2. **The best-documented pipeline in the field measured its own ceiling.** The rank-18 (0.903) writeup reports its LLM label extractor reaches only **~0.83 mean per-class agreement** against expert re-reads, and states plainly: *"The label extractor's own accuracy ceiling is probably close to the real limiting factor on this whole approach right now. Pushing further on image-model architecture alone will run into that ceiling before it runs into anything else."* [HF blog, 2026-08-09]
3. **The image-model side is already a solved pattern.** DINOv2/ViT backbones + per-plane encoding + MIL/attention pooling + fold ensembling is publicly shared and reproducibly reaches ~0.92. Nobody's architecture is exotic; the spread between 0.92 and 0.951 is almost entirely above the neck of the model — in the labels, the validation discipline, and the ensembling.

**A useful way to see it:** the rank-18 pipeline scored 0.903 on Kaggle while its own *honest* cross-validation on the 58 gold studies read only ~0.856. The test labels are cleaner than the training labels. Your model is being trained against noisy targets and evaluated against clean ones — every point of label noise you remove converts almost directly into AUC, because AUC against clean ground truth is bounded by how well your training signal ranks findings, not by how big your backbone is.

**The residual ~20% is a model/ensembling gap:** external-data pretraining, cross-plane fusion quality, TTA, and multi-seed/multi-backbone ensembling are worth roughly the last 0.01–0.015. But nobody reaches 0.95 on architecture alone with 0.83-agreement labels.

### Per-label difficulty ranking (where the AUC is being lost)

From the radiology routing guide and the host's annotation criteria (discussion/733343), ranked hardest → easiest for an image model trained on report-derived labels:

1. **Synovitis** — intrinsically inseparable from effusion on non-contrast MRI; Hoffa-signal specificity 10–38%. Reports also phrase it loosely. Expect this to be the lowest per-label AUC for everyone; it drags the macro average.
2. **Fracture vs Contusion** — distinguished by a linear low-signal line best seen on T1, which the dataset does not guarantee; report language ("bone bruise", "occult fracture", "marrow edema") is inconsistent.
3. **MCL** — high-grade-only host criterion; the ligament sits at FOV/crop edges; grade I/II wording in reports is noisy.
4. **ACL** — host criterion includes high-grade partial tears (>50% fibers), which are genuinely hard even for radiologists.
5. **Medial/Lateral OA & PF OA** — threshold-dependent (≥1 cm of >50% cartilage loss); PF OA depends on axial series, the scarcest plane (~6k axial vs ~10k sagittal series).
6. **Effusion, Baker's cyst, menisci** — easiest: clear imaging correlates, clear report language, moderate-or-large threshold. These are likely near-saturated (>0.97) for the top teams already.

**Implication:** the 0.92 → 0.95 climb is won in labels 1–5 above, and those are exactly the labels where report-mining quality and threshold calibration matter most.

---

## 3. Roadmap to 0.95–0.96, ordered by expected gain

### Lever 1 — Label extraction quality (expected: +0.02 to +0.04; the decisive lever)

1. **Multi-extractor ensemble with per-class reliability weighting.** Run 2–3 independent LLM extractors (different local open-weights families — commercial hosted APIs are banned for report text per host clarification, discussion/733965), measure each extractor's **per-class** accuracy against the 58 gold studies, and fuse votes weighted per class (an extractor weak on Synovitis gets less say on Synovitis). Emit **soft labels** on disagreement, not forced hard votes. This is precisely the machinery the 0.903 writeup describes — and its authors note their submitted run was trained on a *buggy, asymmetric* fusion. A clean implementation of this alone plausibly covers 0.90 → 0.93.
2. **Use the routing guide as the extraction schema.** Feed the extractor per-finding decision rules (host thresholds: ACL = high-grade partial or full; MCL = high-grade acute; meniscus = surface-touching signal on ≥2 images or deformity; OA = ≥1 cm of >50% cartilage loss; effusion/Baker's = moderate-or-large; borderline = negative). Most extractor errors are threshold/phrasing mismatches, not language errors — e.g., reports saying "mild effusion" or "grade 2 intrameniscal signal" must map to **negative**.
3. **Handle multilinguality explicitly.** Translate all reports to English with a strong local MT model *before* extraction, or use a multilingual LLM with per-language few-shot prompts validated on the gold 58. Silent per-language accuracy drift is a classic way to lose 0.01 without noticing.
4. **Iterate: image-model-assisted label reconciliation.** Train the image model on v1 labels → predict on the 4,349 report-only studies → flag studies where image-model and report-label disagree with high confidence → re-extract those reports with a stricter prompt (or drop them). Two or three rounds of this is how noisy-label learning converts weak supervision into clean supervision. Complement with established noisy-label techniques: co-teaching / confident-learning-style per-class noise filtering, and loss functions robust to label noise.

### Lever 2 — External labeled data & pretraining (expected: +0.01 to +0.02)

5. **Pretrain/fine-tune on MRNet, fastMRI+, OAI, SKM-TEA** (external data is explicitly allowed; discussion/733652). MRNet alone gives ~1,370 expert-labeled exams for abnormal/ACL/meniscus — the three highest-prevalence structural labels. Even though planes/sequences differ, representation transfer plus a final fine-tune on RSNA labels consistently beats scratch or purely-ImageNet pipelines in medical-imaging competitions. MRNet-pretrained weights also directly transfer the "what a tear looks like" knowledge that 58 gold studies cannot teach.

### Lever 3 — Image model & input pipeline (expected: +0.005 to +0.015)

6. **Canonicalize orientation and laterality.** No Laterality column exists; derive it from ImagePositionPatient/ImageOrientationPatient geometry (flip coronal/axial, reverse sagittal stack) so "medial" and "lateral" are anatomically consistent. Five of twelve labels are medial/lateral pairs — a laterality bug is a macro-AUC catastrophe. Sort slices by ImagePositionPatient·(row×col), never by filename.
7. **Physical-scale cropping, verified per series.** 130–160 mm physical crop then resize (community standard; discussion/734105). Honor the crop-risk findings: MCL needs the medial soft tissues and ≥75 mm below the joint line; effusion needs the suprapatellar pouch on sagittal; large Baker's cysts extend posteriorly/inferiorly. **Check the crop actually fits each series** — an out-of-range crop silently no-ops. Consider 160 mm or FOV-adaptive cropping given these anatomical constraints.
8. **Architecture:** per-plane encoders (DINOv2 base/large or a radiology-pretrained ViT), 2.5D context (adjacent slices as channels — meniscal tears need the two-slice context by definition), attention/MIL pooling over slices, then cross-plane fusion into 12 sigmoid heads. Keep the sagittal series even when it is T1 — morphology and surface signal still carry meniscus/ACL information.
9. **TTA** (horizontal flip in the canonicalized orientation, small intensity jitters) — cheap +0.002–0.005.

### Lever 4 — Ensembling & selection (expected: +0.005 to +0.01)

10. **5-fold × 2–3 seeds × 2 backbones**, plain sigmoid averaging (proven pattern in the public 0.903 pipeline and the 0.92+ public solutions). Diversity (different backbones, different label-extractor versions, different crop scales) beats more folds of the same model.
11. **Optional per-label stacking** on out-of-fold predictions — but with only 58 clean studies to validate the stacker, keep it shallow (logistic regression per label) or skip it.

### Lever 5 — Validation & shake-up defense (protects your final rank)

12. **Site/scanner-grouped CV**, not random folds: metadata-only probes already showed scanner-grouped generalization is harder (0.598 vs 0.65 macro AUC for a metadata probe; discussion/733517). Your CV must simulate "new site", because the 70% private set will behave like one.
13. **Use the gold 58 honestly**: cross-fitted predictions only, never trained on directly without accounting; the rank-18 writeup is a public cautionary tale of gold58 leakage inflating model *selection* (it did not inflate the Kaggle score).
14. **Respect the 30/70 split.** The overview warns prevalence is not guaranteed to match across train, public, and private sets. Calibrate per-label operating characteristics on your full CV, not on public-LB feedback; treat the 51-entry #1 team's probing as a warning about public-LB overfitting, not as a target to copy. Expect a shake-up; a robust 0.948 private beats a probed 0.951 public.

### Lever 6 — Inference reliability (protects whatever you built)

15. Code-competition hygiene per the public pipeline: DICOM decode fallback chain (pydicom → pylibjpeg → gdcm), periodic `submission.csv` checkpointing every N studies, neutral 0.5 fallback for undecodable studies, and adaptive budget degradation (fewer slices, then fewer ensemble members) against the 9-hour internet-off limit.

---

## 4. What the arithmetic says

| Starting point | Move | Realistic landing |
|---|---|---|
| Public baseline ~0.92 | Clean multi-extractor per-class-weighted label fusion + threshold-correct schema | ~0.93–0.94 |
| ~0.935 | + noisy-label reconciliation loop (image model ⇄ reports) | ~0.94–0.945 |
| ~0.945 | + MRNet/external pretraining, laterality canonicalization, crop fixes | ~0.945–0.955 |
| ~0.95 | + seed/backbone/label-version ensemble diversity + TTA | **0.95–0.96** |
| 0.95+ public | Robust validation, no LB probing | survives the 70% private shake-up |

Every step except the last two rows is **knowledge work**: better extraction of what the reports say, better mapping of report language to the host's thresholds, better use of external expert labels. That is the answer to your question in one sentence:

> **The gap from the public 0.92 pack to 0.95–0.96 is roughly 75–80% a knowledge/supervision gap (label quality, threshold semantics, external expert data) and 20–25% a modeling gap (fusion, ensembling, input geometry) — and essentially 0% a raw architecture gap.**

---

## 5. Caveats

- Leaderboard figures verified 2026-08-19; ranks and scores are moving daily and the public board reflects only 30% of test data.
- Expected-gain numbers are reasoned estimates from public evidence (the documented 0.903 pipeline, public ~0.92 solutions, noisy-label literature), not guarantees; no winning solution writeups exist yet.
- Community techniques cited (crop sizes, DINOv2, fusion patterns) are early-competition practice, not validated winners.

## Sources

- Kaggle leaderboard (pulled 2026-08-19): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard
- Kaggle discussions: 733343 (host annotation criteria) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343 ; 733517 (metadata probe / LB velocity) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517 ; 733652 (external data rules) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652 ; 733965 (LLM API rules) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965 ; 734105 (130 mm physical crop) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105 ; 735304 — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304 ; 735767 — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735767
- "Inside the pipeline that placed 18th…" (0.903, extractor ceiling ~0.83, honest gold58 0.857): https://huggingface.co/blog/bishnoiyash/rsna-competetion
- RSNA press release & challenge page: https://www.rsna.org/news/2026/august/ai-challenge-knee-mri ; https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- Companion document: *RSNA Knee MRI Diagnostic Guide & Anatomical Routing Table* (per-finding planes/sequences/thresholds used for the extraction schema and difficulty ranking)
