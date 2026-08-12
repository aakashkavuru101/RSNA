# Execution Plan — RSNA Knee Abnormality Detection (Kaggle)

**Deadline:** Final submission 22 October 2026 (entry/team-merger 15 October 2026)
**Metric:** Macro-averaged ROC AUC across 12 binary knee-MRI labels
**Constraint:** Kaggle Notebook only, ≤9 h runtime, internet disabled at submission
**Core insight:** This is a weak-supervision competition. Only 58 of 4,407 training studies have expert labels. The other ~4,349 must be labeled by extracting findings from multilingual radiology reports. Label quality is the single largest score lever.

---

## 1. Objective and Success Criteria

| Milestone | Success criterion |
|---|---|
| First real submission | Notebook produces valid `submission.csv` within 9 h on Kaggle |
| Baseline CV | Scanner-grouped 5-fold CV macro AUC ≥ 0.80 (image-only, silver labels) |
| First LB submission | Public LB score that confirms CV-to-LB pipeline correlation |
| Mid-point target | Public LB ≥ 0.87–0.89 (top public baseline range) |
| Final target | Public LB ≥ 0.90+ with conservative ensemble; two diverse final submissions selected |

---

## 2. Phase 0 — Project Setup (Week 1, Days 1–2)

### 2.1 Local repository skeleton

```
rsna-knee/
  AGENTS.md                  # env facts, pinned versions, quotas, do-not-touch paths
  project.yml                # competition metadata
  .env.example               # KAGGLE_USERNAME / KAGGLE_KEY placeholders
  configs/                   # one YAML per experiment
  src/
    dicom_io/                # series grouping, IPP sort, LUT order, photometric fix
    preprocess/              # clip + z-score, resize, slice interpolation -> npy/tar cache
    labels/                  # report parsing, LLM extraction, label assembly
    datasets/                # MONAI CacheDataset/PersistentDataset, shard writers
    models/                  # 2.5D CNN encoder + attention-MIL / BiLSTM slice aggregator
    train.py                 # fp16 AMP loop, fold + seed from config, CSV/JSON logging
    infer.py                 # study-level aggregation, submission.csv writer
  models/                    # per-experiment: weights/, oof/, metrics.json
  notebooks/                 # thin Kaggle wrappers: prep / train / submit
  kaggle/                    # scripts: build wheels dataset, push weights dataset via API
```

### 2.2 Kaggle account setup

- Join competition, accept rules (before Oct 15).
- Create a Kaggle API token (`~/.kaggle/kaggle.json`).
- Verify GPU quota: 30 h/week (P100 or T4×2).

### 2.3 Offline dependency strategy

- Build a wheelhouse dataset: `pip download monai pydicom pylibjpeg SimpleITK timm scikit-multilearn -d ./wheels`
- Upload as Kaggle Dataset → install offline via `pip install --no-index --find-links=...`

---

## 3. Phase 1 — EDA + DICOM Preprocessing Pipeline (Week 1, Days 3–7)

### 3.1 Data EDA (Kaggle notebook, internet on)

- Parse `train.csv` (4,407 rows; use CSV parser, not line counting — reports contain embedded newlines).
- Parse `train_series.csv` — record per-series `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`.
- Count series per study, slices per series, plane distribution, sequence-type distribution.
- Language detection on all 4,407 reports (fastText `lid.176`).
- Compute per-label prevalence on the 58 gold studies.

### 3.2 DICOM reading module (`src/dicom_io/`)

Implement and test against all 4 transfer syntaxes:
1. Group files by `SeriesInstanceUID`.
2. Sort slices by projecting `ImagePositionPatient` onto `cross(row, col)` of `ImageOrientationPatient`; `InstanceNumber` fallback.
3. `apply_modality_lut` → `apply_voi_lut` (in that order).
4. Invert `MONOCHROME1` if needed.
5. Percentile-clip (0.5–99.5%) + per-volume z-score.
6. Interpolate to fixed slice count (24–96 per series).
7. Cache as float16 `.npy` or tar shards (respects Kaggle 1000-file dataset limit).

### 3.3 Validation harness

- 5-fold GroupKFold split by scanner/site group (from DICOM metadata) × iterative multilabel stratification on silver labels.
- Save the split as a fixed artifact — every experiment uses the same folds.

### 3.4 Dry-run submission

- Trivial all-0.5 `submission.csv` to validate the notebook → submission path end-to-end.

**Exit criteria:** DICOM pipeline decodes all syntaxes; folds saved; dry-run submission accepted.

---

## 4. Phase 2 — Silver Label Extraction Pipeline (Weeks 2–3)

This is the critical path. Label quality dominates everything else.

### 4.1 LLM-based extraction (primary)

- **Model:** GPT-4o-mini (or equivalent cheap commercial API — host ruling permits this) as primary; open-weight Qwen3-14B/32B on Kaggle GPU as cross-check.
- **Prompt design:**
  - System prompt contains the official positivity thresholds (ACL >50% fibers; meniscal signal reaching surface on ≥2 images; OA ≥1 cm high-grade cartilage loss; borderline → negative).
  - Few-shot: 3–5 examples covering English, Spanish, and one non-Latin-script report.
  - Output: strict JSON, one field per finding, 5-way value: `{present, absent, uncertain, not_addressed, laterality_ambiguous}` + evidence span.
- **Coverage:** All 4,407 reports, all languages.

### 4.2 Rule-based fallback (secondary voter)

- Multilingual CheXpert-style labeler: per-language positive/negative/uncertain phrase files, NegEx-style triggers.
- OA-consequence vocabulary: osteophytes, joint-space narrowing, chondral loss, "tricompartmental".
- Purpose: deterministic audit trail, precision backup on fracture/Baker's.

### 4.3 Label assembly

- Map `not_addressed` → soft label 0.5 (never 0). Mask in loss (don't zero-fill).
- Map `uncertain` → soft label 0.5 (or 0.6–0.7 for effusion-type findings where silence is weakly negative).
- Per-finding targeted imputation only where evidence supports it (e.g., effusion → synovitis lifting).
- Output: 4,407 × 12 matrix of soft silver labels in [0,1] + per-cell confidence.
- The 58 gold studies are NEVER used for training — they are the only honest local anchor.

### 4.4 Validation of label pipeline

- Evaluate every label variant (LLM-only, LLM+rules, different soft-label policies) on the 58 gold studies.
- Report macro AUC per variant. Treat differences < ±0.02 as unmeasurable noise at n=58.
- Iterate until label pipeline macro AUC ≥ 0.87 on gold.

**Exit criteria:** Silver label matrix saved; label pipeline scores ≥ 0.87 macro AUC on the 58 gold studies.

---

## 5. Phase 3 — Image Classification Model v1 (Weeks 3–4)

### 5.1 Input design

- **Plane routing:** Use `train_series.csv` flags. Train separate models (or shared backbone with plane embeddings) for sagittal, coronal, axial.
- **Sequence routing:** Prioritize fluid-sensitive fat-suppressed (T2-FS/PD-FS) series — these are the meniscal/ligament workhorses.
- **2.5D input:** 3 adjacent slices stacked as 3 channels (e.g., mid-1, mid, mid+1). Resize to 224×224. Normalize with ImageNet stats.
- **Slice selection:** Central slices around the knee joint. Later upgrade to learned slice alignment (MLFANet-SA style).

### 5.2 Model architecture

- **Backbone:** ConvNeXt-Small (ImageNet-1k pretrained). Fallback: EfficientNetV2-S.
- **Slice encoder:** 2.5D — modify first conv to accept 3-channel input from stacked adjacent slices.
- **Slice aggregation:** BiLSTM over per-slice features (proven by RSNA 2024 lumbar winner) or gated attention-MIL.
- **Heads:** 12 independent binary heads (sigmoid). Treat labels as independent tasks — handles severe per-class imbalance.
- **Plane fusion:** Per-plane models + logistic regression stacking (MRNet template) as v1; CoPAS-style cross-plane attention as v2.

### 5.3 Training recipe

- **Loss:** Asymmetric loss (γ−=4, γ+=1, clip=0.05) or BCE with label smoothing (ε≈0.05–0.1). Mask "not addressed" cells (soft label 0.5 → exclude from loss).
- **Sampling:** Balanced study-level sampling (positive/negative studies equally per class group). Sampling > loss weights.
- **Optimizer:** AdamW + cosine annealing with warm-up.
- **Augmentation:** Label-aware horizontal flip (swap Medial↔Lateral Meniscus, Medial↔Lateral OA), slight rotation, brightness/contrast jitter. No heavy augmentation.
- **Precision:** fp16 with `torch.amp.GradScaler('cuda')` (P100/T4 have no bf16).
- **Validation:** Scanner-grouped 5-fold CV (from Phase 1.3). Per-fold OOF predictions saved.

### 5.4 First real submission

- Train on 4 folds, validate on 5th. Repeat for all 5 folds.
- Ensemble 5 fold models by averaging.
- Inference notebook: load fold models, predict on test set, average, write `submission.csv`.
- Verify notebook completes within 9 h on Kaggle GPU.

**Exit criteria:** Scanner-grouped CV macro AUC ≥ 0.80; first real LB submission accepted and scored.

---

## 6. Phase 4 — Validation Hardening and Iteration (Weeks 5–7)

### 6.1 CV-to-LB correlation

- Track CV score vs public LB score for each submission.
- If CV >> LB: investigate domain shift, label noise, or pipeline bugs.
- If LB >> CV: check for leakage in CV split.

### 6.2 Architecture iteration (one at a time, measured on grouped CV)

| Priority | Experiment | Evidence |
|---|---|---|
| 1 | Add coronal + axial planes (if starting sagittal-only) | CoPAS plane-preference matrix |
| 2 | Learned slice alignment (Top-K pooling or attention over slices) | MLFANet-SA: ACL AUC 0.981 on MRNet |
| 3 | ROI localization stage (small U-Net or keypoint model → cropped classification) | RSNA 2022–2025 winners; 87 masks sufficed in 2022 |
| 4 | Cross-plane attention fusion (CoPAS-style) | CoPAS: avg AUC 0.812 internal, 0.72 external |
| 5 | 3D-ResNet with MedicalNet init | Ablation candidate; expected worse than 2.5D |

### 6.3 Label pipeline v2

- Refine LLM prompts based on error analysis on gold studies.
- Targeted gap-filling for specific findings (never blanket imputation).
- Soft/graded targets if the evidence holds (+0.056 macro AUC reported).

### 6.4 TTA (test-time augmentation)

- Label-aware horizontal flip TTA only. Validate on OOF — recent evidence shows TTA can *hurt* medical classification.
- Skip TTA on efficiency-track submission.

**Exit criteria:** Grouped CV ≥ 0.85; public LB correlates with CV direction; at least 2 architecture upgrades validated.

---

## 7. Phase 5 — Ensembling and Final Submission (Weeks 8–9)

### 7.1 Ensemble construction

- Ensemble across: folds × seeds × backbones × plane combinations.
- OOF-weighted stacking: weight each model by its OOF macro AUC.
- Per-class ensemble weighting (some models are better on ACL, others on OA).

### 7.2 Final validation

- Report final grouped CV macro AUC + per-class AUC.
- Report gold-study AUC as a sanity check (but not used for model selection — n=58 is too small).
- Dry-run full notebook end-to-end; confirm ≤ 9 h.

### 7.3 Submission selection (up to 2 final submissions)

1. **Submission A — Best-CV ensemble:** The full ensemble with best scanner-grouped CV.
2. **Submission B — Lean/diverse:** A different architecture or plane combination that decorrelates from A, OR a distilled efficient model for the efficiency track.

Select on grouped CV + gold, not public LB decimals. Expect shake-up (public = 30% of test, prevalence unmatched).

**Exit criteria:** Two final submissions selected and submitted before Oct 22 deadline.

---

## 8. Phase 6 — Efficiency Track (Weeks 9–10, Optional/Parallel)

Only if the main track is stable and time permits.

- **Student model:** ELNet-class (~0.2M params, trained from scratch on single plane).
- **Distillation:** From the main-track ensemble's soft OOF predictions.
- **Acceleration:** PyTorch fp16/AMP + `torch.compile`. No TensorRT INT8 (147× slower risk on T4).
- **Inference:** Minimal series routing, cached preprocessing, zero TTA, single SWA/EMA model.
- **Target:** Runtime well under 9 h with AUC above the all-0.5 benchmark.

---

## 9. Timeline Summary

| Week | Dates (2026) | Phase | GPU-h budget |
|---|---|---|---|
| W1 | Aug 10–16 | Phase 0 + Phase 1 | ~10 |
| W2 | Aug 17–23 | Phase 2 (labels v1) | ~15 |
| W3 | Aug 24–30 | Phase 2 (labels validation) + Phase 3 start | ~30 |
| W4 | Aug 31–Sep 6 | Phase 3 (model v1 + first submission) | ~30 |
| W5 | Sep 7–13 | Phase 4 (iteration: planes, slice alignment) | ~30 |
| W6 | Sep 14–20 | Phase 4 (ROI localization, fusion) | ~30 |
| W7 | Sep 21–27 | Phase 4 (labels v2, retrain) | ~30 |
| W8 | Sep 28–Oct 4 | Phase 5 (ensembling, CV hardening) | ~30 |
| W9 | Oct 5–11 | Phase 5 + Phase 6 (efficiency candidate) | ~35 |
| W10 | Oct 12–22 | Buffer + final selection + submission | ~25 |

Total ≈ 265 GPU-hours planned vs ~300 available. Keep ~35 h reserve.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Silver-label noise caps the model | High | High | Multi-LLM ensemble; validate on 58 gold; mask "not addressed"; soft targets |
| Site/scanner shift (16 institutions) | High | High | Scanner-grouped CV as primary metric; per-volume z-score; intensity augmentation |
| Public LB shake-up (30% of test, prevalence mismatch) | Medium-High | High | Select finals on grouped CV, not LB; two diverse submissions; conservative ensembling |
| Horizontal-flip laterality trap | Medium | High | Label-aware flip (swap medial↔lateral pairs); verify against ImageOrientationPatient |
| Compute budget overrun (30 GPU-h/week) | Medium | High | Cache preprocessing once; iterate on small models; weekly GPU-h ledger |
| Non-Latin report long tail (Greek 7.3%, Cyrillic 5%) | High | Medium | Multilingual LLM extractor; script-detection audit; no regex-only fallback |
| Notebook fails or exceeds 9 h at deadline | Medium | High | Weekly dry-run from W8; per-study try/except with fallback; pin Docker environment |
| External-data ruling pending (MRNet/OAI gated) | Medium | Medium | Treat as unusable until host rules; clone architectures, not data |

---

## 11. Key Decision Points (No Code Yet)

| Decision | Recommendation | Rationale |
|---|---|---|
| LLM for label extraction | GPT-4o-mini (commercial API) as primary | Host explicitly permits; cheapest path to highest label quality; ~$1 total for 4,407 reports |
| Image backbone | ConvNeXt-Small (ImageNet pretrained) | Proven by RSNA 2024 lumbar winner; small > large at this data scale |
| Slice aggregation | BiLSTM (v1) → attention-MIL (v2) | BiLSTM proven; attention-MIL adds +0.02 per RSNA 2024 |
| Plane strategy | All 3 planes, separate models + LR stacking (v1) | MRNet template; CoPAS-style attention as upgrade |
| Validation | Scanner-grouped 5-fold GroupKFold | Only honest metric given 16-site domain shift |
| First submission target | Week 4 | Early submission validates the full pipeline and gives CV-to-LB calibration |

---

*This plan supersedes `baseline_plan.md`. It incorporates all findings from the Kimi research dossier (58 gold studies, train-only reports, label-quality dominance, scanner-grouped CV, label-aware flip, CoPAS template, efficiency track).*
