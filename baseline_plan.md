# RSNA Knee Abnormality Detection — Baseline Plan

> Seed plan. Pending synthesis of external research documents before building the concrete implementation plan.

## Competition Recap

- **Task:** Per-study probability for 12 knee MRI abnormalities.
- **Labels:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
- **Metric:** Macro-averaged ROC AUC across the 12 labels.
- **Data:** >5,000 knee MRI exams (DICOM) + multilingual radiology reports from 16 sites.
- **Submission:** Kaggle Notebook only, ≤9 hours runtime, no internet during scoring, output `submission.csv`.
- **Deadlines:** Entry/merge 15 Oct 2026; final submission 22 Oct 2026.

---

## Baseline Choice: 2.5D ConvNeXt-Tiny on Stacked Knee-MRI Slices

**Model:** `convnext_tiny` pretrained on ImageNet-1k (from `torchvision` or `timm`).

**Why this one:**
- Small enough to train and infer inside Kaggle's 9-hour GPU limit.
- Better inductive bias for medical imaging than ResNet, and easier to fine-tune than Vision Transformers at this stage.
- ImageNet pretraining provides a strong starting feature extractor; the dataset is large enough to adapt it.
- Accepts 3-channel input by stacking 3 adjacent MRI slices, turning a 3D volume problem into a familiar 2D classification problem.

---

## End-to-End Design

### 1. Input Representation

For every MRI study:
- Load all DICOM series.
- Start with the **sagittal** plane (most informative for ACL, meniscus, cartilage).
- For each sagittal series, select **3 adjacent slices around the center of the knee joint** (e.g., indices `mid-1`, `mid`, `mid+1`).
- Stack the 3 grayscale slices into a single 3-channel tensor `(3, H, W)`.
- Resize to `224 × 224`.
- Normalize with ImageNet statistics.

This yields one training sample per series, labeled with the study's 12 binary targets.

### 2. Model Architecture

- Backbone: ConvNeXt-Tiny.
- Replace the final classifier layer with `Linear(in_features, 12)`.
- Output activation: **sigmoid** (independent binary probabilities).
- Loss: **binary cross-entropy** averaged over the 12 labels.

### 3. Training Setup

- **Validation:** 5-fold stratified group split by `StudyInstanceUID`. Stratify using a combined abnormality indicator to balance label distribution across folds.
- **Optimizer:** AdamW.
- **Scheduler:** Cosine annealing with warm-up.
- **Augmentation:** Light medical-safe transforms — random horizontal flip, slight rotation, brightness/contrast jitter. No heavy augmentation at baseline.
- **Epochs:** ~10–15, depending on convergence.
- **Batch size:** 16–32 on a single T4 GPU.

### 4. Inference and Submission

- For each test study, generate predictions for all sagittal series / slice groups.
- **Average predictions** across slices and series to produce one probability vector per study.
- Write `submission.csv` with the required 12 columns.

---

## Kaggle Constraint Fit

| Constraint | How the design satisfies it |
|---|---|
| ≤ 9 hours GPU runtime | ConvNeXt-Tiny trains in a few hours on this data size. |
| No internet during scoring | `torchvision`/pretrained weights pre-downloaded and loaded from `/kaggle/input`. |
| Notebook-only submission | Single training + inference notebook outputs `submission.csv`. |
| Macro ROC AUC metric | Sigmoid outputs are calibrated probabilities; BCE is a good proxy. |

---

## Expected Outcome

This should produce a **stable bronze-level baseline** — reproducible and not top-tier, but a solid foundation. Public notebooks suggest similar image-only approaches score in the high-0.70s to low-0.80s macro AUC range, leaving clear headroom for multimodal and 3D improvements.

---

## Natural Next Steps

After the baseline is running:

1. Add **axial and coronal planes** and average predictions across planes.
2. Replace single-slice stacking with a small **3D CNN** or video-model backbone.
3. Add the **radiology report branch** using a multilingual sentence transformer or small LLM, then fuse with image features.
4. Experiment with stronger backbones: EfficientNetV2-S, DINOv2, or a medical pretrained encoder if available.

---

## Status

- [x] Seed baseline plan drafted.
- [ ] Incorporate findings from Kimi swarm research documents.
- [ ] Finalize concrete implementation plan.
- [ ] Write Kaggle notebook code (pending explicit go-ahead).
