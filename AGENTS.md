# AGENTS.md — RSNA Knee Abnormality Detection

## Environment Facts

| Item | Value |
|---|---|
| Kaggle GPU options | 1× Tesla P100 (16 GB) or 2× Tesla T4; 4 CPU cores, 29 GB RAM |
| GPU quota | 30 h/week; max 2 concurrent batch GPU sessions |
| Session runtime | 12 h CPU/GPU notebooks; 9 h TPU |
| Submission cap | ≤ 9 h CPU or GPU notebook; internet OFF during scoring |
| Auto-saved output | 20 GB in /kaggle/working |
| Dataset file limit | 1000 files per user dataset → ship tar/zip archives |
| Interactive idle timeout | ~20 min; long runs need Save & Run All |
| Docker image | Updated ~every 2 weeks; pin "original environment" in Session options |

## Pinned Versions

| Package | Version | Note |
|---|---|---|
| SimpleITK | 2.3.1 | **Pin exactly.** 2.4.0 changed DICOM series direction handling (Z-sign flip). Silent volume flip between train and inference if mismatched. |
| pydicom | latest stable | Use `pydicom.pixels.apply_modality_lut` and `apply_voi_lut` |
| pylibjpeg | latest stable | Required for JPEG Lossless and JPEG 2000 transfer syntaxes |
| torch | Kaggle default | fp16 AMP via `torch.amp.GradScaler('cuda')`; no bf16 on P100/T4 |
| timm | latest stable | For ConvNeXt-Small and EfficientNetV2-S backbones |
| monai | 1.4.0 | CacheDataset/PersistentDataset for preprocessing cache |
| scikit-multilearn | latest | Iterative multilabel stratification for CV splits |

## Do-Not-Touch Paths

- `/kaggle/input/rsna-knee-abnormality-detection/` — raw competition data (read-only)
- `input/gold_labels.csv` — 58 gold studies. May be used for training/selection ONLY under the cross-fit protocol in `GOLD_INTEGRATION_PLAN.md` §2: never evaluate a candidate on gold folds it trained on; every artifact records `gold_usage: none | crossfit | full`. The file itself is read-only.
- `input/gold_folds.csv` — immutable 5-fold split of the 58 gold studies; NEVER regenerate once created
- `models/*/oof/` — OOF prediction artifacts; NEVER overwrite
- `input/silver_labels_v4.csv` + `input/silver_labels_v4_fusion_audit.json` — fused silver labels from `notebooks/08_label_fusion.py`; regenerate only via that script with `--force`; the audit records the per-label source map and `gold_usage`
- `input/extraction_soft_v3_qwen_*` / `input/silver_labels_soft_v3_qwen_*.csv` — open-weight (Qwen) extraction namespace; one pinned model per namespace, same contract as other `extraction_soft_v3_<slug>_*` artifacts

## Experimental Line (2026-08-20): labels + gold-at-backbone

- `kaggle/qwen_label_extract/` — dev-time extraction kernel (open-weight Qwen via vLLM). **Internet ON is allowed here only** — it is label infrastructure, never produces `submission.csv`. Submission notebooks remain internet-off.
- `kaggle/path20_gold_backbone_train/` — gold-weighted full-backbone fine-tuning under the GOLD_INTEGRATION_PLAN.md §2 cross-fit protocol. Checkpoint contract extension: `gold_training_count` records the true count (0 for gold-free, 46 for cross-fit folds, 58 for full-gold deployment); validators must accept these values instead of asserting 0.

## Offline Install Mechanism

```bash
# Local (or internet-on notebook): build wheelhouse
pip download monai==1.4.0 pydicom pylibjpeg SimpleITK==2.3.1 timm scikit-multilearn -d ./wheels

# Upload ./wheels as a Kaggle Dataset, then in the offline submission notebook:
pip install --no-index --find-links=/kaggle/input/my-wheels/wheels monai pydicom pylibjpeg SimpleITK timm scikit-multilearn
```

## Competition Constraints

- Internet access disabled during submission scoring.
- Output file must be named `submission.csv`.
- 5 submissions per day; up to 2 final submissions selected.
- Maximum team size: 5.
- Winner license: CC-BY-NC 4.0 (code and weights).
- Commercial LLM APIs permitted for report-label extraction (host ruling, 2026-08-09).

## Key Warnings

1. **Horizontal flip swaps medial ↔ lateral labels.** Must swap Medial Meniscus ↔ Lateral Meniscus and Medial OA ↔ Lateral OA on flip, in both augmentation and TTA.
2. **Empty label cells are NOT zeros.** 4,349 studies have empty label columns — treat as missing, mask in loss.
3. **Reports contain embedded newlines.** `train.csv` is 58,556 lines but only 4,407 rows. Use CSV parser.
4. **Never trust InstanceNumber alone.** Sort slices by IPP · cross(row, col) of IOP.
5. **Scanner-grouped CV is mandatory.** Random-fold CV is optimistically biased by ~0.05 AUC due to site memorization.
