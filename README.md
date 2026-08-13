# RSNA Knee Abnormality Detection

Competition code, research notes, and reproducibility materials for the 2026
[RSNA Knee Abnormality Detection challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

The project targets twelve study-level knee MRI findings with a multilingual
report-derived weak-supervision pipeline and an image-only inference model.
The current candidate uses DICOM geometry-aware preprocessing, scanner-isolated
validation, six acquisition slots, and a DINOv2-Small attention-MIL classifier.

## Repository contents

- `src/`: reusable DICOM, preprocessing, label, fold, model, training, and inference code
- `notebooks/`: Kaggle-oriented EDA, label extraction, preprocessing, training, and submission entry points
- `configs/`: experiment configuration
- `kaggle/clean_dino_train/`: offline Kaggle GPU trainer/inference kernel, promoted
  after a bounded smoke run and designed to refuse silent fallback predictions
- `kaggle/oof_diagnostics/`: private checkpoint-only reconstruction of study-level
  scanner-isolated OOF predictions and target/scanner/sequence-coverage gap reports
- `kaggle/localized_dino_train/`: contiguous 2.5D slice windows plus target-specific
  DINO patch attention, guarded against the exact scored 0.871 OOF ensemble
- `Kimi_Agent_RSNA Knee Detection Papers/`: research dossier and supporting analysis
- `EXECUTION_PLAN.md`: phased implementation and validation plan
- `AGENTS.md`: pinned environment facts, competition constraints, and safety rules

## Data and credentials

Competition data, extracted labels, checkpoints, logs, model weights, OOF
predictions, virtual environments, and credentials are intentionally excluded
from Git. Obtain the data directly through Kaggle after accepting the
competition rules. Copy `.env.example` to `.env` for local credentials; never
commit `.env`.

The 58 expert-labeled studies are reserved for evaluation and must never be
used for training. See `AGENTS.md` for the complete operational constraints.

## Clean weak supervision

The DINOv2 candidate consumes `llm_labels_v4_blend.csv` from the competition-public
Kaggle dataset `stevenleehans/rsna-knee-llm-report-labels`. The kernel verifies exact
study coverage, identifies the 58 expert-labeled rows from `train.csv`, and excludes
all of them from optimizer batches. Gold AUC is logged only as a monitor; checkpoint
selection uses scanner-isolated silver validation. The attached DINOv2-Small model is
loaded locally with internet disabled. Neither the label table nor pretrained weights
are redistributed in this repository.

The second candidate preserves the scored 224-pixel ensemble and trains a complementary
336-pixel five-fold family. It chooses one resolution-blend weight using pooled
scanner-isolated OOF predictions only. The kernel refuses to write `submission.csv`
unless that blend improves the validated 224-pixel baseline by at least 0.001 macro AUC.

The localized experiment leaves both scored families untouched. It replaces the
interleaved slice channels with true adjacent three-slice windows, applies mild
scanner-frequency balancing, and learns target-specific patch and acquisition-slot
attention. Coarse per-target blending is selected only from scanner-isolated OOF and
must improve the exact 0.871 scored blend before a candidate file is emitted.

## License

Code and project-authored documentation are licensed under
[CC BY-NC 4.0](LICENSE), matching the competition's stated winner-license
requirement. Competition data and third-party materials retain their original
licenses and are not relicensed by this repository.
