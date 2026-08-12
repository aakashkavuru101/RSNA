# RSNA Knee Abnormality Detection

Competition code, research notes, and reproducibility materials for the 2026
[RSNA Knee Abnormality Detection challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

The project targets twelve study-level knee MRI findings with a multilingual
report-derived weak-supervision pipeline and an image-only inference model.
The current baseline uses DICOM geometry-aware preprocessing, scanner-grouped
validation, and a 2.5D ConvNeXt-Small/BiLSTM classifier.

## Repository contents

- `src/`: reusable DICOM, preprocessing, label, fold, model, training, and inference code
- `notebooks/`: Kaggle-oriented EDA, label extraction, preprocessing, training, and submission entry points
- `configs/`: experiment configuration
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

## License

Code and project-authored documentation are licensed under
[CC BY-NC 4.0](LICENSE), matching the competition's stated winner-license
requirement. Competition data and third-party materials retain their original
licenses and are not relicensed by this repository.
