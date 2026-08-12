# Clean DINOv2 smoke validation

Kaggle kernel: `aakashkavuru/rsna-knee-clean-dinov2-smoke`, version 1

- Runtime: 608.2 seconds (0.17 hours) on one Tesla T4
- Supervision: 609 train / 154 scanner-isolated validation studies
- Gold handling: 58 evaluation studies, **0 used for training**
- DICOM decode: 3,985 selected series, 0 failures
- Scanner fingerprints: 49 total; 40 train / 9 validation
- Epoch 1: silver AUC 0.6778, gold monitor AUC 0.6467
- Epoch 2: silver AUC 0.6861, gold monitor AUC 0.6604
- Checkpoint SHA-256:
  `04433d0a9036d017ffa818e27dc465c26c00f1483487d707bc4ceffc9230f1b8`

Gold AUC was monitoring-only. Checkpoint selection used scanner-isolated silver AUC.
No competition submission was made from this smoke run.

## Full five-fold result

Kaggle kernel: `aakashkavuru/rsna-knee-clean-dinov2-full`, version 1

- Runtime: 6,159.8 seconds (1.71 hours) on one Tesla T4
- DICOM decode: 21,334 selected series, 0 failures
- Gold handling: 58 evaluation studies, **0 used for training**
- Best scanner-isolated fold AUCs: 0.8055, 0.7929, 0.7696, 0.7877, 0.8046
- Five-model gold monitor AUC: 0.8201
- Public leaderboard: **0.857** (submission `55450279`)
