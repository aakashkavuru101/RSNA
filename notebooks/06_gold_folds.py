"""
Create the gold-label artifacts (one-time, immutable):

- input/gold_labels.csv — the 58 expert-labeled studies from train.csv
- input/gold_folds.csv  — 5-fold split of the 58 gold studies for the
  cross-fit protocol in GOLD_INTEGRATION_PLAN.md §2

gold_folds.csv is NEVER regenerated once it exists. Any candidate that trains
on gold may only be evaluated on gold folds it did not train on.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "input" / "train.csv"
GOLD_LABELS_CSV = ROOT / "input" / "gold_labels.csv"
GOLD_FOLDS_CSV = ROOT / "input" / "gold_folds.csv"

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
N_FOLDS = 5
SEED = 20260819


def extract_gold_labels() -> pd.DataFrame:
    # Reports contain embedded newlines — must use a real CSV parser.
    df = pd.read_csv(TRAIN_CSV)
    gold = df[df[LABELS].notna().any(axis=1)][["StudyInstanceUID"] + LABELS].copy()
    gold[LABELS] = gold[LABELS].astype(float)
    if len(gold) != 58:
        raise RuntimeError(f"expected 58 gold studies, found {len(gold)}")
    return gold


def assign_folds(gold: pd.DataFrame) -> pd.Series:
    """Greedy prevalence-balanced assignment: rarest-positive-first, each study
    goes to the fold with the fewest current positives of its rarest label."""
    rng = np.random.default_rng(SEED)
    y = gold[LABELS].fillna(0.0).to_numpy()
    rarity = y.sum(axis=0)  # per-label positive counts
    order = sorted(range(len(gold)), key=lambda i: (y[i] * rarity).min() if y[i].any() else 1e9)
    fold_pos = np.zeros((N_FOLDS, len(LABELS)))
    fold_size = np.zeros(N_FOLDS)
    assignment = np.full(len(gold), -1)
    for i in order:
        labels_i = np.where(y[i] > 0)[0]
        if len(labels_i):
            # score each fold by positives of this study's labels, tie-break by size
            score = fold_pos[:, labels_i].sum(axis=1) * 100 + fold_size
        else:
            score = fold_size
        ties = np.flatnonzero(score == score.min())
        fold = ties[rng.integers(len(ties))]
        assignment[i] = fold
        fold_size[fold] += 1
        fold_pos[fold, labels_i] += 1
    return pd.Series(assignment, index=gold.index, name="gold_fold", dtype=int)


def main() -> None:
    gold = extract_gold_labels()
    if not GOLD_LABELS_CSV.exists():
        gold.to_csv(GOLD_LABELS_CSV, index=False)
        print(f"wrote {GOLD_LABELS_CSV} ({len(gold)} studies)")
    else:
        existing = pd.read_csv(GOLD_LABELS_CSV)
        if not existing.equals(gold):
            raise RuntimeError(f"{GOLD_LABELS_CSV} exists and differs — refusing to modify")

    if GOLD_FOLDS_CSV.exists():
        print(f"{GOLD_FOLDS_CSV} already exists — immutable, leaving untouched")
        return
    folds = assign_folds(gold)
    out = gold[["StudyInstanceUID"]].copy()
    out["gold_fold"] = folds
    out.to_csv(GOLD_FOLDS_CSV, index=False)
    merged = out.merge(gold, on="StudyInstanceUID")
    print(f"wrote {GOLD_FOLDS_CSV}")
    print("per-fold sizes:", merged.groupby("gold_fold").size().to_dict())
    for label in LABELS:
        pos = merged.groupby("gold_fold")[label].sum().astype(int).tolist()
        print(f"  {label}: positives per fold {pos}")


if __name__ == "__main__":
    main()
