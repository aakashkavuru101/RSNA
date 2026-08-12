"""
EDA script for RSNA Knee Abnormality Detection.

Run on Kaggle with internet ON (for language detection model download).
Produces: EDA report, scanner metadata cache, folds.csv.

Usage (Kaggle notebook):
    exec(open("notebooks/01_eda.py").read())
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# ── 1. Parse train.csv ──────────────────────────────────────────────
logger.info("=" * 60)
logger.info("1. Parsing train.csv")
train_csv = pd.read_csv(DATA_ROOT / "train.csv")
logger.info(f"Shape: {train_csv.shape}")
logger.info(f"Columns: {list(train_csv.columns)}")
logger.info(f"Unique studies: {train_csv['StudyInstanceUID'].nunique()}")

# Check which studies have gold labels
has_labels = train_csv[LABEL_COLS[0]].notna()
logger.info(f"Studies with gold labels: {has_labels.sum()}")
logger.info(f"Studies without labels (report-only): {(~has_labels).sum()}")

# Gold label prevalence
if has_labels.any():
    gold = train_csv[has_labels]
    logger.info("\nGold label prevalence (58 studies):")
    for col in LABEL_COLS:
        prevalence = gold[col].mean()
        count = gold[col].sum()
        logger.info(f"  {col}: {prevalence:.3f} ({int(count)}/{len(gold)})")

# PatientSex distribution
logger.info(f"\nPatientSex distribution:\n{train_csv['PatientSex'].value_counts(dropna=False)}")

# Report lengths
report_lengths = train_csv["Report"].fillna("").str.len()
logger.info(f"\nReport length stats:")
logger.info(f"  Mean: {report_lengths.mean():.0f} chars")
logger.info(f"  Median: {report_lengths.median():.0f} chars")
logger.info(f"  Max: {report_lengths.max():.0f} chars")
logger.info(f"  Empty reports: {(report_lengths == 0).sum()}")

# ── 2. Parse train_series.csv ───────────────────────────────────────
logger.info("\n" + "=" * 60)
logger.info("2. Parsing train_series.csv")
train_series = pd.read_csv(DATA_ROOT / "train_series.csv")
logger.info(f"Shape: {train_series.shape}")
logger.info(f"Columns: {list(train_series.columns)}")

logger.info(f"\nAnatomical_Plane distribution:\n{train_series['Anatomical_Plane'].value_counts()}")
logger.info(f"\nFluid_Sensitive distribution:\n{train_series['Fluid_Sensitive'].value_counts()}")
logger.info(f"\nFat_Suppression distribution:\n{train_series['Fat_Suppression'].value_counts()}")

# Series per study
series_per_study = train_series.groupby("StudyInstanceUID").size()
logger.info(f"\nSeries per study:")
logger.info(f"  Mean: {series_per_study.mean():.1f}")
logger.info(f"  Median: {series_per_study.median():.0f}")
logger.info(f"  Min: {series_per_study.min()}")
logger.info(f"  Max: {series_per_study.max()}")

# ── 3. Language detection ───────────────────────────────────────────
logger.info("\n" + "=" * 60)
logger.info("3. Language detection on reports")
try:
    import fasttext
    # Download fastText language identification model
    import urllib.request
    model_path = OUTPUT_DIR / "lid.176.bin"
    if not model_path.exists():
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            model_path,
        )
    lid_model = fasttext.load_model(str(model_path))

    languages = []
    for report in train_csv["Report"].fillna(""):
        # Use first 500 chars for speed
        pred = lid_model.predict(report[:500].replace("\n", " "), k=1)
        lang = pred[0][0].replace("__label__", "")
        languages.append(lang)

    train_csv["language"] = languages
    lang_counts = Counter(languages)
    logger.info(f"Language distribution:")
    for lang, count in lang_counts.most_common(20):
        logger.info(f"  {lang}: {count} ({count/len(languages)*100:.1f}%)")

except ImportError:
    logger.warning("fasttext not available, skipping language detection")
    train_csv["language"] = "unknown"

# ── 4. DICOM metadata scan (scanner groups) ─────────────────────────
logger.info("\n" + "=" * 60)
logger.info("4. DICOM metadata scan (scanner groups)")

import pydicom

scanner_records = []
train_series_dir = DATA_ROOT / "train_series"

for _, row in train_csv.iterrows():
    study_uid = row["StudyInstanceUID"]
    study_dir = train_series_dir / study_uid

    manufacturer = "unknown"
    model_name = "unknown"
    num_series = 0
    total_slices = 0

    if study_dir.exists():
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            num_series += 1
            dcm_files = list(series_dir.glob("*.dcm"))
            total_slices += len(dcm_files)

            if dcm_files and manufacturer == "unknown":
                try:
                    ds = pydicom.dcmread(
                        str(dcm_files[0]), stop_before_pixels=True
                    )
                    manufacturer = getattr(ds, "Manufacturer", "unknown")
                    model_name = getattr(ds, "ManufacturerModelName", "unknown")
                except Exception:
                    pass

    scanner_records.append({
        "StudyInstanceUID": study_uid,
        "Manufacturer": manufacturer,
        "ManufacturerModelName": model_name,
        "num_series": num_series,
        "total_slices": total_slices,
    })

scanner_df = pd.DataFrame(scanner_records)
logger.info(f"Scanner distribution:")
scanner_counts = scanner_df.groupby(["Manufacturer", "ManufacturerModelName"]).size()
for (mfr, model), count in scanner_counts.sort_values(ascending=False).items():
    logger.info(f"  {mfr} / {model}: {count} studies")

logger.info(f"\nSlices per study:")
logger.info(f"  Mean: {scanner_df['total_slices'].mean():.0f}")
logger.info(f"  Median: {scanner_df['total_slices'].median():.0f}")

# ── 5. Create scanner-grouped folds ─────────────────────────────────
logger.info("\n" + "=" * 60)
logger.info("5. Creating scanner-grouped 5-fold split")

scanner_df["scanner_group"] = (
    scanner_df["Manufacturer"].fillna("unknown")
    + "_"
    + scanner_df["ManufacturerModelName"].fillna("unknown")
)

from sklearn.model_selection import GroupKFold

groups = scanner_df["scanner_group"].values
gkf = GroupKFold(n_splits=5)
fold_assignments = np.full(len(scanner_df), -1, dtype=int)

X = np.arange(len(scanner_df)).reshape(-1, 1)
for fold_idx, (_, val_idx) in enumerate(gkf.split(X, groups=groups)):
    fold_assignments[val_idx] = fold_idx

folds_df = pd.DataFrame({
    "StudyInstanceUID": scanner_df["StudyInstanceUID"],
    "fold": fold_assignments,
    "scanner_group": scanner_df["scanner_group"],
})

logger.info(f"Fold sizes:")
for fold in range(5):
    count = (fold_assignments == fold).sum()
    logger.info(f"  Fold {fold}: {count} studies")

# ── 6. Save outputs ─────────────────────────────────────────────────
logger.info("\n" + "=" * 60)
logger.info("6. Saving outputs")

train_csv.to_csv(OUTPUT_DIR / "train_parsed.csv", index=False)
scanner_df.to_csv(OUTPUT_DIR / "scanner_metadata.csv", index=False)
folds_df.to_csv(OUTPUT_DIR / "folds.csv", index=False)

# Save gold labels separately
if has_labels.any():
    gold_labels = train_csv[has_labels][["StudyInstanceUID"] + LABEL_COLS]
    gold_labels.to_csv(OUTPUT_DIR / "gold_labels.csv", index=False)
    logger.info(f"Gold labels saved: {len(gold_labels)} studies")

# Save EDA summary
eda_summary = {
    "total_studies": len(train_csv),
    "gold_labeled": int(has_labels.sum()),
    "silver_labeled": int((~has_labels).sum()),
    "total_series": len(train_series),
    "languages": dict(lang_counts.most_common(20)) if "lang_counts" in dir() else {},
    "scanner_groups": len(scanner_df["scanner_group"].unique()),
    "fold_sizes": {f"fold_{i}": int((fold_assignments == i).sum()) for i in range(5)},
}
with open(OUTPUT_DIR / "eda_summary.json", "w") as f:
    json.dump(eda_summary, f, indent=2)

logger.info("\nEDA complete. Outputs saved to /kaggle/working/")
logger.info("Next: run 02_label_extraction.py to build silver labels")
