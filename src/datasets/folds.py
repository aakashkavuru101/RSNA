"""
Scanner-grouped cross-validation fold creation.

Random-fold CV is optimistically biased by ~0.05 AUC on this dataset
because models memorize scanner/site fingerprints. GroupKFold by
scanner is the only honest validation strategy.

Evidence: metadata-only classifier drops from 0.652 (random folds)
to 0.598 (scanner-grouped) macro AUC.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def get_scanner_groups(
    train_csv: pd.DataFrame,
    dicom_root: Optional[Path] = None,
    metadata_cache: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Assign each study to a scanner group based on DICOM metadata.

    Uses Manufacturer + ManufacturerModelName as the group key.
    If metadata_cache is provided (from a prior DICOM header scan),
    uses that; otherwise reads from DICOM files.

    Args:
        train_csv: DataFrame with StudyInstanceUID column.
        dicom_root: Root path to train_series/ directory (for header scan).
        metadata_cache: Pre-computed DataFrame with StudyInstanceUID,
                       Manufacturer, ManufacturerModelName columns.

    Returns:
        Series mapping StudyInstanceUID -> scanner group string.
    """
    if metadata_cache is not None:
        merged = train_csv.merge(
            metadata_cache[["StudyInstanceUID", "Manufacturer", "ManufacturerModelName"]],
            on="StudyInstanceUID",
            how="left",
        )
        groups = (
            merged["Manufacturer"].fillna("unknown")
            + "_"
            + merged["ManufacturerModelName"].fillna("unknown")
        )
        return groups

    if dicom_root is None:
        raise ValueError("Either dicom_root or metadata_cache must be provided")

    # Scan DICOM headers for scanner metadata
    import pydicom

    records = []
    for _, row in train_csv.iterrows():
        study_uid = row["StudyInstanceUID"]
        study_dir = dicom_root / study_uid

        manufacturer = "unknown"
        model_name = "unknown"

        if study_dir.exists():
            # Read first .dcm file in first series directory
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                dcm_files = list(series_dir.glob("*.dcm"))
                if dcm_files:
                    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                    manufacturer = getattr(ds, "Manufacturer", "unknown")
                    model_name = getattr(ds, "ManufacturerModelName", "unknown")
                    break

        records.append({
            "StudyInstanceUID": study_uid,
            "Manufacturer": manufacturer,
            "ManufacturerModelName": model_name,
        })

    metadata_df = pd.DataFrame(records)
    groups = (
        metadata_df["Manufacturer"].fillna("unknown")
        + "_"
        + metadata_df["ManufacturerModelName"].fillna("unknown")
    )
    return groups


def create_scanner_grouped_folds(
    train_csv: pd.DataFrame,
    label_matrix: Optional[np.ndarray] = None,
    n_folds: int = 5,
    dicom_root: Optional[Path] = None,
    metadata_cache: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create scanner-grouped CV folds.

    Uses GroupKFold with scanner groups. If label_matrix is provided,
    also checks per-fold label distribution and warns on imbalance.

    Args:
        train_csv: DataFrame with StudyInstanceUID and label columns.
        label_matrix: (n_studies, 12) array of labels for stratification check.
        n_folds: Number of folds.
        dicom_root: Root path to DICOM data (for scanner group extraction).
        metadata_cache: Pre-computed scanner metadata.
        seed: Random seed.

    Returns:
        DataFrame with StudyInstanceUID, fold (0 to n_folds-1), scanner_group.
    """
    groups = get_scanner_groups(train_csv, dicom_root, metadata_cache)
    unique_groups = groups.unique()

    logger.info(f"Found {len(unique_groups)} unique scanner groups")
    for g in sorted(unique_groups):
        count = (groups == g).sum()
        logger.info(f"  {g}: {count} studies")

    # GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    fold_assignments = np.full(len(train_csv), -1, dtype=int)

    X = np.arange(len(train_csv)).reshape(-1, 1)  # dummy X
    for fold_idx, (_, val_idx) in enumerate(gkf.split(X, groups=groups)):
        fold_assignments[val_idx] = fold_idx

    result = train_csv[["StudyInstanceUID"]].copy()
    result["fold"] = fold_assignments
    result["scanner_group"] = groups.values

    # Check label distribution per fold if labels provided
    if label_matrix is not None:
        label_cols = [c for c in train_csv.columns if c not in
                      ["StudyInstanceUID", "PatientSex", "Report"]]
        if len(label_cols) == label_matrix.shape[1]:
            for fold in range(n_folds):
                mask = fold_assignments == fold
                fold_labels = label_matrix[mask]
                # Only count non-NaN (non-empty) labels
                valid = ~np.isnan(fold_labels)
                if valid.any():
                    prevalence = np.nanmean(fold_labels, axis=0)
                    logger.info(f"Fold {fold} label prevalence: "
                                f"min={np.nanmin(prevalence):.3f}, "
                                f"max={np.nanmax(prevalence):.3f}, "
                                f"mean={np.nanmean(prevalence):.3f}")

    return result
