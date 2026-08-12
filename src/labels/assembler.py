"""
Assemble silver labels from extraction results into a training-ready matrix.

Key rules:
- not_addressed → soft label 0.5 (masked in loss, never coerced to 0)
- uncertain → soft label 0.5 (or 0.6-0.7 for effusion-type findings)
- present → 1.0
- absent → 0.0
- laterality_ambiguous → 0.5 for both medial and lateral variants

The 58 gold studies are NEVER used for training — they are the only
honest local anchor for validation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .extractor import ExtractionResult, FindingValue, LABELS

logger = logging.getLogger(__name__)


# Findings where silence is weakly informative (not fully uninformative)
# Baker's cyst: silence ≈ negative (present only ~3% when silent)
# Synovitis: silence is uninformative (still ~34% positive when silent)
SILENCE_INFORMATIVE = {
    "Baker's": 0.1,      # Silence strongly suggests absence
    "Effusion": 0.3,     # Silence weakly suggests absence
    "Fracture": 0.2,     # Silence weakly suggests absence
}


def assemble_silver_labels(
    extraction_results: Dict[str, List[ExtractionResult]],
    gold_study_uids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert extraction results into a silver label matrix.

    Args:
        extraction_results: Dict mapping StudyInstanceUID -> List[ExtractionResult].
        gold_study_uids: List of StudyInstanceUIDs with gold labels.
                         These are excluded from silver labels.

    Returns:
        DataFrame with StudyInstanceUID + 12 label columns.
        Values are soft labels in [0, 1], or NaN for masked cells.
    """
    rows = []

    for study_uid, results in extraction_results.items():
        # Skip gold studies — they must not be trained on
        if gold_study_uids and study_uid in gold_study_uids:
            continue

        row = {"StudyInstanceUID": study_uid}

        for result in results:
            label = result.finding
            value = result.value

            if value == FindingValue.PRESENT:
                row[label] = 1.0
            elif value == FindingValue.ABSENT:
                row[label] = 0.0
            elif value == FindingValue.UNCERTAIN:
                row[label] = 0.5
            elif value == FindingValue.NOT_ADDRESSED:
                # Use per-finding silence semantics
                if label in SILENCE_INFORMATIVE:
                    row[label] = SILENCE_INFORMATIVE[label]
                else:
                    row[label] = 0.5  # Uninformative — mask in loss
            elif value == FindingValue.LATERALITY_AMBIGUOUS:
                # Assign 0.5 to both medial and lateral variants
                row[label] = 0.5
            else:
                row[label] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all 12 label columns exist
    for label in LABELS:
        if label not in df.columns:
            df[label] = np.nan

    # Reorder columns
    df = df[["StudyInstanceUID"] + LABELS]

    logger.info(f"Assembled silver labels for {len(df)} studies")
    for label in LABELS:
        valid = df[label].notna()
        if valid.any():
            prevalence = df.loc[valid, label].mean()
            logger.info(f"  {label}: {valid.sum()} labeled, prevalence={prevalence:.3f}")

    return df


def evaluate_against_gold(
    silver_labels: pd.DataFrame,
    gold_labels: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate silver label quality against the 58 gold studies.

    Computes per-label and macro AUC.

    Args:
        silver_labels: DataFrame with StudyInstanceUID + 12 label columns.
        gold_labels: DataFrame with StudyInstanceUID + 12 label columns (binary).

    Returns:
        Dict with per-label AUC and macro AUC.
    """
    from sklearn.metrics import roc_auc_score

    merged = gold_labels.merge(silver_labels, on="StudyInstanceUID", suffixes=("_gold", "_silver"))

    results = {}
    aucs = []

    for label in LABELS:
        gold_col = f"{label}_gold"
        silver_col = f"{label}_silver"

        if gold_col not in merged.columns or silver_col not in merged.columns:
            continue

        y_true = merged[gold_col].dropna()
        y_score = merged.loc[y_true.index, silver_col].dropna()
        y_true = y_true.loc[y_score.index]

        if len(y_true.unique()) < 2:
            logger.warning(f"  {label}: only one class in gold, skipping AUC")
            continue

        auc = roc_auc_score(y_true, y_score)
        results[label] = auc
        aucs.append(auc)

    results["macro_auc"] = np.mean(aucs) if aucs else 0.0

    logger.info(f"Silver label quality vs gold:")
    for label, auc in sorted(results.items(), key=lambda x: x[1]):
        logger.info(f"  {label}: {auc:.4f}")

    return results
