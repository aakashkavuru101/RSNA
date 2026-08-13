"""Checkpoint-only OOF diagnostics for the scored RSNA Knee DINO ensembles.

This notebook never trains. It reconstructs the exact scanner folds, reloads the
validated 224 px and 336 px checkpoints, and saves study-level out-of-fold predictions
plus cohort diagnostics. Hidden test labels are unavailable by competition design, so
scanner-isolated OOF and the untouched 58-study gold set are the honest error proxies.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import __main__ as p


BLEND_224 = 0.20
BLEND_336 = 0.80
MIN_SCANNER_COHORT = 25


def checkpoint_set(marker: str) -> list[Path]:
    packages: dict[Path, dict[int, Path]] = {}
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if marker not in root.lower():
            continue
        for fold in range(5):
            filename = f"clean_dino_fold{fold}.pt"
            if filename in files:
                packages.setdefault(Path(root), {})[fold] = Path(root) / filename
    complete = [files for files in packages.values() if set(files) == set(range(5))]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"Expected one complete five-fold package containing {marker!r}, "
            f"found {len(complete)}"
        )
    return [complete[0][fold] for fold in range(5)]


def load_checkpoint_model(
    dino_path: Path,
    checkpoint_path: Path,
    expected_fold: int,
    expected_image_size: int,
    device: torch.device,
) -> p.KneeDINO:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checks = {
        "fold": expected_fold,
        "img_size": expected_image_size,
        "gold_training_count": 0,
        "targets": p.TARGETS,
        "slots": p.SLOTS,
        "group_size": p.GROUP_SIZE,
        "n_groups": p.N_GROUPS,
    }
    for key, expected in checks.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"{checkpoint_path.name}: {key}={checkpoint.get(key)!r}, expected {expected!r}"
            )
    model = p.build_model(dino_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def masked_macro_auc(
    truth: np.ndarray, prediction: np.ndarray, row_mask: np.ndarray
) -> tuple[float, dict[str, float]]:
    scores = {}
    for index, target in enumerate(p.TARGETS):
        valid = row_mask.copy()
        if valid.sum() < 2 or len(np.unique(truth[valid, index])) != 2:
            continue
        scores[target] = float(roc_auc_score(truth[valid, index], prediction[valid, index]))
    return (float(np.mean(list(scores.values()))) if scores else float("nan"), scores)


def family_metrics(truth: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    output = {}
    rows = np.ones(len(truth), dtype=bool)
    for name, prediction in predictions.items():
        macro, per_target = masked_macro_auc(truth, prediction, rows)
        output[name] = {"macro_auc": macro, "per_target_auc": per_target}
    return output


def cohort_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    cohort_values: np.ndarray,
    minimum: int,
) -> list[dict]:
    rows = []
    for value in sorted(pd.unique(cohort_values).tolist(), key=str):
        mask = cohort_values == value
        if int(mask.sum()) < minimum:
            continue
        macro, per_target = masked_macro_auc(truth, prediction, mask)
        rows.append({
            "cohort": str(value),
            "n": int(mask.sum()),
            "macro_auc": macro,
            "targets_scored": len(per_target),
            "per_target_auc": per_target,
        })
    return rows


def main() -> None:
    p.seed_everything()
    root = p.find_competition_root()
    label_path = p.find_v4_labels()
    dino_path = p.find_dinov2_small()
    checkpoints_224 = checkpoint_set("clean-dinov2-full")
    checkpoints_336 = checkpoint_set("resolution-blend")

    train_df = pd.read_csv(root / "train.csv")
    labels = p.validate_and_join_labels(train_df, label_path)
    series_csv = pd.read_csv(root / "train_series.csv")
    series = p.list_series(
        root / "train_series", series_csv, set(labels["StudyInstanceUID"])
    )
    slots = p.choose_slots(series)
    headers = p.study_headers(slots)
    sides = p.laterality_map(headers)
    scanner_map = p.scanner_groups(headers)
    studies, cache_336, cache_224, slot_mask = p.build_cache(slots, sides, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("OOF reconstruction requires a GPU")
    table = labels.set_index("StudyInstanceUID").loc[studies]
    targets_soft = table[p.TARGETS].values.astype(np.float32)
    targets_binary = (targets_soft > 0.5).astype(np.int8)
    eligible = np.flatnonzero(~table["is_gold"].values)
    gold_indices = np.flatnonzero(table["is_gold"].values)
    fold_assignment = np.full(len(studies), -1, dtype=np.int8)
    oof_224 = np.full((len(studies), len(p.TARGETS)), np.nan, dtype=np.float32)
    oof_336 = np.full_like(oof_224, np.nan)
    gold_224_folds, gold_336_folds = [], []

    for fold in range(5):
        _, _, validation, gold, available = p.make_clean_split(
            studies, labels, scanner_map, fold
        )
        if available != 5 or not np.array_equal(gold, gold_indices):
            raise ValueError("Fold reconstruction differs from training")
        fold_assignment[validation] = fold

        model_224 = load_checkpoint_model(
            dino_path, checkpoints_224[fold], fold, 224, device
        )
        oof_224[validation] = p.predict(
            model_224, cache_224, slot_mask, validation, device
        )
        gold_224_folds.append(
            p.predict(model_224, cache_224, slot_mask, gold_indices, device)
        )
        del model_224
        gc.collect()
        torch.cuda.empty_cache()

        model_336 = load_checkpoint_model(
            dino_path, checkpoints_336[fold], fold, 336, device
        )
        oof_336[validation] = p.predict(
            model_336, cache_336, slot_mask, validation, device
        )
        gold_336_folds.append(
            p.predict(model_336, cache_336, slot_mask, gold_indices, device)
        )
        del model_336
        gc.collect()
        torch.cuda.empty_cache()

    if (fold_assignment[eligible] < 0).any():
        raise RuntimeError("At least one non-gold study has no validation fold")
    if not np.isfinite(oof_224[eligible]).all() or not np.isfinite(oof_336[eligible]).all():
        raise RuntimeError("OOF prediction matrix is incomplete")

    ranked_224 = p.rank_predictions(oof_224[eligible])
    ranked_336 = p.rank_predictions(oof_336[eligible])
    blend = BLEND_224 * ranked_224 + BLEND_336 * ranked_336
    truth = targets_binary[eligible]
    predictions = {"dino_224": ranked_224, "dino_336": ranked_336, "blend": blend}

    scanner = np.array([scanner_map.get(study, "unknown") for study in studies])[eligible]
    fold_values = fold_assignment[eligible]
    slot_count = slot_mask.sum(axis=1).astype(int)[eligible]
    confidence = (0.25 + 0.75 * np.abs(targets_soft - 0.5) * 2.0)[eligible]
    confidence_mean = confidence.mean(axis=1)
    confidence_band = np.asarray(pd.cut(
        confidence_mean, bins=[-np.inf, 0.50, 0.75, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str))

    diagnostics = {
        "scored_public_lb": 0.871,
        "blend_weights": {"dino_224": BLEND_224, "dino_336": BLEND_336},
        "gold_training_studies": 0,
        "gold_eval_studies": int(len(gold_indices)),
        "oof_studies": int(len(eligible)),
        "overall": family_metrics(truth, predictions),
        "by_fold": cohort_metrics(truth, blend, fold_values, 1),
        "by_scanner": cohort_metrics(truth, blend, scanner, MIN_SCANNER_COHORT),
        "by_slot_count": cohort_metrics(truth, blend, slot_count, 20),
        "by_confidence_band": cohort_metrics(truth, blend, confidence_band, 20),
        "slot_coverage": {
            p.SLOTS[index][0]: float(slot_mask[eligible, index].mean())
            for index in range(len(p.SLOTS))
        },
        "family_rank_correlation": {
            target: float(pd.Series(ranked_224[:, index]).corr(
                pd.Series(ranked_336[:, index]), method="spearman"
            ))
            for index, target in enumerate(p.TARGETS)
        },
    }

    gold_truth = table[[f"{target}__gold" for target in p.TARGETS]].values[
        gold_indices
    ].astype(int)
    gold_224 = p.rank_predictions(np.mean(gold_224_folds, axis=0))
    gold_336 = p.rank_predictions(np.mean(gold_336_folds, axis=0))
    gold_blend = BLEND_224 * gold_224 + BLEND_336 * gold_336
    diagnostics["gold_monitor"] = family_metrics(
        gold_truth, {"dino_224": gold_224, "dino_336": gold_336, "blend": gold_blend}
    )

    # Save study-level predictions without report text or raw image data.
    oof = pd.DataFrame({
        "StudyInstanceUID": np.asarray(studies)[eligible],
        "fold": fold_values,
        "scanner_group": scanner,
        "slot_count": slot_count,
        "confidence_mean": confidence_mean,
    })
    for index, (slot_name, _, _) in enumerate(p.SLOTS):
        oof[f"has_{slot_name}"] = slot_mask[eligible, index].astype(int)
    for index, target in enumerate(p.TARGETS):
        safe = target.lower().replace(" ", "_").replace("'", "")
        oof[f"target_{safe}"] = targets_soft[eligible, index]
        oof[f"pred224_{safe}"] = ranked_224[:, index]
        oof[f"pred336_{safe}"] = ranked_336[:, index]
        oof[f"blend_{safe}"] = blend[:, index]

    probability = np.clip(blend, 1e-6, 1 - 1e-6)
    per_cell_bce = -(
        targets_soft[eligible] * np.log(probability)
        + (1 - targets_soft[eligible]) * np.log(1 - probability)
    )
    oof["mean_soft_bce"] = per_cell_bce.mean(axis=1)
    oof["mean_absolute_error"] = np.abs(targets_soft[eligible] - blend).mean(axis=1)
    oof.to_csv("oof_predictions.csv", index=False)
    oof.nlargest(250, "mean_soft_bce").to_csv("largest_error_cohorts.csv", index=False)
    Path("diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    p.log(
        f"saved {len(oof)} OOF rows; blend macro AUC="
        f"{diagnostics['overall']['blend']['macro_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
