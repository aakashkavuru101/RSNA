from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path3_e11_clean_train" / "rsna_knee_path3_e11_clean_train.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path5_supervision_fusion_train.ipynb"


def replace_block(source: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return source.replace(old, new, 1)


TARGET_POLICY = r'''PATH5_LABEL_SOURCE_SPECS = [
    # Keep all source choices fixed before the run. Flight hybrid is excluded here because
    # its published recipe used the 58 official rows to choose the fracture substitution.
    ("pilkwang_v2", "report_labels_v2.csv", 1.00),
    ("steven_v2", "llm_labels_v2.csv", 0.80),
    ("steven_v4", "llm_labels_v4_blend.csv", 1.00),
    ("lixin_sol56", "labels_llm_gpt56sol.csv", 0.40),
]
PATH5_NEUTRAL_VALUES = (0.25, 0.28, 0.50)
PATH5_STRICT_POSITIVE = {"ACL", "Contusion", "Fracture", "Medial Meniscus", "Lateral Meniscus"}


def _path5_source_strength(name, aligned, values, base_weight):
    conf_cols = [target + "__conf" for target in TARGETS]
    if all(col in aligned.columns for col in conf_cols):
        strength = aligned[conf_cols].to_numpy(float)
    else:
        strength = np.clip(2.0 * np.abs(values - 0.5), 0.0, 1.0)
    neutral = np.zeros_like(values, dtype=bool)
    for value in PATH5_NEUTRAL_VALUES:
        neutral |= np.isclose(values, value, atol=1e-6)
    return np.where(np.isfinite(values) & ~neutral, strength * base_weight, 0.0)


def make_targets(train):
    """Consensus report supervision; gold rows remain monitor-only and zero-weighted."""
    uid = "StudyInstanceUID"
    values, strengths, source_names = [], [], []
    for name, filename, base_weight in PATH5_LABEL_SOURCE_SPECS:
        frame = pd.read_csv(find_input_file(filename))
        if frame[uid].duplicated().any():
            raise ValueError(f"duplicate study in {name}")
        keep = [uid] + TARGETS + [target + "__conf" for target in TARGETS if target + "__conf" in frame.columns]
        aligned = train[[uid]].merge(frame[keep], on=uid, how="left", validate="one_to_one")
        source_values = aligned[TARGETS].to_numpy(float)
        values.append(source_values)
        strengths.append(_path5_source_strength(name, aligned, source_values, base_weight))
        source_names.append(name)

    cube = np.stack(values)
    strength = np.stack(strengths)
    if np.any(np.isfinite(cube).sum(0) < 2):
        raise ValueError("fewer than two report teachers for a study/target")

    safe_cube = np.where(np.isfinite(cube), cube, 0.0)
    total = strength.sum(0)
    weighted = (safe_cube * strength).sum(0)
    y = np.divide(weighted, total, out=np.full_like(total, 0.5, dtype=float), where=total > 0)

    pos_votes = ((cube >= 0.65) & (strength > 0.05)).sum(0)
    neg_votes = ((cube <= 0.20) & (strength > 0.05)).sum(0)
    contradiction = (pos_votes > 0) & (neg_votes > 0)
    consensus = ((pos_votes >= 2) & (neg_votes == 0)) | ((neg_votes >= 2) & (pos_votes == 0))
    one_unopposed = ((pos_votes == 1) & (neg_votes == 0)) | ((neg_votes == 1) & (pos_votes == 0))

    y = np.where((pos_votes >= 2) & (neg_votes == 0), np.maximum(y, 0.70), y)
    y = np.where((neg_votes >= 2) & (pos_votes == 0), np.minimum(y, 0.20), y)

    certainty = np.clip(2.0 * np.abs(y - 0.5), 0.0, 1.0)
    single_high = one_unopposed & (total >= 0.75)
    trainable = consensus | single_high
    w = np.clip(total / 2.2, 0.0, 1.0) * (0.35 + 0.65 * certainty)
    w *= np.where(consensus, 1.00, np.where(single_high, 0.45, 0.00))
    w *= np.where(contradiction, 0.20, 1.00)

    for index, target in enumerate(TARGETS):
        if target == "Synovitis":
            # Report silence is common for synovitis. Do not let weak/silent negatives
            # dominate; train mainly on explicit positives or multi-teacher negatives.
            weak_negative = (y[:, index] < 0.5) & (neg_votes[:, index] < 2)
            w[weak_negative, index] *= 0.20
        if target in PATH5_STRICT_POSITIVE:
            weak_positive = (y[:, index] >= 0.5) & (pos_votes[:, index] < 2)
            w[weak_positive, index] *= 0.35

    y = y.astype(np.float32)
    w = w.astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    w[gold] = 0.0
    globals()["PATH5_TARGET_AUDIT"] = {
        "label_sources": source_names,
        "policy": "consensus weighted teacher fusion; neutral 0.25/0.28/0.50 masked; contradictions downweighted",
        "trainable_rule": "at least two hard teachers agree, or one high-confidence teacher is unopposed",
        "trainable_cells_non_gold": int((w[~gold] > 0).sum()),
        "mean_weight_non_gold": float(w[~gold].mean()),
        "per_target_trainable_cells": {
            target: int((w[~gold, index] > 0).sum())
            for index, target in enumerate(TARGETS)
        },
        "per_target_mean_weight": {
            target: float(w[~gold, index].mean())
            for index, target in enumerate(TARGETS)
        },
    }
    return y, w, gold'''


MASKED_AUC = r'''def macro_auc(y, pred, weights=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    pred = np.asarray(pred)
    hard = (y >= .5).astype(np.uint8)
    values = []
    for j in range(hard.shape[1]):
        mask = np.isfinite(y[:, j]) & np.isfinite(pred[:, j])
        if weights is not None:
            mask &= np.asarray(weights)[:, j] > 0
        if np.unique(hard[mask, j]).size == 2:
            values.append(roc_auc_score(hard[mask, j], pred[mask, j]))
    if not values:
        raise ValueError("no binary targets available for AUC")
    return float(np.mean(values))


def _v52_target_auc(y, pred, weights=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    pred = np.asarray(pred)
    hard = (y >= .5).astype(np.uint8)
    scores = {}
    for index, target in enumerate(TARGETS):
        mask = np.isfinite(y[:, index]) & np.isfinite(pred[:, index])
        if weights is not None:
            mask &= np.asarray(weights)[:, index] > 0
        if np.unique(hard[mask, index]).size == 2:
            scores[target] = float(roc_auc_score(hard[mask, index], pred[mask, index]))
    return scores'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 5 — clean supervision-fusion E11 training\n\n"
        "Supervision-only experiment derived from Path 3. Architecture, scanner-grouped "
        "folding, and gold monitor policy stay fixed. This run changes only weak-label "
        "fusion: neutral report states are masked, contradictory teachers are downweighted, "
        "and checkpoint/blend diagnostics use the same non-gold label mask as the loss.\n"
    )

    source = "".join(notebook["cells"][32]["source"])
    source = replace_once(
        source,
        '"mode": "path3-e11-diverse-clean-training-only",',
        '"mode": "path5-e11-supervision-fusion-clean-training-only",',
        "audit mode",
    )
    source = replace_once(
        source,
        "Path3 E11 scanner-grouped weak OOF macro AUC",
        "Path5 E11 masked scanner-grouped weak OOF macro AUC",
        "weak OOF log",
    )
    source = replace_once(
        source,
        "Path3 E11 gold monitor macro AUC",
        "Path5 E11 gold monitor macro AUC",
        "gold monitor log",
    )
    source = replace_block(source, "def make_targets(train):", "\ndef report_groups", TARGET_POLICY)
    source = replace_block(source, "def macro_auc(y, pred):", "\n\n@torch.inference_mode()", MASKED_AUC)
    source = replace_once(
        source,
        "score = macro_auc(y[val_idx], pred)",
        "score = macro_auc(y[val_idx], pred, weights[val_idx])",
        "masked fold score",
    )
    source = replace_once(
        source,
        "weak_auc = macro_auc(y[non_gold], oof[non_gold])",
        "weak_auc = macro_auc(y[non_gold], oof[non_gold], weights[non_gold])",
        "masked weak auc",
    )
    source = replace_once(
        source,
        "audit[\"gold_monitor_rows\"] = int(len(gold_rows))",
        "audit[\"gold_monitor_rows\"] = int(len(gold_rows))\n        audit[\"target_policy\"] = globals().get(\"PATH5_TARGET_AUDIT\", {})",
        "target audit insertion",
    )
    source = replace_once(
        source,
        "reference = _v52_target_auc(weak_y, base)",
        "weak_weights = weights[non_gold].astype(np.float64)\n        reference = _v52_target_auc(weak_y, base, weak_weights)",
        "masked reference",
    )
    source = replace_once(
        source,
        "audit[\"e2_base_weak_macro\"] = float(np.mean([reference[t] for t in TARGETS]))",
        "audit[\"e2_base_weak_macro\"] = float(np.mean(list(reference.values())))",
        "reference macro",
    )
    source = replace_once(
        source,
        "scores = _v52_target_auc(weak_y, (1.0 - alpha) * base + alpha * new)",
        "scores = _v52_target_auc(weak_y, (1.0 - alpha) * base + alpha * new, weak_weights)",
        "masked ladder score",
    )
    source = replace_once(
        source,
        '"macro": float(np.mean([scores[t] for t in TARGETS])),',
        '"macro": float(np.mean(list(scores.values()))),',
        "ladder macro",
    )
    source = replace_once(
        source,
        '"per_target_delta": {t: float(scores[t] - reference[t]) for t in TARGETS},',
        '"per_target_delta": {t: float(scores[t] - reference[t]) for t in scores},',
        "ladder target delta",
    )
    source = replace_once(
        source,
        '"version": "path3-e11-radimagenet-diverse-clean-1",',
        '"version": "path5-e11-supervision-fusion-clean-1",',
        "bundle version",
    )
    source = replace_once(
        source,
        '"weak_oof_auc": float(weak_auc),',
        '"weak_oof_auc": float(weak_auc),\n                "target_policy": globals().get("PATH5_TARGET_AUDIT", {}),',
        "bundle target policy",
    )
    source = replace_once(
        source,
        'audit["status"] = "PATH3_E11_CLEAN_TRAINED_PARENT_PRESERVED"',
        'audit["status"] = "PATH5_E11_SUPERVISION_FUSION_TRAINED_PARENT_PRESERVED"',
        "status",
    )
    notebook["cells"][32]["source"] = [line + "\n" for line in source.splitlines()]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
