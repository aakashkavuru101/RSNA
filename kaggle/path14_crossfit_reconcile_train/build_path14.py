from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path6_kimi_fs160_train" / "rsna_knee_path6_kimi_fs160_train.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path14_crossfit_reconcile_train.ipynb"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return source.replace(old, new, 1)


def replace_last_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count < 1:
        raise RuntimeError(f"{label}: expected at least one match, observed {count}")
    head, separator, tail = source.rpartition(old)
    if not separator:
        raise RuntimeError(f"{label}: last-match replacement failed")
    return head + new + tail


REPORT_STATE = '''    globals()["PATH14_REPORT_STATE"] = {
        "pos_votes": pos_votes.astype(np.int8),
        "neg_votes": neg_votes.astype(np.int8),
        "trainable": trainable.astype(bool),
        "contradiction": contradiction.astype(bool),
    }
    return y, w, gold'''


RECONCILIATION_HELPERS = r'''PATH14_TEACHER_FILE = "v52_e11_oof.csv"
PATH14_TEACHER_SHA256 = "4929f6eaf6a6ca8f11779547c9f31633fcc309e93e09470ae5c1adc41294db5c"
PATH14_IMAGE_QUANTILE = 0.875
PATH14_PSEUDO_POSITIVE = 0.82
PATH14_PSEUDO_NEGATIVE = 0.18
PATH14_POSITIVE_WEIGHT = 0.18
PATH14_NEGATIVE_WEIGHT = 0.10
PATH14_CONFLICT_SCALE = 0.20


def _path14_rank_non_gold(values, gold):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(gold), len(TARGETS)) or not np.isfinite(values).all():
        raise ValueError(f"invalid Path14 teacher shape/value: {values.shape}")
    ranked = np.full_like(values, np.nan, dtype=np.float64)
    non_gold = ~np.asarray(gold, dtype=bool)
    ranked[non_gold] = pd.DataFrame(values[non_gold]).rank(
        method="average", pct=True
    ).to_numpy(np.float64)
    return ranked


def _path14_load_teacher_oof(train, gold):
    matches = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if PATH14_TEACHER_FILE not in files:
            continue
        candidate = Path(root) / PATH14_TEACHER_FILE
        if _v52_sha256(candidate) == PATH14_TEACHER_SHA256:
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"Path14 expected one hash-pinned teacher OOF, observed {len(matches)}"
        )
    path = matches[0]
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    expected = ["StudyInstanceUID", *TARGETS, "fold", "is_gold"]
    if frame.columns.tolist() != expected or frame.StudyInstanceUID.duplicated().any():
        raise RuntimeError("Path14 teacher OOF schema/identity drift")
    aligned = train[["StudyInstanceUID"]].merge(
        frame, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    values = aligned[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Path14 teacher OOF coverage is incomplete")
    if not np.array_equal(aligned["is_gold"].astype(bool).to_numpy(), gold):
        raise RuntimeError("Path14 teacher gold-mask contract drift")
    folds = aligned["fold"].to_numpy()
    if not np.isin(folds[~gold], np.arange(5)).all():
        raise RuntimeError("Path14 teacher has non-cross-fitted silver rows")
    return values


def make_reconciled_targets(y, weights, gold, path5_oof, path6_oof):
    """Reconcile report supervision using two clean cross-fitted image views.

    Image consensus may add a low-weight target only where the report teachers
    were not trainable. It may downweight a report/image contradiction, but it
    never flips a confident report label. Gold rows remain zero-weighted.
    """
    state = globals().get("PATH14_REPORT_STATE")
    if not isinstance(state, dict):
        raise RuntimeError("Path14 report-vote state is absent")
    teacher_rank = _path14_rank_non_gold(path5_oof, gold)
    current_rank = _path14_rank_non_gold(path6_oof, gold)
    threshold = float(PATH14_IMAGE_QUANTILE)
    image_high = (teacher_rank >= threshold) & (current_rank >= threshold)
    image_low = (teacher_rank <= 1.0 - threshold) & (
        current_rank <= 1.0 - threshold
    )

    pos_votes = np.asarray(state["pos_votes"])
    neg_votes = np.asarray(state["neg_votes"])
    report_positive = (pos_votes >= 2) & (neg_votes == 0)
    report_negative = (neg_votes >= 2) & (pos_votes == 0)
    report_trainable = np.asarray(state["trainable"], dtype=bool)
    untrusted = ~report_trainable
    add_positive = untrusted & image_high
    add_negative = untrusted & image_low
    conflict = (report_positive & image_low) | (report_negative & image_high)
    agreement = (report_positive & image_high) | (report_negative & image_low)

    reconciled_y = np.asarray(y, dtype=np.float32).copy()
    reconciled_w = np.asarray(weights, dtype=np.float32).copy()
    reconciled_y[add_positive] = PATH14_PSEUDO_POSITIVE
    reconciled_y[add_negative] = PATH14_PSEUDO_NEGATIVE
    reconciled_w[add_positive] = np.maximum(
        reconciled_w[add_positive], PATH14_POSITIVE_WEIGHT
    )
    reconciled_w[add_negative] = np.maximum(
        reconciled_w[add_negative], PATH14_NEGATIVE_WEIGHT
    )
    reconciled_w[conflict] *= PATH14_CONFLICT_SCALE
    reconciled_w[agreement] = np.minimum(1.0, reconciled_w[agreement] * 1.05)
    reconciled_w[gold] = 0.0
    if np.any(reconciled_w[gold] != 0):
        raise RuntimeError("Path14 gold row received training weight")

    non_gold = ~gold
    audit = {
        "teacher_file": PATH14_TEACHER_FILE,
        "teacher_sha256": PATH14_TEACHER_SHA256,
        "image_quantile": threshold,
        "policy": (
            "two-view cross-fitted image consensus adds low-weight targets only to "
            "untrusted report cells; contradictions are downweighted, never flipped"
        ),
        "added_positive_cells": int((add_positive & non_gold[:, None]).sum()),
        "added_negative_cells": int((add_negative & non_gold[:, None]).sum()),
        "downweighted_conflicts": int((conflict & non_gold[:, None]).sum()),
        "reinforced_agreements": int((agreement & non_gold[:, None]).sum()),
        "per_target": {
            target: {
                "added_positive": int((add_positive[:, index] & non_gold).sum()),
                "added_negative": int((add_negative[:, index] & non_gold).sum()),
                "conflicts": int((conflict[:, index] & non_gold).sum()),
                "teacher_rank_correlation": float(np.corrcoef(
                    teacher_rank[non_gold, index], current_rank[non_gold, index]
                )[0, 1]),
            }
            for index, target in enumerate(TARGETS)
        },
    }
    return reconciled_y, reconciled_w, audit
'''


SECOND_STAGE = r'''        teacher_oof = _path14_load_teacher_oof(train, gold)
        path14_y, path14_weights, reconciliation_audit = make_reconciled_targets(
            y, weights, gold, teacher_oof, oof
        )
        stage1_reconciled_auc = macro_auc(
            path14_y[non_gold], oof[non_gold], path14_weights[non_gold]
        )
        path14_oof = np.full_like(path14_y, np.nan, dtype=np.float32)
        path14_folds = []
        path14_gold_predictions = []
        for fold, (tr_pos, va_pos) in enumerate(splits):
            tr = non_gold[tr_pos]
            va = non_gold[va_pos]
            if set(groups[tr]).intersection(groups[va]):
                raise RuntimeError(f"Path14 scanner leakage in fold {fold}")
            if np.intersect1d(tr, gold_rows).size or np.intersect1d(va, gold_rows).size:
                raise RuntimeError(f"gold row entered Path14 fold {fold}")
            state, score = train_fold(
                features, token_mask, path14_y, path14_weights,
                tr, va, fold + 20, device
            )
            if state is None:
                raise RuntimeError(f"Path14 fold {fold} produced no checkpoint")
            head = FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            path14_oof[va] = predict_head(head, features, token_mask, va, device)
            path14_gold_predictions.append(
                predict_head(head, features, token_mask, gold_rows, device)
            )
            path14_folds.append({
                "fold": fold,
                "reconciled_weak_auc": float(score),
                "state_dict": state,
            })
            del head
            torch.cuda.empty_cache()
        if not np.isfinite(path14_oof[non_gold]).all():
            raise RuntimeError("incomplete Path14 non-gold OOF")

        path14_gold_monitor = np.mean(np.stack(path14_gold_predictions), axis=0)
        path14_oof[gold_rows] = path14_gold_monitor
        path14_original_auc = macro_auc(
            y[non_gold], path14_oof[non_gold], weights[non_gold]
        )
        path14_reconciled_auc = macro_auc(
            path14_y[non_gold], path14_oof[non_gold], path14_weights[non_gold]
        )
        path14_gold_auc = macro_auc(gold_y, path14_gold_monitor)
        original_stage1 = _v52_target_auc(
            y[non_gold], oof[non_gold], weights[non_gold]
        )
        original_path14 = _v52_target_auc(
            y[non_gold], path14_oof[non_gold], weights[non_gold]
        )
        reconciliation_audit.update({
            "stage1_original_weak_auc": float(weak_auc),
            "stage1_reconciled_weak_auc": float(stage1_reconciled_auc),
            "path14_original_weak_auc": float(path14_original_auc),
            "path14_reconciled_weak_auc": float(path14_reconciled_auc),
            "path14_gold_monitor_auc": float(path14_gold_auc),
            "per_target_original_delta": {
                target: float(original_path14[target] - original_stage1[target])
                for target in TARGETS
            },
            "mean_rank_correlation_with_stage1": float(np.mean([
                np.corrcoef(
                    _v52_rank_columns(oof[non_gold])[:, index],
                    _v52_rank_columns(path14_oof[non_gold])[:, index],
                )[0, 1]
                for index in range(len(TARGETS))
            ])),
        })
        non_regressing = sum(
            delta >= -0.005
            for delta in reconciliation_audit["per_target_original_delta"].values()
        )
        candidate_ready = bool(
            path14_reconciled_auc >= stage1_reconciled_auc + 0.002
            and path14_original_auc >= weak_auc - 0.002
            and non_regressing >= 9
        )
        reconciliation_audit["candidate_ready_without_gold_selection"] = candidate_ready
        reconciliation_audit["candidate_gate"] = {
            "minimum_reconciled_gain": 0.002,
            "maximum_original_report_regression": 0.002,
            "minimum_non_regressing_targets_at_minus_0.005": 9,
            "observed_non_regressing_targets": int(non_regressing),
        }
        audit["reconciliation"] = reconciliation_audit
        audit["candidate_ready"] = candidate_ready
        audit["path14_original_weak_auc"] = float(path14_original_auc)
        audit["path14_reconciled_weak_auc"] = float(path14_reconciled_auc)
        audit["path14_gold_monitor_auc"] = float(path14_gold_auc)
        log(
            f"Path14 original/reconciled weak AUC "
            f"{path14_original_auc:.5f}/{path14_reconciled_auc:.5f}; "
            f"stage1 reconciled={stage1_reconciled_auc:.5f}; "
            f"gold monitor={path14_gold_auc:.5f}; candidate_ready={candidate_ready}"
        )'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 14 — cross-fitted supervision reconciliation\n\n"
        "Original supervision-first experiment on the proven Path 6 FS/160 mm pixels. "
        "A second clean OOF image view and the Path 6 OOF must agree in the top or bottom "
        "12.5% before they can add a low-weight target to a report cell that the report "
        "teachers could not train. Image evidence only downweights contradictions; it never "
        "flips a confident report label. The 58 gold rows remain monitor-only and are absent "
        "from optimization and candidate selection.\n"
    )
    source = "".join(notebook["cells"][32]["source"])
    source = replace_once(
        source,
        '"mode": "path6-kimi-fs160-clean-training-only",',
        '"mode": "path14-crossfit-reconciliation-training-only",',
        "audit mode",
    )
    source = replace_once(
        source,
        '    return y, w, gold\n\n\ndef report_groups',
        REPORT_STATE + "\n\n\n" + RECONCILIATION_HELPERS + "\n\ndef report_groups",
        "report state and reconciliation helpers",
    )
    marker = '        log(f"Path6 Kimi FS160 gold monitor macro AUC {gold_auc:.5f} on 58 excluded studies")'
    source = replace_once(
        source, marker, marker + "\n\n" + SECOND_STAGE,
        "second-stage insertion",
    )
    source = replace_once(
        source,
        "new = _v52_rank_columns(oof[non_gold].astype(np.float64))",
        "new = _v52_rank_columns(path14_oof[non_gold].astype(np.float64))",
        "final OOF ladder",
    )
    old_package = '''                "version": "path6-kimi-fs160-clean-1",
                "gold_training_or_selection_used": False,
                "cv_grouping": "scanner-site DICOM signature",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "slots": [list(slot) for slot in E11_SLOTS],
                "crop_mm": E11_CROP_MM,
                "img": E11_IMG,
                "slices_per_plane": E11_CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": folds,
                "weak_oof_auc": float(weak_auc),
                "target_policy": globals().get("PATH5_TARGET_AUDIT", {}),
                "gold_monitor_auc": float(gold_auc),'''
    new_package = '''                "version": "path14-crossfit-reconciliation-clean-1",
                "gold_training_or_selection_used": False,
                "cv_grouping": "scanner-site DICOM signature",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "slots": [list(slot) for slot in E11_SLOTS],
                "crop_mm": E11_CROP_MM,
                "img": E11_IMG,
                "slices_per_plane": E11_CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": path14_folds,
                "weak_oof_auc": float(path14_original_auc),
                "reconciled_weak_oof_auc": float(path14_reconciled_auc),
                "target_policy": globals().get("PATH5_TARGET_AUDIT", {}),
                "reconciliation": reconciliation_audit,
                "candidate_ready": candidate_ready,
                "gold_monitor_auc": float(path14_gold_auc),'''
    source = replace_once(source, old_package, new_package, "Path14 package")
    source = replace_once(
        source,
        '            output / "v52_e11_heads.pt",',
        '            output / "path14_reconciled_heads.pt",',
        "head output name",
    )
    source = replace_last_once(
        source,
        "oof_frame = pd.DataFrame(oof, columns=TARGETS)",
        "oof_frame = pd.DataFrame(path14_oof, columns=TARGETS)",
        "final OOF frame",
    )
    source = replace_once(
        source,
        'oof_frame.to_csv(output / "v52_e11_oof.csv", index=False)',
        'oof_frame.to_csv(output / "path14_reconciled_oof.csv", index=False)',
        "OOF output name",
    )
    source = replace_once(
        source,
        'audit["status"] = "PATH6_KIMI_FS160_TRAINED_PARENT_PRESERVED"',
        'audit["status"] = ("PATH14_CANDIDATE_READY_PARENT_PRESERVED" if candidate_ready else "PATH14_REJECTED_PARENT_PRESERVED")',
        "status",
    )
    source = replace_once(
        source,
        'audit["heads_sha256"] = _v52_sha256(output / "v52_e11_heads.pt")',
        'audit["heads_sha256"] = _v52_sha256(output / "path14_reconciled_heads.pt")',
        "head hash",
    )
    source = replace_once(
        source,
        'audit["oof_sha256"] = _v52_sha256(output / "v52_e11_oof.csv")',
        'audit["oof_sha256"] = _v52_sha256(output / "path14_reconciled_oof.csv")',
        "OOF hash",
    )
    notebook["cells"][32]["source"] = [line + "\n" for line in source.splitlines()]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
