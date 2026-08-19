from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontier_unrestricted_submit" / "rsna_knee_frontier_v43_unrestricted.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path3_e11_clean_train.ipynb"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return source.replace(old, new, 1)


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 3 — clean diverse E11 training\n\n"
        "Experimental training run derived from the verified Path 2 0.909 parent. "
        "The new RadImageNet arm reads non-fat-suppressed series at a 130 mm physical "
        "crop. The 58 fully annotated studies are excluded from training, checkpoint "
        "selection, and blend selection; they are monitor-only. Five-fold CV is grouped "
        "by scanner/site signatures extracted from DICOM metadata. This run preserves "
        "the parent submission and emits heads plus OOF diagnostics for a later Path 3 "
        "application notebook.\n"
    )

    header_source = "".join(notebook["cells"][13]["source"])
    header_source = replace_once(
        header_source,
        "'ImagePositionPatient', 'ImageOrientationPatient']",
        "'ImagePositionPatient', 'ImageOrientationPatient', 'Manufacturer', "
        "'ManufacturerModelName', 'MagneticFieldStrength', 'StationName', "
        "'InstitutionName']",
        "scanner DICOM tags",
    )
    notebook["cells"][13]["source"] = [line + "\n" for line in header_source.splitlines()]

    source = "".join(notebook["cells"][32]["source"])
    source = replace_once(source, 'ARM_MODE = "e10"', 'ARM_MODE = "e11"', "E11 mode")
    source = replace_once(
        source,
        '''def make_targets(train):
    """Three independent public report teachers; image-read gold always wins."""
''',
        '''def make_targets(train):
    """Three report teachers; gold rows are identified and zero-weighted."""
''',
        "clean target docstring",
    )
    source = replace_once(
        source,
        '''    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    y[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    w[gold] = 3.0
    return y, w, gold
''',
        '''    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    # AGENTS contract: the 58 official image-label rows never enter optimisation.
    # Keep their report-teacher values only so array shapes remain stable; their
    # weights are zero and the grouped split below excludes them entirely.
    w[gold] = 0.0
    return y, w, gold
''',
        "gold exclusion",
    )

    scanner_helper = r'''

def scanner_groups(train, headers):
    """Build study-level scanner/site groups from DICOM metadata.

    Manufacturer/site identity is preferred. Geometry is included as a stable
    fallback for anonymised scanners; report text and labels are never part of
    the grouping key.
    """
    fields = [
        "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
        "StationName", "InstitutionName", "Rows", "Columns", "PixelSpacing",
    ]
    available = [field for field in fields if field in headers.columns]
    if not available:
        raise RuntimeError("scanner-group DICOM fields are absent")

    def normalise(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            value = "x".join(map(str, value))
        text = str(value).strip().upper()
        return "?" if text in {"", "NAN", "NONE"} else text[:96]

    signatures = {}
    for study, rows in headers.groupby("StudyInstanceUID", sort=False):
        parts = []
        for field in available:
            values = [normalise(value) for value in rows[field].tolist()]
            values = [value for value in values if value != "?"]
            parts.append(max(set(values), key=values.count) if values else "?")
        signature = "|".join(parts)
        signatures[str(study)] = hashlib.sha256(signature.encode()).hexdigest()[:24]
    groups = np.asarray(
        [signatures.get(str(uid), "MISSING_SCANNER") for uid in train.StudyInstanceUID]
    )
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < 5:
        raise RuntimeError(f"only {len(unique)} scanner groups; grouped CV undefined")
    log(
        f"Path3 scanner groups: {len(unique)} groups; "
        f"largest={int(counts.max())} studies"
    )
    return groups
'''
    source = replace_once(
        source,
        '''def report_groups(train):
    report = (train.Report.fillna("").astype(str).str.lower()
              .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.array([hashlib.sha256(x.encode()).hexdigest()[:24] for x in report])
''',
        '''def report_groups(train):
    report = (train.Report.fillna("").astype(str).str.lower()
              .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.array([hashlib.sha256(x.encode()).hexdigest()[:24] for x in report])
''' + scanner_helper,
        "scanner helper insertion",
    )

    source = replace_once(
        source,
        '"mode": "e11-diverse-recipe-training-only",',
        '"mode": "path3-e11-diverse-clean-training-only",\n'
        '        "gold_policy": "58 official rows excluded from optimisation, checkpoint selection, and blend selection; monitor only",',
        "clean audit mode",
    )
    source = replace_once(
        source,
        '''            "Every value here is a local out-of-fold diagnostic on 58 official image "
            "labels. It is not a Kaggle score, and this mode never replaces the parent "
            "submission under any outcome."
''',
        '''            "Weak OOF and blend values are scanner-grouped diagnostics on 4,349 "
            "non-gold report-label rows. The 58 official rows are monitor-only. This is "
            "not a Kaggle score, and training mode never replaces the parent submission."
''',
        "clean evidence boundary",
    )

    e11_start = source.index("def main_v52_e11():")
    e11_end = source.index('\n\nif ARM_MODE == "e11":', e11_start)
    e11_source = source[e11_start:e11_end]
    e11_source = replace_once(
        e11_source,
        '''        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()
''',
        '''        # Scanner/site groups must be resolved while DICOM headers are still resident.
        groups = scanner_groups(train, headers)
        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()
''',
        "scanner grouping before header release",
    )
    source = source[:e11_start] + e11_source + source[e11_end:]

    old_train = '''        y, weights, gold = make_targets(train)
        if int(gold.sum()) != 58:
            raise RuntimeError(f"expected 58 fully gold studies, observed {int(gold.sum())}")
        groups = report_groups(train)
        if len(np.unique(groups)) < 4000:
            raise RuntimeError("unexpected report-group collapse")

        splits = list(GroupKFold(5).split(features, groups=groups))
        fold_id = np.full(len(train), -1, dtype=np.int8)
        folds = []
        oof = np.zeros_like(y, dtype=np.float32)
        for fold, (tr, va) in enumerate(splits):
            if set(groups[tr]).intersection(groups[va]):
                raise RuntimeError(f"report leakage in fold {fold}")
            fold_id[va] = fold
            state, score = train_fold(
                features, token_mask, y, weights, tr, va, fold, device
            )
            if state is None:
                raise RuntimeError(f"fold {fold} produced no checkpoint")
            head = FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            oof[va] = predict_head(head, features, token_mask, va, device)
            folds.append({"fold": fold, "weak_auc": float(score), "state_dict": state})
            del head
            torch.cuda.empty_cache()
        if (fold_id < 0).any() or not np.isfinite(oof).all():
            raise RuntimeError("incomplete E11 OOF")

        weak_auc = macro_auc(y, oof)
        gold_auc = macro_auc(y[gold], oof[gold])
        audit["weak_oof_auc"] = float(weak_auc)
        audit["gold_oof_auc"] = float(gold_auc)
        log(f"E11 OOF weak macro AUC {weak_auc:.5f}")
        log(f"E11 OOF gold macro AUC {gold_auc:.5f} on 58 studies")
'''
    new_train = '''        y, weights, gold = make_targets(train)
        if int(gold.sum()) != 58:
            raise RuntimeError(f"expected 58 fully gold studies, observed {int(gold.sum())}")
        non_gold = np.flatnonzero(~gold)
        gold_rows = np.flatnonzero(gold)
        gold_y = train.loc[gold, TARGETS].to_numpy(np.float32)
        audit["scanner_group_count"] = int(len(np.unique(groups[non_gold])))
        audit["weak_training_rows"] = int(len(non_gold))
        audit["gold_monitor_rows"] = int(len(gold_rows))

        splits = list(GroupKFold(5).split(non_gold, groups=groups[non_gold]))
        fold_id = np.full(len(train), -1, dtype=np.int8)
        folds = []
        oof = np.full_like(y, np.nan, dtype=np.float32)
        gold_fold_predictions = []
        for fold, (tr_pos, va_pos) in enumerate(splits):
            tr = non_gold[tr_pos]
            va = non_gold[va_pos]
            if set(groups[tr]).intersection(groups[va]):
                raise RuntimeError(f"scanner leakage in fold {fold}")
            if np.intersect1d(tr, gold_rows).size or np.intersect1d(va, gold_rows).size:
                raise RuntimeError(f"gold row entered clean fold {fold}")
            fold_id[va] = fold
            state, score = train_fold(
                features, token_mask, y, weights, tr, va, fold, device
            )
            if state is None:
                raise RuntimeError(f"fold {fold} produced no checkpoint")
            head = FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            oof[va] = predict_head(head, features, token_mask, va, device)
            gold_fold_predictions.append(
                predict_head(head, features, token_mask, gold_rows, device)
            )
            folds.append({"fold": fold, "weak_auc": float(score), "state_dict": state})
            del head
            torch.cuda.empty_cache()
        if (fold_id[non_gold] < 0).any() or not np.isfinite(oof[non_gold]).all():
            raise RuntimeError("incomplete Path3 non-gold OOF")

        gold_monitor = np.mean(np.stack(gold_fold_predictions), axis=0)
        oof[gold_rows] = gold_monitor
        weak_auc = macro_auc(y[non_gold], oof[non_gold])
        gold_auc = macro_auc(gold_y, gold_monitor)
        audit["weak_oof_auc"] = float(weak_auc)
        audit["gold_monitor_auc"] = float(gold_auc)
        log(f"Path3 E11 scanner-grouped weak OOF macro AUC {weak_auc:.5f}")
        log(f"Path3 E11 gold monitor macro AUC {gold_auc:.5f} on 58 excluded studies")
'''
    source = replace_once(source, old_train, new_train, "clean scanner-grouped training")

    old_diagnostic = '''        # The question E11 exists to answer is not whether this arm is strong on its own but
        # whether it says something the portfolio does not already know. Both halves are
        # measured against the same 58 rows and the same rank basis the deployed blend uses.
        base_npz = find_input_file("oof.npz")
        with np.load(base_npz, allow_pickle=False) as bundle:
            if bundle["targets"].astype(str).tolist() != TARGETS:
                raise RuntimeError("E11 E2 OOF target order drift")
            if not np.array_equal(
                bundle["ids"].astype(str), train.StudyInstanceUID.astype(str).to_numpy()
            ):
                raise RuntimeError("E11 E2 OOF study order drift")
            base_prediction = bundle["pred"].astype(np.float64)
        base = _v52_rank_columns(base_prediction[gold])
        new = _v52_rank_columns(oof[gold].astype(np.float64))
        gold_y = train.loc[gold, TARGETS].to_numpy(np.float64)
        reference = _v52_target_auc(gold_y, base)
        audit["e2_base_gold_macro"] = float(np.mean([reference[t] for t in TARGETS]))
        ladder = {}
        for alpha in (0.20, 0.35, 0.50):
            scores = _v52_target_auc(gold_y, (1.0 - alpha) * base + alpha * new)
            ladder[f"{alpha:.2f}"] = {
                "macro": float(np.mean([scores[t] for t in TARGETS])),
                "per_target_delta": {t: float(scores[t] - reference[t]) for t in TARGETS},
            }
            log(f"E11 blend alpha={alpha:.2f} gold macro {ladder[f'{alpha:.2f}']['macro']:.5f}")
        audit["blend_vs_e2"] = ladder

        try:
            parent_oof = pd.read_csv(
                find_input_file("v52_oof.csv"), dtype={"StudyInstanceUID": str}
            )
            aligned = train[["StudyInstanceUID"]].merge(
                parent_oof, on="StudyInstanceUID", how="left", validate="one_to_one"
            )
            parent = _v52_rank_columns(aligned[TARGETS].to_numpy(np.float64)[gold])
            audit["correlation_with_parent_arm"] = {
                t: float(np.corrcoef(parent[:, i], new[:, i])[0, 1])
                for i, t in enumerate(TARGETS)
            }
            log(
                "E11 mean rank correlation with the parent arm: "
                f"{np.mean(list(audit['correlation_with_parent_arm'].values())):.3f}"
            )
        except FileNotFoundError:
            audit["correlation_with_parent_arm"] = None
'''
    new_diagnostic = '''        # Select blend evidence only on the 4,349 non-gold scanner-grouped OOF rows.
        base_npz = find_input_file("oof.npz")
        with np.load(base_npz, allow_pickle=False) as bundle:
            if bundle["targets"].astype(str).tolist() != TARGETS:
                raise RuntimeError("Path3 E2 OOF target order drift")
            if not np.array_equal(
                bundle["ids"].astype(str), train.StudyInstanceUID.astype(str).to_numpy()
            ):
                raise RuntimeError("Path3 E2 OOF study order drift")
            base_prediction = bundle["pred"].astype(np.float64)
        base = _v52_rank_columns(base_prediction[non_gold])
        new = _v52_rank_columns(oof[non_gold].astype(np.float64))
        weak_y = y[non_gold].astype(np.float64)
        reference = _v52_target_auc(weak_y, base)
        audit["e2_base_weak_macro"] = float(np.mean([reference[t] for t in TARGETS]))
        ladder = {}
        for alpha in (0.10, 0.20, 0.35, 0.50):
            scores = _v52_target_auc(weak_y, (1.0 - alpha) * base + alpha * new)
            ladder[f"{alpha:.2f}"] = {
                "macro": float(np.mean([scores[t] for t in TARGETS])),
                "per_target_delta": {t: float(scores[t] - reference[t]) for t in TARGETS},
            }
            log(f"Path3 E11 blend alpha={alpha:.2f} weak macro {ladder[f'{alpha:.2f}']['macro']:.5f}")
        audit["blend_vs_e2_weak_oof"] = ladder

        try:
            parent_oof = pd.read_csv(
                find_input_file("v52_oof.csv"), dtype={"StudyInstanceUID": str}
            )
            aligned = train[["StudyInstanceUID"]].merge(
                parent_oof, on="StudyInstanceUID", how="left", validate="one_to_one"
            )
            parent = _v52_rank_columns(aligned[TARGETS].to_numpy(np.float64)[non_gold])
            audit["correlation_with_parent_arm_non_gold"] = {
                t: float(np.corrcoef(parent[:, i], new[:, i])[0, 1])
                for i, t in enumerate(TARGETS)
            }
            log(
                "Path3 E11 mean non-gold rank correlation with parent arm: "
                f"{np.mean(list(audit['correlation_with_parent_arm_non_gold'].values())):.3f}"
            )
        except FileNotFoundError:
            audit["correlation_with_parent_arm_non_gold"] = None
'''
    source = replace_once(source, old_diagnostic, new_diagnostic, "clean weak OOF diagnostic")
    source = replace_once(
        source,
        '"version": "e11-radimagenet-resnet50-diverse-1",',
        '"version": "path3-e11-radimagenet-diverse-clean-1",\n'
        '                "gold_training_or_selection_used": False,\n'
        '                "cv_grouping": "scanner-site DICOM signature",',
        "clean payload contract",
    )
    source = replace_once(
        source,
        '"gold_oof_auc": float(gold_auc),',
        '"gold_monitor_auc": float(gold_auc),',
        "monitor metric name",
    )
    source = replace_once(
        source,
        'audit["status"] = "E11_TRAINED_E2_PRESERVED"',
        'audit["status"] = "PATH3_E11_CLEAN_TRAINED_PARENT_PRESERVED"',
        "clean completion status",
    )

    notebook["cells"][32]["source"] = [line + "\n" for line in source.splitlines()]
    # Cells after E11 are submission-only V40/V43 receipts. Training mode
    # intentionally preserves the parent and does not create an E10 audit, so
    # executing those receipts would turn a successful training run into an error.
    notebook["cells"] = notebook["cells"][:33]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
