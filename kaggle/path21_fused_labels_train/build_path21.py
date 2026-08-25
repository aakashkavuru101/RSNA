from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path6_kimi_fs160_train" / "rsna_knee_path6_kimi_fs160_train.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path21_fused_labels_train.ipynb"
METADATA_SOURCE = ROOT / "path17_gold_crossfit_train" / "kernel-metadata.json"
METADATA_OUT = Path(__file__).resolve().parent / "kernel-metadata.json"
GOLD_FOLDS = ROOT.parent / "input" / "gold_folds.csv"
LABELS_V5 = ROOT.parent / "input" / "silver_labels_v5.csv"

GOLD_FOLDS_CONTENT = GOLD_FOLDS.read_text()
GOLD_FOLDS_SHA256 = hashlib.sha256(GOLD_FOLDS_CONTENT.encode("utf-8")).hexdigest()
if '"""' in GOLD_FOLDS_CONTENT:
    raise RuntimeError("gold_folds.csv cannot be embedded as a triple-quoted constant")
LABELS_V5_SHA256 = hashlib.sha256(LABELS_V5.read_bytes()).hexdigest()


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


OLD_MAKE_TARGETS = '''def make_targets(train):
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


NEW_MAKE_TARGETS = (
    'PATH21_LABELS_FILE = "silver_labels_v5.csv"\n'
    f'PATH21_LABELS_SHA256 = "{LABELS_V5_SHA256}"\n'
    "PATH21_LAMBDAS = (1.0, 2.0, 4.0, 8.0)\n"
    "PATH21_DEFAULT_LAMBDA = 2.0\n"
    "PATH21_NOISE_FLOOR = 0.02\n"
    'PATH21_GOLD_FOLDS_FILE = "gold_folds.csv"\n'
    f'PATH21_GOLD_FOLDS_SHA256 = "{GOLD_FOLDS_SHA256}"\n'
    'PATH21_GOLD_FOLDS_CSV = """' + GOLD_FOLDS_CONTENT + '"""\n'
    r'''


def make_targets(train):
    """Fused silver-v5 supervision; gold rows remain monitor-only and zero-weighted.

    silver_labels_v5.csv is the per-label fusion of the public teachers
    (flight/steven/pilkwang/lixin) and the in-house extractor, selected under
    the gold cross-fit protocol by notebooks/08_label_fusion.py. A cell value
    of exactly 0.5 encodes a missing label and is masked to zero weight.
    """
    uid = "StudyInstanceUID"
    label_path = find_input_file(PATH21_LABELS_FILE)
    if _v52_sha256(label_path) != PATH21_LABELS_SHA256:
        raise RuntimeError(f"{PATH21_LABELS_FILE} failed its pinned sha256")
    frame = pd.read_csv(label_path)
    if frame.columns.tolist() != [uid] + TARGETS:
        raise ValueError(f"unexpected {PATH21_LABELS_FILE} columns: {frame.columns.tolist()}")
    if len(frame) != 4407 or frame[uid].duplicated().any():
        raise ValueError(f"{PATH21_LABELS_FILE} is malformed")
    aligned = train[[uid]].merge(frame, on=uid, how="left", validate="one_to_one")
    y = aligned[TARGETS].to_numpy(np.float32)
    if np.isnan(y).all(axis=1).any():
        raise ValueError(f"{PATH21_LABELS_FILE} leaves a study fully unlabelled")
    missing = ~np.isfinite(y) | np.isclose(y, 0.5, atol=1e-6)
    y = np.where(np.isfinite(y), y, np.float32(0.5)).astype(np.float32)
    w = (~missing).astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    w[gold] = 0.0
    globals()["PATH5_TARGET_AUDIT"] = {
        "label_sources": [PATH21_LABELS_FILE],
        "label_source_sha256": PATH21_LABELS_SHA256,
        "label_source_path": str(label_path),
        "policy": "fused silver v5 matrix; 0.5-missing cells masked; gold rows zero-weighted",
        "trainable_rule": "every fused cell at full unit weight unless missing",
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
    return y, w, gold


def _path21_load_gold_folds(train, gold, gold_rows):
    """Load the immutable 5-fold split of the 58 gold studies.

    The split ships inside this notebook as a sha256-pinned constant; a copy
    discovered under /kaggle/input is preferred only when its hash matches the
    pin. Any drift, schema change, or UID mismatch against the notebook's own
    gold mask is a hard fail.
    """
    from io import StringIO

    content = PATH21_GOLD_FOLDS_CSV
    source = "embedded-constant"
    try:
        candidate = find_input_file(PATH21_GOLD_FOLDS_FILE)
    except FileNotFoundError:
        pass
    else:
        if _v52_sha256(candidate) != PATH21_GOLD_FOLDS_SHA256:
            raise RuntimeError(f"gold_folds.csv hash drift at {candidate}")
        content = candidate.read_text()
        source = str(candidate)
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != PATH21_GOLD_FOLDS_SHA256:
        raise RuntimeError("gold_folds content failed its pinned sha256")
    frame = pd.read_csv(StringIO(content), dtype={"StudyInstanceUID": str})
    if frame.columns.tolist() != ["StudyInstanceUID", "gold_fold"] or len(frame) != 58:
        raise RuntimeError("gold_folds.csv is malformed")
    if frame.StudyInstanceUID.duplicated().any():
        raise RuntimeError("gold_folds.csv has a duplicated study")
    aligned = train[["StudyInstanceUID"]].merge(
        frame, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    mask_uids = set(train.StudyInstanceUID.astype(str).to_numpy()[gold_rows])
    if set(frame.StudyInstanceUID.astype(str)) != mask_uids:
        raise RuntimeError("gold_folds.csv studies differ from the notebook gold mask")
    values = aligned["gold_fold"].to_numpy(np.float64)
    if np.isfinite(values[~gold]).any():
        raise RuntimeError("gold_folds.csv assigns a fold to a non-gold row")
    picked = values[gold_rows]
    if not np.isfinite(picked).all():
        raise RuntimeError("gold_folds.csv is missing a gold study")
    if not np.array_equal(picked, picked.astype(np.int64)):
        raise RuntimeError("gold_folds.csv has a non-integer fold")
    if not np.isin(picked.astype(np.int64), np.arange(5)).all():
        raise RuntimeError("gold_folds.csv fold outside 0..4")
    gold_fold_of = np.full(len(train), -1, dtype=np.int64)
    gold_fold_of[gold_rows] = picked.astype(np.int64)
    if sorted(np.unique(gold_fold_of[gold_rows]).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("gold_folds.csv does not cover all five folds")
    return gold_fold_of, source


def _path21_gold_override(y, weights, gold, gold_y, lam):
    """Gold override: binary gold targets at relative weight lam on the 58 gold
    rows. Every non-gold (target, weight) cell stays byte-identical to the
    parent, which is asserted rather than assumed.
    """
    over_y = np.asarray(y, dtype=np.float32).copy()
    over_w = np.asarray(weights, dtype=np.float32).copy()
    over_y[gold] = gold_y.astype(np.float32)
    over_w[gold] = np.float32(lam)
    if not (
        np.array_equal(over_y[~gold], y[~gold], equal_nan=True)
        and np.array_equal(over_w[~gold], weights[~gold], equal_nan=True)
    ):
        raise RuntimeError("Path21 override touched a non-gold cell")
    if not np.array_equal(over_y[gold], gold_y.astype(np.float32)):
        raise RuntimeError("Path21 gold target override drift")
    if np.any(over_w[gold] != np.float32(lam)):
        raise RuntimeError("Path21 gold weight override drift")
    return over_y, over_w'''
)


SECOND_STAGE = r'''        gold_fold_of, gold_folds_source = _path21_load_gold_folds(train, gold, gold_rows)
        gfold = gold_fold_of[gold_rows]
        parent_gold_per_target = _v52_target_auc(gold_y, gold_monitor)
        remaining = TIME_BUDGET - (time.time() - float(globals().get("T0", time.time())))
        if remaining < 3.0 * 3600:
            raise TimeoutError(f"only {remaining / 60:.1f} minutes remain for the Path21 sweep")
        log(f"Path21 gold folds from {gold_folds_source}; lambda sweep {PATH21_LAMBDAS}")

        path21_crossfit_pred = {}
        path21_deploy_oof = {}
        path21_deploy_folds = {}
        path21_per_lambda = {}
        for lam_index, lam in enumerate(PATH21_LAMBDAS):
            over_y, over_w = _path21_gold_override(y, weights, gold, gold_y, lam)
            if np.any(over_w[gold_rows] != np.float32(lam)):
                raise RuntimeError("Path21 gold row missing its override weight")

            # Honest gold evidence: gold fold k is absent from both the training
            # set and the early-stopping validation set of cross-fit model k.
            cross = np.full((len(gold_rows), len(TARGETS)), np.nan, dtype=np.float32)
            for k in range(5):
                tr_pos, va_pos = splits[k]
                tr = np.concatenate([non_gold[tr_pos], gold_rows[gfold != k]])
                va = non_gold[va_pos]
                held = gold_rows[gfold == k]
                if set(groups[non_gold[tr_pos]]).intersection(groups[va]):
                    raise RuntimeError(f"Path21 scanner leakage in crossfit fold {k}")
                if np.intersect1d(tr, held).size or np.intersect1d(va, held).size:
                    raise RuntimeError(f"held-out gold entered Path21 crossfit fold {k}")
                state, score = train_fold(
                    features, token_mask, over_y, over_w,
                    tr, va, 40 + 10 * lam_index + k, device
                )
                if state is None:
                    raise RuntimeError(
                        f"Path21 crossfit fold {k} (lambda={lam}) produced no checkpoint"
                    )
                head = FoundationQueryHead().to(device)
                head.load_state_dict(state, strict=True)
                cross[gfold == k] = predict_head(head, features, token_mask, held, device)
                del head, state
                torch.cuda.empty_cache()
            if not np.isfinite(cross).all():
                raise RuntimeError(f"Path21 cross-fitted gold OOF has NaN at lambda={lam}")
            path21_crossfit_pred[lam] = cross

            # Deployment set at this lambda: the parent's scanner-grouped splits,
            # silver cells byte-identical, all 58 gold rows overridden in training
            # only. Validation folds stay purely non-gold.
            deploy_oof = np.full_like(y, np.nan, dtype=np.float32)
            deploy_states = []
            for fold, (tr_pos, va_pos) in enumerate(splits):
                tr = np.concatenate([non_gold[tr_pos], gold_rows])
                va = non_gold[va_pos]
                if set(groups[non_gold[tr_pos]]).intersection(groups[va]):
                    raise RuntimeError(f"Path21 scanner leakage in deployment fold {fold}")
                if np.intersect1d(va, gold_rows).size:
                    raise RuntimeError(
                        f"gold row entered Path21 deployment validation fold {fold}"
                    )
                state, score = train_fold(
                    features, token_mask, over_y, over_w,
                    tr, va, 90 + 10 * lam_index + fold, device
                )
                if state is None:
                    raise RuntimeError(
                        f"Path21 deployment fold {fold} (lambda={lam}) produced no checkpoint"
                    )
                head = FoundationQueryHead().to(device)
                head.load_state_dict(state, strict=True)
                deploy_oof[va] = predict_head(head, features, token_mask, va, device)
                deploy_states.append({
                    "fold": fold,
                    "weak_auc": float(score),
                    "state_dict": state,
                })
                del head
                torch.cuda.empty_cache()
            if not np.isfinite(deploy_oof[non_gold]).all():
                raise RuntimeError(f"incomplete Path21 deployment silver OOF at lambda={lam}")
            path21_deploy_oof[lam] = deploy_oof
            path21_deploy_folds[f"{lam:.1f}"] = deploy_states

            cross_target_auc = _v52_target_auc(gold_y, cross)
            deploy_weak_auc = macro_auc(y[non_gold], deploy_oof[non_gold], weights[non_gold])
            path21_per_lambda[f"{lam:.1f}"] = {
                "crossfit_gold_macro_auc": float(macro_auc(gold_y, cross)),
                "crossfit_gold_per_target_auc": {
                    target: float(value) for target, value in cross_target_auc.items()
                },
                "deployment_weak_oof_auc": float(deploy_weak_auc),
            }
            log(
                f"Path21 lambda={lam:.1f} cross-fitted gold macro AUC "
                f"{path21_per_lambda[f'{lam:.1f}']['crossfit_gold_macro_auc']:.5f}; "
                f"deployment weak OOF {deploy_weak_auc:.5f}"
            )

        # Per-target selection on the honest cross-fitted gold read only, with a
        # 0.02 noise floor (n=58 cannot resolve smaller gaps) and ties broken
        # toward the smallest lambda.
        lambda_star = {}
        lambda_star_basis = {}
        for target in TARGETS:
            observed = {
                lam: path21_per_lambda[f"{lam:.1f}"]["crossfit_gold_per_target_auc"].get(target)
                for lam in PATH21_LAMBDAS
            }
            observed = {lam: auc for lam, auc in observed.items() if auc is not None}
            if len(observed) < 2:
                lambda_star[target] = PATH21_DEFAULT_LAMBDA
                lambda_star_basis[target] = "insufficient_crossfit_evidence_default"
                continue
            spread = max(observed.values()) - min(observed.values())
            if spread < PATH21_NOISE_FLOOR:
                lambda_star[target] = PATH21_DEFAULT_LAMBDA
                lambda_star_basis[target] = f"noise_floor_spread_{spread:.4f}_default"
            else:
                best = max(observed.values())
                lambda_star[target] = min(
                    lam for lam, auc in observed.items() if auc == best
                )
                lambda_star_basis[target] = f"argmax_spread_{spread:.4f}"

        path21_oof = np.full_like(y, np.nan, dtype=np.float32)
        for index, target in enumerate(TARGETS):
            lam = lambda_star[target]
            path21_oof[non_gold, index] = path21_deploy_oof[lam][non_gold, index]
            path21_oof[gold_rows, index] = path21_crossfit_pred[lam][:, index]
        if not np.isfinite(path21_oof).all():
            raise RuntimeError("incomplete Path21 per-target lambda-star OOF")
        path21_weak_auc = macro_auc(y[non_gold], path21_oof[non_gold], weights[non_gold])
        path21_gold_auc = macro_auc(gold_y, path21_oof[gold_rows])
        path21_lambda_star_gold_per_target = {
            target: float(
                path21_per_lambda[f"{lambda_star[target]:.1f}"]["crossfit_gold_per_target_auc"][target]
            )
            for target in TARGETS
            if target
            in path21_per_lambda[f"{lambda_star[target]:.1f}"]["crossfit_gold_per_target_auc"]
        }
        audit["gold_usage"] = "crossfit"
        audit["gold_integration"] = {
            "protocol": (
                "GOLD_INTEGRATION_PLAN.md section 2 cross-fit; a gold study is "
                "never evaluated on a model whose training or early stopping saw it"
            ),
            "label_source": PATH21_LABELS_FILE,
            "label_source_sha256": PATH21_LABELS_SHA256,
            "gold_folds_file": PATH21_GOLD_FOLDS_FILE,
            "gold_folds_sha256": PATH21_GOLD_FOLDS_SHA256,
            "gold_folds_source": gold_folds_source,
            "lambdas": [float(lam) for lam in PATH21_LAMBDAS],
            "noise_floor": PATH21_NOISE_FLOOR,
            "default_lambda": PATH21_DEFAULT_LAMBDA,
            "cv_grouping": "scanner-site DICOM signature",
            "parent_lambda0_gold_monitor_auc": float(gold_auc),
            "parent_gold_monitor_per_target_auc": {
                target: float(value) for target, value in parent_gold_per_target.items()
            },
            "per_lambda": path21_per_lambda,
            "lambda_star": {target: float(lam) for target, lam in lambda_star.items()},
            "lambda_star_basis": lambda_star_basis,
            "lambda_star_gold_per_target_auc": path21_lambda_star_gold_per_target,
            "lambda_star_weak_oof_auc": float(path21_weak_auc),
            "lambda_star_gold_crossfit_auc": float(path21_gold_auc),
            "deployment": (
                "per-target lambda-star; deployment heads trained with all 58 gold "
                "overridden on the parent's scanner-grouped splits; fold state_dicts "
                "for every lambda variant saved in path21_gold_heads.pt"
            ),
        }
        log(
            f"Path21 lambda-star weak OOF {path21_weak_auc:.5f}; "
            f"cross-fitted gold AUC {path21_gold_auc:.5f} "
            f"(parent gold-free monitor {gold_auc:.5f}); lambda_star={lambda_star}"
        )'''


OLD_PACKAGE = '''                "version": "path6-kimi-fs160-clean-1",
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
NEW_PACKAGE = '''                "version": "path21-fused-labels-crossfit-1",
                "gold_training_or_selection_used": True,
                "gold_usage": "crossfit",
                "label_source": PATH21_LABELS_FILE,
                "label_source_sha256": PATH21_LABELS_SHA256,
                "cv_grouping": "scanner-site DICOM signature",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "slots": [list(slot) for slot in E11_SLOTS],
                "crop_mm": E11_CROP_MM,
                "img": E11_IMG,
                "slices_per_plane": E11_CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": folds,
                "gold_fold_variants": path21_deploy_folds,
                "lambda_star": {target: float(lam) for target, lam in lambda_star.items()},
                "weak_oof_auc": float(weak_auc),
                "lambda_star_weak_oof_auc": float(path21_weak_auc),
                "target_policy": globals().get("PATH5_TARGET_AUDIT", {}),
                "gold_monitor_auc": float(gold_auc),'''


EXPECTED_MUTATIONS = 12


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    fired = 0

    notebook["cells"][0]["source"] = (
        "# Path 21 — fused silver-v5 labels + cross-fitted gold head training\n\n"
        "A label-upgrade fork of the proven Path 6 FS/160 mm pixel arm with the "
        "Path 17 gold-integration second stage. Stage 1 is gold-free head training "
        "on `silver_labels_v5.csv` (per-label fusion of the public teachers and the "
        "in-house extractor, selected under the gold cross-fit protocol by "
        "`notebooks/08_label_fusion.py`; 0.5-missing cells masked), with the 58 "
        "official rows monitor-only. Stage 2 is the Path 17 protocol unchanged: "
        "gold-override weight λ ∈ {1, 2, 4, 8} swept cross-fitted on the immutable "
        "5-fold split of the 58 gold studies (`input/gold_folds.csv`, embedded "
        "in-notebook with a pinned sha256), per-target λ* = argmax cross-fitted "
        "gold AUC with a 0.02 noise floor falling back to λ = 2.0, and deployment "
        "heads at λ* trained with all 58 gold overridden on the parent's "
        "scanner-grouped splits. Every artifact records `gold_usage: crossfit`.\n"
    )

    source = "".join(notebook["cells"][32]["source"])
    source = replace_once(
        source,
        '"mode": "path6-kimi-fs160-clean-training-only",',
        '"mode": "path21-fused-labels-crossfit-training-only",',
        "audit mode",
    )
    fired += 1
    source = replace_once(
        source,
        '"gold_policy": "58 official rows excluded from optimisation, checkpoint selection, and blend selection; monitor only",',
        '"gold_policy": "stage 1 keeps the 58 official rows monitor-only on fused silver v5; stage 2 trains with gold override weight lambda under the GOLD_INTEGRATION_PLAN.md section 2 cross-fit protocol; selection uses cross-fitted gold OOF only",',
        "gold policy",
    )
    fired += 1
    source = replace_once(
        source, OLD_MAKE_TARGETS, NEW_MAKE_TARGETS, "Path21 labels + helpers"
    )
    fired += 1
    marker = '        log(f"Path6 Kimi FS160 gold monitor macro AUC {gold_auc:.5f} on 58 excluded studies")'
    source = replace_once(
        source, marker, marker + "\n\n" + SECOND_STAGE,
        "second-stage insertion",
    )
    fired += 1
    source = replace_once(source, OLD_PACKAGE, NEW_PACKAGE, "Path21 package")
    fired += 1
    source = replace_once(
        source,
        '            output / "v52_e11_heads.pt",',
        '            output / "path21_gold_heads.pt",',
        "head output name",
    )
    fired += 1
    source = replace_last_once(
        source,
        "oof_frame = pd.DataFrame(oof, columns=TARGETS)",
        "oof_frame = pd.DataFrame(path21_oof, columns=TARGETS)",
        "final OOF frame",
    )
    fired += 1
    source = replace_once(
        source,
        'oof_frame.to_csv(output / "v52_e11_oof.csv", index=False)',
        'oof_frame.to_csv(output / "path21_gold_oof.csv", index=False)',
        "OOF output name",
    )
    fired += 1
    source = replace_once(
        source,
        'audit["status"] = "PATH6_KIMI_FS160_TRAINED_PARENT_PRESERVED"',
        'audit["status"] = "PATH21_FUSED_LABELS_CROSSFIT_TRAINED"',
        "status",
    )
    fired += 1
    source = replace_once(
        source,
        'audit["heads_sha256"] = _v52_sha256(output / "v52_e11_heads.pt")',
        'audit["heads_sha256"] = _v52_sha256(output / "path21_gold_heads.pt")',
        "head hash",
    )
    fired += 1
    source = replace_once(
        source,
        'audit["oof_sha256"] = _v52_sha256(output / "v52_e11_oof.csv")',
        'audit["oof_sha256"] = _v52_sha256(output / "path21_gold_oof.csv")',
        "OOF hash",
    )
    fired += 1
    source = replace_once(
        source,
        'audit_path = Path("/kaggle/working/rad_e11_audit.json")',
        'audit_path = Path("/kaggle/working/path21_audit.json")',
        "audit output name",
    )
    fired += 1

    if fired != EXPECTED_MUTATIONS:
        raise RuntimeError(f"expected {EXPECTED_MUTATIONS} mutations, fired {fired}")

    notebook["cells"][32]["source"] = [line + "\n" for line in source.splitlines()]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")

    # Compile-check every code cell of the generated notebook.
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), f"{OUTPUT.name}:cell{index}", "exec")

    metadata = json.loads(METADATA_SOURCE.read_text())
    metadata["id"] = "aakashkavuru/rsna-knee-path21-fused-labels-train"
    metadata["title"] = "RSNA Knee Path21 Fused Labels Train"
    metadata["code_file"] = "rsna_knee_path21_fused_labels_train.ipynb"
    if "aakashkavuru/rsna-knee-silver-labels-v5" not in metadata["dataset_sources"]:
        metadata["dataset_sources"].append("aakashkavuru/rsna-knee-silver-labels-v5")
    METADATA_OUT.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells, {fired} mutations")
    print(f"wrote {METADATA_OUT}")


if __name__ == "__main__":
    main()
