from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path6_kimi_fs160_train" / "rsna_knee_path6_kimi_fs160_train.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path17_gold_crossfit_train.ipynb"
GOLD_FOLDS = ROOT.parent / "input" / "gold_folds.csv"

GOLD_FOLDS_CONTENT = GOLD_FOLDS.read_text()
GOLD_FOLDS_SHA256 = hashlib.sha256(GOLD_FOLDS_CONTENT.encode("utf-8")).hexdigest()
if '"""' in GOLD_FOLDS_CONTENT:
    raise RuntimeError("gold_folds.csv cannot be embedded as a triple-quoted constant")


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


HELPER_FUNCS = r'''
def _path17_load_gold_folds(train, gold, gold_rows):
    """Load the immutable 5-fold split of the 58 gold studies.

    The split ships inside this notebook as a sha256-pinned constant; a copy
    discovered under /kaggle/input is preferred only when its hash matches the
    pin. Any drift, schema change, or UID mismatch against the notebook's own
    gold mask is a hard fail.
    """
    from io import StringIO

    content = PATH17_GOLD_FOLDS_CSV
    source = "embedded-constant"
    try:
        candidate = find_input_file(PATH17_GOLD_FOLDS_FILE)
    except FileNotFoundError:
        pass
    else:
        if _v52_sha256(candidate) != PATH17_GOLD_FOLDS_SHA256:
            raise RuntimeError(f"gold_folds.csv hash drift at {candidate}")
        content = candidate.read_text()
        source = str(candidate)
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != PATH17_GOLD_FOLDS_SHA256:
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


def _path17_gold_override(y, weights, gold, gold_y, lam):
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
        raise RuntimeError("Path17 override touched a non-gold cell")
    if not np.array_equal(over_y[gold], gold_y.astype(np.float32)):
        raise RuntimeError("Path17 gold target override drift")
    if np.any(over_w[gold] != np.float32(lam)):
        raise RuntimeError("Path17 gold weight override drift")
    return over_y, over_w
'''


HELPERS = (
    "PATH17_LAMBDAS = (1.0, 2.0, 4.0, 8.0)\n"
    "PATH17_DEFAULT_LAMBDA = 2.0\n"
    "PATH17_NOISE_FLOOR = 0.02\n"
    'PATH17_GOLD_FOLDS_FILE = "gold_folds.csv"\n'
    f'PATH17_GOLD_FOLDS_SHA256 = "{GOLD_FOLDS_SHA256}"\n'
    'PATH17_GOLD_FOLDS_CSV = """' + GOLD_FOLDS_CONTENT + '"""\n'
    + HELPER_FUNCS.rstrip("\n")
)


SECOND_STAGE = r'''        gold_fold_of, gold_folds_source = _path17_load_gold_folds(train, gold, gold_rows)
        gfold = gold_fold_of[gold_rows]
        remaining = TIME_BUDGET - (time.time() - float(globals().get("T0", time.time())))
        if remaining < 3.0 * 3600:
            raise TimeoutError(f"only {remaining / 60:.1f} minutes remain for the Path17 sweep")
        log(f"Path17 gold folds from {gold_folds_source}; lambda sweep {PATH17_LAMBDAS}")

        path17_crossfit_pred = {}
        path17_deploy_oof = {}
        path17_deploy_folds = {}
        path17_per_lambda = {}
        for lam_index, lam in enumerate(PATH17_LAMBDAS):
            over_y, over_w = _path17_gold_override(y, weights, gold, gold_y, lam)
            if np.any(over_w[gold_rows] != np.float32(lam)):
                raise RuntimeError("Path17 gold row missing its override weight")

            # Honest gold evidence: gold fold k is absent from both the training
            # set and the early-stopping validation set of cross-fit model k.
            cross = np.full((len(gold_rows), len(TARGETS)), np.nan, dtype=np.float32)
            for k in range(5):
                tr_pos, va_pos = splits[k]
                tr = np.concatenate([non_gold[tr_pos], gold_rows[gfold != k]])
                va = non_gold[va_pos]
                held = gold_rows[gfold == k]
                if set(groups[non_gold[tr_pos]]).intersection(groups[va]):
                    raise RuntimeError(f"Path17 scanner leakage in crossfit fold {k}")
                if np.intersect1d(tr, held).size or np.intersect1d(va, held).size:
                    raise RuntimeError(f"held-out gold entered Path17 crossfit fold {k}")
                state, score = train_fold(
                    features, token_mask, over_y, over_w,
                    tr, va, 40 + 10 * lam_index + k, device
                )
                if state is None:
                    raise RuntimeError(
                        f"Path17 crossfit fold {k} (lambda={lam}) produced no checkpoint"
                    )
                head = FoundationQueryHead().to(device)
                head.load_state_dict(state, strict=True)
                cross[gfold == k] = predict_head(head, features, token_mask, held, device)
                del head, state
                torch.cuda.empty_cache()
            if not np.isfinite(cross).all():
                raise RuntimeError(f"Path17 cross-fitted gold OOF has NaN at lambda={lam}")
            path17_crossfit_pred[lam] = cross

            # Deployment set at this lambda: the parent's scanner-grouped splits,
            # silver cells byte-identical, all 58 gold rows overridden in training
            # only. Validation folds stay purely non-gold.
            deploy_oof = np.full_like(y, np.nan, dtype=np.float32)
            deploy_states = []
            for fold, (tr_pos, va_pos) in enumerate(splits):
                tr = np.concatenate([non_gold[tr_pos], gold_rows])
                va = non_gold[va_pos]
                if set(groups[non_gold[tr_pos]]).intersection(groups[va]):
                    raise RuntimeError(f"Path17 scanner leakage in deployment fold {fold}")
                if np.intersect1d(va, gold_rows).size:
                    raise RuntimeError(
                        f"gold row entered Path17 deployment validation fold {fold}"
                    )
                state, score = train_fold(
                    features, token_mask, over_y, over_w,
                    tr, va, 90 + 10 * lam_index + fold, device
                )
                if state is None:
                    raise RuntimeError(
                        f"Path17 deployment fold {fold} (lambda={lam}) produced no checkpoint"
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
                raise RuntimeError(f"incomplete Path17 deployment silver OOF at lambda={lam}")
            path17_deploy_oof[lam] = deploy_oof
            path17_deploy_folds[f"{lam:.1f}"] = deploy_states

            cross_target_auc = _v52_target_auc(gold_y, cross)
            deploy_weak_auc = macro_auc(y[non_gold], deploy_oof[non_gold], weights[non_gold])
            path17_per_lambda[f"{lam:.1f}"] = {
                "crossfit_gold_macro_auc": float(macro_auc(gold_y, cross)),
                "crossfit_gold_per_target_auc": {
                    target: float(value) for target, value in cross_target_auc.items()
                },
                "deployment_weak_oof_auc": float(deploy_weak_auc),
            }
            log(
                f"Path17 lambda={lam:.1f} cross-fitted gold macro AUC "
                f"{path17_per_lambda[f'{lam:.1f}']['crossfit_gold_macro_auc']:.5f}; "
                f"deployment weak OOF {deploy_weak_auc:.5f}"
            )

        # Per-target selection on the honest cross-fitted gold read only, with a
        # 0.02 noise floor (n=58 cannot resolve smaller gaps) and ties broken
        # toward the smallest lambda.
        lambda_star = {}
        lambda_star_basis = {}
        for target in TARGETS:
            observed = {
                lam: path17_per_lambda[f"{lam:.1f}"]["crossfit_gold_per_target_auc"].get(target)
                for lam in PATH17_LAMBDAS
            }
            observed = {lam: auc for lam, auc in observed.items() if auc is not None}
            if len(observed) < 2:
                lambda_star[target] = PATH17_DEFAULT_LAMBDA
                lambda_star_basis[target] = "insufficient_crossfit_evidence_default"
                continue
            spread = max(observed.values()) - min(observed.values())
            if spread < PATH17_NOISE_FLOOR:
                lambda_star[target] = PATH17_DEFAULT_LAMBDA
                lambda_star_basis[target] = f"noise_floor_spread_{spread:.4f}_default"
            else:
                best = max(observed.values())
                lambda_star[target] = min(
                    lam for lam, auc in observed.items() if auc == best
                )
                lambda_star_basis[target] = f"argmax_spread_{spread:.4f}"

        path17_oof = np.full_like(y, np.nan, dtype=np.float32)
        for index, target in enumerate(TARGETS):
            lam = lambda_star[target]
            path17_oof[non_gold, index] = path17_deploy_oof[lam][non_gold, index]
            path17_oof[gold_rows, index] = path17_crossfit_pred[lam][:, index]
        if not np.isfinite(path17_oof).all():
            raise RuntimeError("incomplete Path17 per-target lambda-star OOF")
        path17_weak_auc = macro_auc(y[non_gold], path17_oof[non_gold], weights[non_gold])
        path17_gold_auc = macro_auc(gold_y, path17_oof[gold_rows])
        audit["gold_usage"] = "crossfit"
        audit["gold_integration"] = {
            "protocol": (
                "GOLD_INTEGRATION_PLAN.md section 2 cross-fit; a gold study is "
                "never evaluated on a model whose training or early stopping saw it"
            ),
            "gold_folds_file": PATH17_GOLD_FOLDS_FILE,
            "gold_folds_sha256": PATH17_GOLD_FOLDS_SHA256,
            "gold_folds_source": gold_folds_source,
            "lambdas": [float(lam) for lam in PATH17_LAMBDAS],
            "noise_floor": PATH17_NOISE_FLOOR,
            "default_lambda": PATH17_DEFAULT_LAMBDA,
            "cv_grouping": "scanner-site DICOM signature",
            "parent_lambda0_gold_monitor_auc": float(gold_auc),
            "per_lambda": path17_per_lambda,
            "lambda_star": {target: float(lam) for target, lam in lambda_star.items()},
            "lambda_star_basis": lambda_star_basis,
            "lambda_star_weak_oof_auc": float(path17_weak_auc),
            "lambda_star_gold_crossfit_auc": float(path17_gold_auc),
            "deployment": (
                "per-target lambda-star; deployment heads trained with all 58 gold "
                "overridden on the parent's scanner-grouped splits; fold state_dicts "
                "for every lambda variant saved in path17_gold_heads.pt"
            ),
        }
        log(
            f"Path17 lambda-star weak OOF {path17_weak_auc:.5f}; "
            f"cross-fitted gold AUC {path17_gold_auc:.5f} "
            f"(parent gold-free monitor {gold_auc:.5f}); lambda_star={lambda_star}"
        )'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 17 — cross-fitted gold head training\n\n"
        "A gold-integration second stage on the proven Path 6 FS/160 mm pixel arm, under "
        "the cross-fit protocol of `GOLD_INTEGRATION_PLAN.md` §2. Stage 1 is the untouched "
        "gold-free parent (silver scanner-grouped OOF plus a monitor-only read on the 58 "
        "gold studies). Stage 2 sweeps a gold-override weight λ in {1, 2, 4, 8}: for each "
        "λ, heads are trained cross-fitted on the immutable 5-fold split of the 58 gold "
        "studies (`input/gold_folds.csv`, embedded in-notebook with a pinned sha256) to "
        "produce an honest 58-row gold OOF, and a full-gold deployment set is trained on "
        "the parent's scanner-grouped splits with all 58 gold rows overridden. Non-gold "
        "targets and weights stay byte-identical to the parent. Per target, λ* is the "
        "argmax of cross-fitted gold AUC with a 0.02 noise floor falling back to λ = 2.0. "
        "Every artifact records `gold_usage: crossfit`.\n"
    )
    source = "".join(notebook["cells"][32]["source"])
    source = replace_once(
        source,
        '"mode": "path6-kimi-fs160-clean-training-only",',
        '"mode": "path17-gold-crossfit-training-only",',
        "audit mode",
    )
    source = replace_once(
        source,
        '"gold_policy": "58 official rows excluded from optimisation, checkpoint selection, and blend selection; monitor only",',
        '"gold_policy": "stage 1 keeps the 58 official rows monitor-only; stage 2 trains with gold override weight lambda under the GOLD_INTEGRATION_PLAN.md section 2 cross-fit protocol; selection uses cross-fitted gold OOF only",',
        "gold policy",
    )
    source = replace_once(
        source,
        '    return y, w, gold\n\n\ndef report_groups',
        '    return y, w, gold\n\n\n' + HELPERS + "\n\n\ndef report_groups",
        "Path17 helpers",
    )
    marker = '        log(f"Path6 Kimi FS160 gold monitor macro AUC {gold_auc:.5f} on 58 excluded studies")'
    source = replace_once(
        source, marker, marker + "\n\n" + SECOND_STAGE,
        "second-stage insertion",
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
    new_package = '''                "version": "path17-gold-crossfit-1",
                "gold_training_or_selection_used": True,
                "gold_usage": "crossfit",
                "cv_grouping": "scanner-site DICOM signature",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "slots": [list(slot) for slot in E11_SLOTS],
                "crop_mm": E11_CROP_MM,
                "img": E11_IMG,
                "slices_per_plane": E11_CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": folds,
                "gold_fold_variants": path17_deploy_folds,
                "lambda_star": {target: float(lam) for target, lam in lambda_star.items()},
                "weak_oof_auc": float(weak_auc),
                "lambda_star_weak_oof_auc": float(path17_weak_auc),
                "target_policy": globals().get("PATH5_TARGET_AUDIT", {}),
                "gold_monitor_auc": float(gold_auc),'''
    source = replace_once(source, old_package, new_package, "Path17 package")
    source = replace_once(
        source,
        '            output / "v52_e11_heads.pt",',
        '            output / "path17_gold_heads.pt",',
        "head output name",
    )
    source = replace_last_once(
        source,
        "oof_frame = pd.DataFrame(oof, columns=TARGETS)",
        "oof_frame = pd.DataFrame(path17_oof, columns=TARGETS)",
        "final OOF frame",
    )
    source = replace_once(
        source,
        'oof_frame.to_csv(output / "v52_e11_oof.csv", index=False)',
        'oof_frame.to_csv(output / "path17_gold_oof.csv", index=False)',
        "OOF output name",
    )
    source = replace_once(
        source,
        'audit["status"] = "PATH6_KIMI_FS160_TRAINED_PARENT_PRESERVED"',
        'audit["status"] = "PATH17_GOLD_CROSSFIT_TRAINED"',
        "status",
    )
    source = replace_once(
        source,
        'audit["heads_sha256"] = _v52_sha256(output / "v52_e11_heads.pt")',
        'audit["heads_sha256"] = _v52_sha256(output / "path17_gold_heads.pt")',
        "head hash",
    )
    source = replace_once(
        source,
        'audit["oof_sha256"] = _v52_sha256(output / "v52_e11_oof.csv")',
        'audit["oof_sha256"] = _v52_sha256(output / "path17_gold_oof.csv")',
        "OOF hash",
    )
    notebook["cells"][32]["source"] = [line + "\n" for line in source.splitlines()]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
