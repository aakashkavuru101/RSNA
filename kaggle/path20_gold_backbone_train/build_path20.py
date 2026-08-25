"""Codegen for the Path20 gold-weighted full-backbone kernel.

Clones kaggle/clean_dino_train/clean_dino_train.py and applies guarded
replace_once mutations (same pattern as build_path17.py): every mutation must
match the parent text exactly once or the build fails loudly. The result is a
kernel whose stage 0 is the untouched gold-free parent pass (silver OOF
baseline + monitor-only gold read + submission), followed by a lambda-swept
cross-fit stage that trains the FULL backbone on silver + gold under the
GOLD_INTEGRATION_PLAN.md section 2 protocol, a per-target lambda* selection,
and a full-gold deployment stage that writes path20_gold_backbone_fold{0..4}.pt.
It also carries six SMOKE-only accommodations (each a strict no-op when
SMOKE=False) that keep the parent's smoke run from tripping guards written
for the 5-fold full configuration.
"""

from __future__ import annotations

import hashlib
import json
import py_compile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "kaggle" / "clean_dino_train" / "clean_dino_train.py"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUTPUT_DIR / "path20_gold_backbone_train.py"
METADATA_SOURCE = REPO / "kaggle" / "clean_dino_train" / "kernel-metadata.json"
METADATA_OUT = OUTPUT_DIR / "kernel-metadata.json"
GOLD_FOLDS = REPO / "input" / "gold_folds.csv"

GOLD_FOLDS_CONTENT = GOLD_FOLDS.read_text()
GOLD_FOLDS_SHA256 = hashlib.sha256(GOLD_FOLDS_CONTENT.encode("utf-8")).hexdigest()
if '"""' in GOLD_FOLDS_CONTENT:
    raise RuntimeError("gold_folds.csv cannot be embedded as a triple-quoted constant")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return source.replace(old, new, 1)


NEW_DOCSTRING = '''"""Path20 — gold-weighted full-backbone DINOv2 fine-tuning for RSNA Knee MRI.

Stage 0 is the untouched gold-free parent pass (clean_dino_train): silver
scanner-grouped OOF, honest gold monitor, blend regression guard, submission.
Stage 1 cross-fits the FULL backbone on silver + gold with binary gold targets
at per-cell loss weight lambda in PATH20_LAMBDAS over the immutable 5-fold
split of the 58 gold studies (input/gold_folds.csv, embedded with a pinned
sha256), producing an honest cross-fitted gold OOF and a matched silver OOF
per lambda. Stage 2 selects a per-target lambda* (argmax cross-fitted gold
AUC, 0.02 noise floor, silver-OOF non-regression >= -0.005 vs stage 0,
default 2.0) and trains the deployment folds on the parent's scanner splits
with all 58 gold overridden. GOLD_INTEGRATION_PLAN.md section 2: a gold study
is never evaluated on a model whose training or early stopping saw it; every
artifact records gold_usage.
"""'''


CONSTANTS_BLOCK = (
    "BLEND_WEIGHTS = np.linspace(0.0, 1.0, 11)\n\n"
    "# --- Path20: gold-weighted full-backbone fine-tuning ---\n"
    "# Single top-level pointer to the silver labels CSV; a rerun on a newer\n"
    "# silver_labels_v4.csv is a one-line change here (plus its labels dataset).\n"
    'LABELS_FILE = "llm_labels_v4_blend.csv"\n'
    "PATH20_LAMBDAS = (1, 2, 4, 8)\n"
    "PATH20_LAMBDAS_SMOKE = (2,)\n"
    "PATH20_NOISE_FLOOR = 0.02\n"
    "PATH20_DEFAULT_LAMBDA = 2.0\n"
    'PATH20_GOLD_FOLDS_FILE = "gold_folds.csv"\n'
    f'PATH20_GOLD_FOLDS_SHA256 = "{GOLD_FOLDS_SHA256}"\n'
    "# Soft stop: finish the current fold, then write a partial audit.\n"
    "PATH20_FOLD_GUARD_S = 0.45 * 3600\n"
    'PATH20_GOLD_FOLDS_CSV = """' + GOLD_FOLDS_CONTENT + '"""\n'
)


OLD_FIND_LABELS = '''def find_v4_labels() -> Path:
    exact = []
    for root, directories, files in os.walk("/kaggle/input"):
        # Never recursively index the raw DICOM trees just to locate a small CSV.
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if "llm_labels_v4_blend.csv" in files:
            exact.append(Path(root) / "llm_labels_v4_blend.csv")
    if len(exact) != 1:
        raise FileNotFoundError(
            f"Expected exactly one llm_labels_v4_blend.csv, found {len(exact)}"
        )
    return exact[0]'''

NEW_FIND_LABELS = '''def find_v4_labels() -> Path:
    exact = []
    for root, directories, files in os.walk("/kaggle/input"):
        # Never recursively index the raw DICOM trees just to locate a small CSV.
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if LABELS_FILE in files:
            exact.append(Path(root) / LABELS_FILE)
    if len(exact) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {LABELS_FILE}, found {len(exact)}"
        )
    return exact[0]'''


HELPERS = r'''def _path20_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path20_load_gold_folds(studies, table):
    """Load the immutable 5-fold split of the 58 gold studies.

    The split ships inside this kernel as a sha256-pinned constant; a copy
    discovered under /kaggle/input is used only when its hash matches the pin.
    Any drift, schema change, or UID mismatch against the kernel's own gold
    mask is a hard fail.
    """
    from io import StringIO

    content = PATH20_GOLD_FOLDS_CSV
    source = "embedded-constant"
    candidates = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if PATH20_GOLD_FOLDS_FILE in files:
            candidates.append(Path(root) / PATH20_GOLD_FOLDS_FILE)
    if candidates:
        matching = [c for c in candidates if _path20_sha256(c) == PATH20_GOLD_FOLDS_SHA256]
        if not matching:
            raise RuntimeError("gold_folds.csv under /kaggle/input failed its pinned sha256")
        content = matching[0].read_text()
        source = str(matching[0])
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != PATH20_GOLD_FOLDS_SHA256:
        raise RuntimeError("gold_folds content failed its pinned sha256")
    frame = pd.read_csv(StringIO(content), dtype={"StudyInstanceUID": str})
    if frame.columns.tolist() != ["StudyInstanceUID", "gold_fold"] or len(frame) != 58:
        raise RuntimeError("gold_folds.csv is malformed")
    if frame.StudyInstanceUID.duplicated().any():
        raise RuntimeError("gold_folds.csv has a duplicated study")
    gold_mask = table["is_gold"].values
    mask_uids = set(table.index.astype(str)[gold_mask])
    if set(frame.StudyInstanceUID.astype(str)) != mask_uids:
        raise RuntimeError("gold_folds.csv studies differ from the kernel gold mask")
    fold_map = dict(zip(frame.StudyInstanceUID.astype(str), frame.gold_fold.astype(int)))
    gold_fold_of = np.array(
        [fold_map.get(str(study), -1) for study in studies], dtype=np.int64
    )
    if (gold_fold_of[~gold_mask] != -1).any():
        raise RuntimeError("gold_folds.csv assigns a fold to a non-gold row")
    picked = gold_fold_of[gold_mask]
    if (picked < 0).any() or not np.isin(picked, np.arange(5)).all():
        raise RuntimeError("gold_folds.csv fold outside 0..4")
    if sorted(np.unique(picked).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("gold_folds.csv does not cover all five folds")
    return gold_fold_of, source


def _path20_gold_override(targets, confidence, gold_indices, gold_binary, lam):
    """Gold override: binary gold targets, per-cell loss weight x lam, on the
    58 gold rows only. lam is a scalar (cross-fit sweep) or a per-target
    vector (deployment at lambda*). Non-gold (target, weight) cells stay
    byte-identical to the parent, which is asserted rather than assumed.
    """
    lam_array = np.broadcast_to(np.asarray(lam, dtype=np.float32), (len(TARGETS),))
    gold_base = confidence[gold_indices]
    if not np.isfinite(gold_binary[gold_indices]).all():
        raise RuntimeError("Path20 gold targets contain missing cells")
    if not np.isin(gold_binary[gold_indices], [0.0, 1.0]).all():
        raise RuntimeError("Path20 gold targets are not binary")
    over_y = np.asarray(targets, dtype=np.float32).copy()
    over_w = np.asarray(confidence, dtype=np.float32).copy()
    over_y[gold_indices] = gold_binary[gold_indices].astype(np.float32)
    over_w[gold_indices] = gold_base * lam_array
    gold_mask = np.zeros(len(over_y), dtype=bool)
    gold_mask[gold_indices] = True
    if not (
        np.array_equal(over_y[~gold_mask], targets[~gold_mask], equal_nan=True)
        and np.array_equal(over_w[~gold_mask], confidence[~gold_mask], equal_nan=True)
    ):
        raise RuntimeError("Path20 override touched a non-gold cell")
    if not np.array_equal(over_y[gold_indices], gold_binary[gold_indices].astype(np.float32)):
        raise RuntimeError("Path20 gold target override drift")
    if not np.array_equal(over_w[gold_indices], gold_base * lam_array):
        raise RuntimeError("Path20 gold weight override drift")
    return over_y, over_w


def rank_predictions(predictions: np.ndarray) -> np.ndarray:'''


OLD_TRAIN_SIGNATURE = (
    "def train_model(model, cache, mask, table, train_indices, val_indices, gold_indices, device):"
)
NEW_TRAIN_SIGNATURE = (
    "def train_model(\n"
    "    model, cache, mask, table, train_indices, val_indices, gold_indices, device,\n"
    "    label_override=None,\n"
    "):"
)

OLD_TRAIN_LABELS = '''    targets = table[TARGETS].values.astype(np.float32)
    confidence = 0.25 + 0.75 * np.abs(targets - 0.5) * 2.0
    gold_columns = [f"{target}__gold" for target in TARGETS]'''
NEW_TRAIN_LABELS = '''    targets = table[TARGETS].values.astype(np.float32)
    confidence = 0.25 + 0.75 * np.abs(targets - 0.5) * 2.0
    if label_override is not None:
        # Path20: (targets, confidence) with binary gold targets at weight x
        # lambda on gold rows; silver cells byte-identical (asserted upstream).
        targets, confidence = label_override
    gold_columns = [f"{target}__gold" for target in TARGETS]'''


TAIL_ANCHOR = '''    write_submission(test_ensemble, test_studies, test_df)
    log(f"complete in {(time.time() - T0) / 3600:.2f} hours")'''


SECOND_STAGE = r'''
    # ================= Path20: gold-weighted full-backbone stages =================
    # Stage 0 above is the untouched gold-free parent pass and stays the
    # comparison baseline and honest gold monitor. Stage 1 cross-fits the full
    # backbone on silver + gold at each override weight lambda; stage 2 selects
    # a per-target lambda* and trains the deployment folds with all 58 gold
    # overridden.
    log("Path20: starting gold-weighted backbone cross-fit stage")
    path20_lambdas = PATH20_LAMBDAS_SMOKE if SMOKE else PATH20_LAMBDAS
    n_gold_folds = 2 if SMOKE else 5
    gold_fold_of, gold_folds_source = _path20_load_gold_folds(studies, table)
    gold_binary = table[gold_columns].values.astype(np.float32)
    label_sha256 = _path20_sha256(label_path)
    base_targets = table[TARGETS].values.astype(np.float32)
    base_confidence = 0.25 + 0.75 * np.abs(base_targets - 0.5) * 2.0
    path20_status = "PATH20_GOLD_BACKBONE_TRAINED"
    crossfit_gold_pred = {}
    crossfit_silver_oof = {}
    crossfit_gold_counts = {}
    per_lambda_report = {}
    lambda_star = {}
    lambda_star_basis = {}
    lambda_star_oof_key = {}
    deploy_oof = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)
    deploy_checkpoints = {}
    deploy_per_target = {}
    stage0_per_target = {}
    oof_written = False
    path20_audit = {
        "parent": "clean_dino_train",
        "mode": "path20-gold-backbone-train",
        "smoke": SMOKE,
        "label_source": LABELS_FILE,
        "label_source_path": str(label_path),
        "label_source_sha256": label_sha256,
        "gold_folds_file": PATH20_GOLD_FOLDS_FILE,
        "gold_folds_sha256": PATH20_GOLD_FOLDS_SHA256,
        "gold_folds_source": gold_folds_source,
        "lambdas": [float(lam) for lam in path20_lambdas],
        "noise_floor": PATH20_NOISE_FLOOR,
        "default_lambda": PATH20_DEFAULT_LAMBDA,
        "protocol": (
            "GOLD_INTEGRATION_PLAN.md section 2 cross-fit; a gold study is never "
            "evaluated on a model whose training or early stopping saw it"
        ),
        "gold_usage": {
            "stage0_baseline": "none",
            "crossfit_selection": "crossfit",
            "deployment_artifacts": "full",
        },
        "stage0_parent": {
            "gold_training_studies": 0,
            "silver_oof_macro_auc": float(new_oof_auc),
            "blend_gold_monitor_auc": float(gold_auc),
        },
    }
    try:
        # ---- Stage 1: one honest cross-fitted gold OOF per lambda ----
        for lam_index, lam in enumerate(path20_lambdas):
            lam_key = f"{float(lam):.1f}"
            over_targets, over_confidence = _path20_gold_override(
                base_targets, base_confidence, gold_indices, gold_binary, lam
            )
            cross_gold = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)
            cross_silver = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)
            for k in range(n_gold_folds):
                if time.time() - T0 > TIME_LIMIT_S - PATH20_FOLD_GUARD_S:
                    raise TimeoutError(
                        f"Path20 cross-fit paused before lambda={lam_key} gold fold {k}"
                    )
                seed_everything(SEED + 1000 + 10 * lam_index + k)
                _, tr_pos, va_pos, _, _ = make_clean_split(studies, labels, groups, k)
                held = gold_indices[gold_fold_of[gold_indices] == k]
                trained_gold = gold_indices[gold_fold_of[gold_indices] != k]
                # Scanner split k supplies the silver train/val; gold folds != k
                # join training only, gold fold k is fully held out (train AND
                # early-stopping val).
                tr = np.concatenate([tr_pos, trained_gold])
                va = va_pos
                if np.intersect1d(tr, held).size or np.intersect1d(va, held).size:
                    raise RuntimeError(f"held-out gold entered Path20 crossfit fold {k}")
                if np.intersect1d(va, gold_indices).size:
                    raise RuntimeError(f"gold entered Path20 crossfit validation fold {k}")
                model = build_model(dino_path)
                model, state, history = train_model(
                    model, train_cache, train_mask, table, tr, va, held, device,
                    label_override=(over_targets, over_confidence),
                )
                cross_gold[held] = predict(model, train_cache, train_mask, held, device)
                cross_silver[va] = predict(model, train_cache, train_mask, va, device)
                crossfit_gold_counts[f"lambda={lam_key}/gold_fold={k}"] = int(len(trained_gold))
                log(
                    f"Path20 crossfit lambda={lam_key} gold fold {k}: "
                    f"train={len(tr)} ({len(trained_gold)} gold), val={len(va)}, "
                    f"held={len(held)}"
                )
                del model, state
                gc.collect()
                torch.cuda.empty_cache()
            crossfit_gold_pred[lam_key] = cross_gold
            crossfit_silver_oof[lam_key] = cross_silver

        if not crossfit_gold_pred:
            raise TimeoutError("Path20: no lambda completed the cross-fit stage")

        # ---- Per-target lambda* selection on the honest cross-fitted read ----
        stage0_mask = np.zeros(len(studies), dtype=bool)
        stage0_mask[eligible] = np.isfinite(new_oof[eligible]).all(axis=1)
        if not stage0_mask.any():
            raise RuntimeError("Path20 stage-0 baseline OOF is empty")
        _, stage0_per_target = macro_auc(
            (targets[stage0_mask] > 0.5).astype(int), new_oof[stage0_mask]
        )
        for lam_key, cross_gold in crossfit_gold_pred.items():
            covered_gold = gold_indices[np.isfinite(cross_gold[gold_indices]).all(axis=1)]
            covered_silver = np.flatnonzero(
                stage0_mask & np.isfinite(crossfit_silver_oof[lam_key]).all(axis=1)
            )
            gold_macro, gold_auc_map = macro_auc(
                gold_binary[covered_gold].astype(int), cross_gold[covered_gold]
            )
            silver_macro, silver_auc_map = macro_auc(
                (targets[covered_silver] > 0.5).astype(int),
                crossfit_silver_oof[lam_key][covered_silver],
            )
            per_lambda_report[lam_key] = {
                "crossfit_gold_macro_auc": float(gold_macro),
                "crossfit_gold_per_target_auc": gold_auc_map,
                "crossfit_gold_studies": int(len(covered_gold)),
                "silver_oof_macro_auc": float(silver_macro),
                "silver_oof_per_target_auc": silver_auc_map,
                "silver_oof_studies": int(len(covered_silver)),
                "silver_oof_delta_vs_stage0": {
                    target: float(silver_auc_map[target] - stage0_per_target[target])
                    for target in TARGETS
                    if target in silver_auc_map and target in stage0_per_target
                },
            }
            log(
                f"Path20 lambda={lam_key}: cross-fitted gold macro AUC {gold_macro:.5f} "
                f"on {len(covered_gold)} studies; matched silver OOF {silver_macro:.5f}"
            )

        for target in TARGETS:
            observed = {
                lam_key: report["crossfit_gold_per_target_auc"].get(target)
                for lam_key, report in per_lambda_report.items()
            }
            observed = {key: value for key, value in observed.items() if value is not None}
            passing = {
                key: value
                for key, value in observed.items()
                if per_lambda_report[key]["silver_oof_delta_vs_stage0"].get(target, -1.0)
                >= -0.005
            }
            pool = passing if passing else observed
            gate = "" if passing else "/silver_gate_no_passing_lambda"
            if len(observed) < 2:
                lambda_star[target] = PATH20_DEFAULT_LAMBDA
                lambda_star_basis[target] = "insufficient_crossfit_evidence_default" + gate
                continue
            spread = max(pool.values()) - min(pool.values())
            if spread < PATH20_NOISE_FLOOR:
                lambda_star[target] = PATH20_DEFAULT_LAMBDA
                lambda_star_basis[target] = f"noise_floor_spread_{spread:.4f}_default" + gate
            else:
                best = max(pool.values())
                lambda_star[target] = float(min(
                    float(key) for key, value in pool.items() if value == best
                ))
                lambda_star_basis[target] = f"argmax_spread_{spread:.4f}" + gate

        # ---- Stage 2: deployment on the parent's scanner splits, all 58 gold
        # overridden at the per-target lambda* vector. The loss is per-cell
        # weighted, so one backbone run carries a distinct lambda per target.
        lam_vector = np.array([lambda_star[target] for target in TARGETS], dtype=np.float32)
        over_targets, over_confidence = _path20_gold_override(
            base_targets, base_confidence, gold_indices, gold_binary, lam_vector
        )
        for fold in range(run_folds):
            if time.time() - T0 > TIME_LIMIT_S - PATH20_FOLD_GUARD_S:
                raise TimeoutError(f"Path20 deployment paused before fold {fold}")
            seed_everything(SEED + 2000 + fold)
            _, tr_pos, va_pos, _, _ = make_clean_split(studies, labels, groups, fold)
            tr = np.concatenate([tr_pos, gold_indices])
            va = va_pos
            if np.intersect1d(va, gold_indices).size:
                raise RuntimeError(f"gold entered Path20 deployment validation fold {fold}")
            model = build_model(dino_path)
            model, state, history = train_model(
                model, train_cache, train_mask, table, tr, va, gold_indices, device,
                label_override=(over_targets, over_confidence),
            )
            checkpoint_name = (
                "path20_gold_backbone_smoke.pt" if SMOKE
                else f"path20_gold_backbone_fold{fold}.pt"
            )
            torch.save({
                "state_dict": state,
                "targets": TARGETS,
                "slots": SLOTS,
                "img_size": IMG_SIZE,
                "crop_mm": CROP_MM,
                "slice_band": SLICE_BAND,
                "group_size": GROUP_SIZE,
                "n_groups": N_GROUPS,
                "cache_slices": CACHE_SLICES,
                "fold": fold,
                "train_studies": [studies[i] for i in tr],
                "validation_studies": [studies[i] for i in va],
                "gold_training_count": int(len(gold_indices)),
                "gold_usage": "full",
                "gold_lambda_per_target": {
                    target: float(value) for target, value in lambda_star.items()
                },
                "label_source": LABELS_FILE,
                "label_source_sha256": label_sha256,
                "parent": "clean_dino_train",
                "smoke": SMOKE,
            }, checkpoint_name)
            deploy_checkpoints[checkpoint_name] = _path20_sha256(Path(checkpoint_name))
            deploy_oof[va] = predict(model, train_cache, train_mask, va, device)
            log(f"Path20 deployment fold {fold}: saved {checkpoint_name}")
            del model, state
            gc.collect()
            torch.cuda.empty_cache()

        # ---- OOF export: deployment silver OOF + honest cross-fitted gold ----
        deploy_covered = eligible[np.isfinite(deploy_oof[eligible]).all(axis=1)]
        _, deploy_per_target = macro_auc(
            (targets[deploy_covered] > 0.5).astype(int), deploy_oof[deploy_covered]
        )
        path20_oof = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)
        path20_oof[deploy_covered] = deploy_oof[deploy_covered]
        for index, target in enumerate(TARGETS):
            star_key = f"{lambda_star[target]:.1f}"
            if star_key not in crossfit_gold_pred:
                star_key = next(iter(crossfit_gold_pred))
            lambda_star_oof_key[target] = star_key
            covered_gold = gold_indices[
                np.isfinite(crossfit_gold_pred[star_key][gold_indices, index])
            ]
            path20_oof[covered_gold, index] = crossfit_gold_pred[star_key][covered_gold, index]
        fold_of = np.full(len(studies), -1, dtype=np.int64)
        for fold in range(run_folds):
            _, _, va_pos, _, _ = make_clean_split(studies, labels, groups, fold)
            fold_of[va_pos] = fold
        fold_of[gold_indices] = gold_fold_of[gold_indices]
        oof_frame = pd.DataFrame(path20_oof, columns=TARGETS)
        oof_frame.insert(0, "StudyInstanceUID", np.asarray(studies, dtype=str))
        oof_frame["fold"] = fold_of
        oof_frame["is_gold"] = table["is_gold"].values.astype(int)
        complete_rows = np.isfinite(path20_oof).all(axis=1)
        if not complete_rows.all():
            if not SMOKE:
                raise RuntimeError("Path20 OOF has uncovered studies")
            oof_frame = oof_frame.loc[complete_rows]
        oof_frame.to_csv("path20_gold_oof.csv", index=False)
        oof_written = True
        path20_audit["crossfit"] = {
            "gold_training_counts": crossfit_gold_counts,
            "per_lambda": per_lambda_report,
        }
        path20_audit["lambda_star"] = {
            target: float(value) for target, value in lambda_star.items()
        }
        path20_audit["lambda_star_basis"] = lambda_star_basis
        path20_audit["lambda_star_oof_key"] = lambda_star_oof_key
        path20_audit["deployment"] = {
            "gold_training_count_per_fold": int(len(gold_indices)),
            "gold_monitor_note": (
                "per-epoch gold reads inside deployment runs are in-sample "
                "(trained-on) and are never used for checkpoint selection"
            ),
            "checkpoints": deploy_checkpoints,
            "silver_oof_per_target_auc": deploy_per_target,
            "silver_oof_delta_vs_stage0": {
                target: float(deploy_per_target[target] - stage0_per_target[target])
                for target in TARGETS
                if target in deploy_per_target and target in stage0_per_target
            },
        }
        path20_audit["artifacts"] = dict(deploy_checkpoints)
        path20_audit["artifacts"]["path20_gold_oof.csv"] = _path20_sha256(
            Path("path20_gold_oof.csv")
        )
    except TimeoutError as exc:
        # Time guard: finish the current fold, keep every completed artifact,
        # and write a partial audit instead of dying silently.
        path20_status = "PATH20_PARTIAL_TIMEOUT"
        path20_audit["partial_reason"] = str(exc)
        path20_audit["crossfit"] = {
            "gold_training_counts": crossfit_gold_counts,
            "completed_lambdas": sorted(crossfit_gold_pred),
            "per_lambda": per_lambda_report,
        }
        path20_audit["lambda_star"] = {
            target: float(value) for target, value in lambda_star.items()
        }
        path20_audit["lambda_star_basis"] = lambda_star_basis
        path20_audit["deployment"] = {"checkpoints": deploy_checkpoints}
        path20_audit["artifacts"] = dict(deploy_checkpoints)
        if oof_written:
            path20_audit["artifacts"]["path20_gold_oof.csv"] = _path20_sha256(
                Path("path20_gold_oof.csv")
            )
        log(f"Path20 stopped early with a partial audit: {exc}")
    path20_audit["status"] = path20_status
    path20_audit["elapsed_seconds"] = time.time() - T0
    Path("path20_audit.json").write_text(json.dumps(path20_audit, indent=2, default=float))
    log(f"Path20 audit written: status={path20_status}, lambda_star={lambda_star}")'''


EXPECTED_MUTATIONS = 14


def main() -> None:
    source = SOURCE.read_text()
    fired = 0

    old_docstring = source.split('"""', 2)[1]
    source = replace_once(source, '"""' + old_docstring + '"""', NEW_DOCSTRING, "docstring")
    fired += 1

    source = replace_once(
        source,
        "TIME_LIMIT_S = 7.8 * 3600",
        "TIME_LIMIT_S = 11.0 * 3600  # Path20: stage0 + 20 crossfit + 5 deployment runs",
        "time limit",
    )
    fired += 1

    source = replace_once(
        source,
        "BLEND_WEIGHTS = np.linspace(0.0, 1.0, 11)",
        CONSTANTS_BLOCK.rstrip("\n"),
        "Path20 constants",
    )
    fired += 1

    source = replace_once(source, OLD_FIND_LABELS, NEW_FIND_LABELS, "labels file constant")
    fired += 1

    source = replace_once(
        source, OLD_TRAIN_SIGNATURE, NEW_TRAIN_SIGNATURE, "train_model signature"
    )
    fired += 1

    source = replace_once(source, OLD_TRAIN_LABELS, NEW_TRAIN_LABELS, "label override hook")
    fired += 1

    source = replace_once(
        source,
        "def rank_predictions(predictions: np.ndarray) -> np.ndarray:",
        HELPERS,
        "Path20 helpers",
    )
    fired += 1

    source = replace_once(
        source, TAIL_ANCHOR, TAIL_ANCHOR + "\n" + SECOND_STAGE, "second stage"
    )
    fired += 1

    # --- SMOKE-only accommodations: every edit below is a no-op when SMOKE is
    # False (the full-run behavior stays byte-identical in effect) and only
    # keeps the parent's smoke run from tripping guards that assume 5 folds.

    # Site 1: baseline 224 px loop expected available_folds == run_folds (1 in
    # SMOKE, but make_clean_split still builds 5 scanner folds).
    source = replace_once(
        source,
        "        if available_folds != run_folds:\n"
        '            raise ValueError(f"Expected {run_folds} scanner folds, found {available_folds}")',
        "        if not SMOKE and available_folds != run_folds:\n"
        '            raise ValueError(f"Expected {run_folds} scanner folds, found {available_folds}")',
        "smoke guard: baseline fold count",
    )
    fired += 1

    # Site 2: baseline OOF full-coverage check assumes all 5 folds ran.
    source = replace_once(
        source,
        "    if not np.isfinite(baseline_oof[eligible]).all():\n"
        '        raise RuntimeError("Baseline OOF predictions do not cover every non-gold study")',
        "    if not SMOKE and not np.isfinite(baseline_oof[eligible]).all():\n"
        '        raise RuntimeError("Baseline OOF predictions do not cover every non-gold study")',
        "smoke guard: baseline OOF coverage",
    )
    fired += 1

    # Site 3: new-model OOF full-coverage check assumes all 5 folds ran.
    source = replace_once(
        source,
        "    if not np.isfinite(new_oof[eligible]).all():\n"
        '        raise RuntimeError("New OOF predictions do not cover every non-gold study")',
        "    if not SMOKE and not np.isfinite(new_oof[eligible]).all():\n"
        '        raise RuntimeError("New OOF predictions do not cover every non-gold study")',
        "smoke guard: new OOF coverage",
    )
    fired += 1

    # Site 4: the blend AUCs would see NaN ranks on uncovered rows in SMOKE.
    # Masking to covered rows is a strict no-op in full runs because sites 2/3
    # then guarantee complete coverage.
    source = replace_once(
        source,
        "    oof_truth = (targets[eligible] > 0.5).astype(int)\n"
        "    baseline_oof_rank = rank_predictions(baseline_oof[eligible])\n"
        "    new_oof_rank = rank_predictions(new_oof[eligible])",
        "    covered_eligible = eligible[\n"
        "        np.isfinite(baseline_oof[eligible]).all(axis=1)\n"
        "        & np.isfinite(new_oof[eligible]).all(axis=1)\n"
        "    ]  # SMOKE covers only fold-0 val rows; full runs cover every row\n"
        "    oof_truth = (targets[covered_eligible] > 0.5).astype(int)\n"
        "    baseline_oof_rank = rank_predictions(baseline_oof[covered_eligible])\n"
        "    new_oof_rank = rank_predictions(new_oof[covered_eligible])",
        "smoke guard: blend coverage mask",
    )
    fired += 1

    # Site 5: the MIN_OOF_GAIN submission refusal is a production gate; on a
    # one-fold smoke sample it would fire spuriously and block the smoke run
    # from exercising the submission path.
    source = replace_once(
        source,
        "    if best_oof_auc < baseline_oof_auc + MIN_OOF_GAIN:",
        "    if not SMOKE and best_oof_auc < baseline_oof_auc + MIN_OOF_GAIN:",
        "smoke guard: MIN_OOF_GAIN refusal",
    )
    fired += 1

    # Site 6: load_baseline_model expects n_groups == N_GROUPS (1 in SMOKE),
    # but the attached baseline checkpoints come from the prior kernel's full
    # run, which recorded n_groups=2. Channel count is unchanged (GROUP_SIZE
    # is fixed), so loading them in SMOKE is sound. No-op when SMOKE is False
    # because N_GROUPS == 2 there.
    source = replace_once(
        source,
        '        "group_size": GROUP_SIZE,\n'
        '        "n_groups": N_GROUPS,\n'
        "    }",
        '        "group_size": GROUP_SIZE,\n'
        '        "n_groups": N_GROUPS if not SMOKE else 2,\n'
        "    }",
        "smoke guard: baseline checkpoint n_groups",
    )
    fired += 1

    if fired != EXPECTED_MUTATIONS:
        raise RuntimeError(f"expected {EXPECTED_MUTATIONS} mutations, fired {fired}")

    OUTPUT.write_text(source)
    py_compile.compile(str(OUTPUT), doraise=True)

    metadata = json.loads(METADATA_SOURCE.read_text())
    metadata["id"] = "aakashkavuru/rsna-knee-path20-gold-backbone-train"
    metadata["title"] = "RSNA Knee Path20 Gold Backbone Train"
    metadata["code_file"] = "path20_gold_backbone_train.py"
    METADATA_OUT.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {OUTPUT} ({len(source.splitlines())} lines, {fired} mutations)")
    print(f"wrote {METADATA_OUT}")


if __name__ == "__main__":
    main()
