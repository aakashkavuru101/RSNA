from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path13_aman_super_0920" / "rsna_knee_path13_aman_super_0920.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path21_strong_alpha_apply.ipynb"
METADATA_SOURCE = ROOT / "path18_path13_goldstar_apply" / "kernel-metadata.json"
METADATA_OUT = Path(__file__).resolve().parent / "kernel-metadata.json"
LABELS_V5 = ROOT.parent / "input" / "silver_labels_v5.csv"
GOLD_FOLDS = ROOT.parent / "input" / "gold_folds.csv"

LABELS_V5_SHA256 = hashlib.sha256(LABELS_V5.read_bytes()).hexdigest()
GOLD_FOLDS_SHA256 = hashlib.sha256(GOLD_FOLDS.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return source.replace(old, new, 1)


def require_once(source: str, needle: str, label: str) -> None:
    count = source.count(needle)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")


OLD_TITLE = "# Path 13 — Aman Super Ensemble 0920"
NEW_TITLE = "# Path 21 — Path13 super-0920 parent + Path21 fused-labels λ* heads at strong α"

OLD_BODY = "Fork of `amanatar/rsna-knee-super-ensemble-0920` with an appended audit gate."
NEW_BODY = (
    "Exact clone of the Path13 fork of `amanatar/rsna-knee-super-ensemble-0920` "
    "(public LB 0.920), including its appended audit gate, plus one final stage: a "
    "per-target rank blend of the Path21 fused-silver-v5 gold-crossfit deployment "
    "heads at each target's λ*. Alphas are computed in-kernel from the Path21 "
    "train audit's cross-fitted gold evidence: α = 0.50 where the λ* cross-fitted "
    "gold AUC beats the gold-free parent monitor by >0.02, α = 0.34 where the "
    "gain is in [0.005, 0.02], α = 0 (pure parent) otherwise — on a 3-study test "
    "only α > 1/3 can change an ordering, so smaller alphas are never spent. "
    "Heads and train audit are located by filename from the "
    "`rsna-knee-path21-fused-labels-train` kernel output and receipt-checked "
    "(pinned silver-v5 and gold-folds sha256, heads sha256 from the train audit). "
    "Any guard failure restores the preserved Path13 submission byte-for-byte."
)


FINAL_CELL = (
    r'''# Path 21 finalizer: Path13 Aman super-0920 parent + Path21 fused-labels lambda-star
# gold heads at strong, gold-evidence-gated alphas.
import hashlib as _p21_hashlib
import json as _p21_json
import shutil as _p21_shutil
from pathlib import Path as _P21Path

_P21_HEADS_NAME = "path21_gold_heads.pt"
_P21_TRAIN_AUDIT_NAME = "path21_audit.json"
_P21_BUNDLE_VERSION = "path21-fused-labels-crossfit-1"
_P21_LABELS_FILE = "silver_labels_v5.csv"
'''
    f'_P21_LABELS_SHA256 = "{LABELS_V5_SHA256}"\n'
    f'_P21_GOLD_FOLDS_SHA256 = "{GOLD_FOLDS_SHA256}"\n'
    r'''_P21_ALPHA_STRONG = 0.50
_P21_ALPHA_MODERATE = 0.34
_P21_GAIN_STRONG = 0.02
_P21_GAIN_MODERATE = 0.005
_P21_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("COR_NOFS", "Coronal", None, False),
]


def _p21_sha256(path):
    return _p21_hashlib.sha256(_P21Path(path).read_bytes()).hexdigest()


def _p21_find_unique(name):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            hits.append(_P21Path(root) / name)
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name}, found {len(hits)}: {hits[:8]}")
    return hits[0]


_p21_work = _P21Path("/kaggle/working")
_p21_primary = _p21_work / "submission.csv"
_p21_parent_preserved = _p21_work / "submission_path13_0920_preserved.csv"
_p21_parent_audit_path = _p21_work / "path13_aman_super_0920_audit.json"
_p21_audit_path = _p21_work / "path21_strong_alpha_apply_audit.json"
_p21_audit = {
    "status": "PATH21_PENDING",
    "strategy": (
        "Path13 Aman super-0920 parent preserved, then per-target rank blend of "
        "Path21 fused-silver-v5 lambda-star gold-crossfit deployment heads at "
        "strong alphas (0.50 / 0.34 / 0) gated on cross-fitted gold gain"
    ),
    "parent": "Path13 Aman super-0920, public score 0.920",
    "blend_component": "aakashkavuru/rsna-knee-path21-fused-labels-train",
    "alpha_rule": (
        "alpha=0.50 if crossfit gold AUC at lambda-star beats the gold-free "
        "parent monitor by >0.02; alpha=0.34 if the gain is in [0.005, 0.02]; "
        "alpha=0 otherwise (alpha must exceed 1/3 to move a 3-study ordering)"
    ),
    "label_source": _P21_LABELS_FILE,
    "label_source_sha256": _P21_LABELS_SHA256,
    "gold_folds_sha256": _P21_GOLD_FOLDS_SHA256,
    "gold_usage": "crossfit",
    "gold_policy": "Path21 component trained with all 58 gold rows overridden under the GOLD_INTEGRATION_PLAN.md section 2 cross-fit protocol; alpha map computed on cross-fitted gold OOF evidence only",
    "test_studies_expected": 3,
}
if not _p21_primary.is_file() or not _p21_parent_audit_path.is_file():
    raise RuntimeError("Path21 requires completed Path13 artifacts")
_p21_parent_receipt = _p21_json.loads(_p21_parent_audit_path.read_text())
if _p21_parent_receipt.get("status") != "PATH13_AMAN_SUPER_0920_VALID":
    raise RuntimeError("Path21 Path13 audit mismatch")
if _p21_sha256(_p21_primary) != _p21_parent_receipt.get("submission_sha256"):
    raise RuntimeError("Path21 Path13 primary hash drift")
_p21_shutil.copy2(_p21_primary, _p21_parent_preserved)

_p21_error = None
try:
    _p21_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if _p21_device.type != "cuda":
        raise RuntimeError("Path21 apply requires CUDA")
    _p21_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    _p21_parent_frame = pd.read_csv(_p21_parent_preserved, dtype={"StudyInstanceUID": str})
    _rad_validate(_p21_parent_frame, _p21_test.StudyInstanceUID)
    if len(_p21_test) != 3 or len(_p21_parent_frame) != 3:
        raise RuntimeError("Path21 parent row count drift")

    # Receipt chain: the train audit is located by name and pinned to the
    # silver-v5 / gold-folds hashes; the heads file must match the audit's
    # own heads_sha256 receipt.
    _p21_train_audit_path = _p21_find_unique(_P21_TRAIN_AUDIT_NAME)
    _p21_train_audit = _p21_json.loads(_p21_train_audit_path.read_text())
    if _p21_train_audit.get("status") != "PATH21_FUSED_LABELS_CROSSFIT_TRAINED":
        raise RuntimeError("Path21 train audit status mismatch")
    _p21_gold_integration = _p21_train_audit.get("gold_integration", {})
    if _p21_train_audit.get("gold_usage") != "crossfit":
        raise RuntimeError("Path21 train cross-fit receipt absent")
    if _p21_gold_integration.get("gold_folds_sha256") != _P21_GOLD_FOLDS_SHA256:
        raise RuntimeError("Path21 train gold-folds pin drift")
    if _p21_gold_integration.get("label_source") != _P21_LABELS_FILE:
        raise RuntimeError("Path21 train label source drift")
    if _p21_gold_integration.get("label_source_sha256") != _P21_LABELS_SHA256:
        raise RuntimeError("Path21 train silver-v5 pin drift")

    _p21_heads_path = _p21_find_unique(_P21_HEADS_NAME)
    _p21_heads_sha256 = _p21_sha256(_p21_heads_path)
    if _p21_train_audit.get("heads_sha256") != _p21_heads_sha256:
        raise RuntimeError("Path21 heads sha256 does not match the train audit receipt")
    _p21_bundle = torch.load(_p21_heads_path, map_location="cpu", weights_only=False)
    if _p21_bundle.get("version") != _P21_BUNDLE_VERSION:
        raise RuntimeError("Path21 payload version drift")
    if _p21_bundle.get("gold_usage") != "crossfit":
        raise RuntimeError("Path21 bundle cross-fit receipt absent")
    if _p21_bundle.get("cv_grouping") != "scanner-site DICOM signature":
        raise RuntimeError("Path21 scanner-grouped receipt absent")
    if _p21_bundle.get("targets") != TARGETS:
        raise RuntimeError("Path21 target order drift")
    if _p21_bundle.get("label_source_sha256") != _P21_LABELS_SHA256:
        raise RuntimeError("Path21 bundle silver-v5 pin drift")
    if _p21_bundle.get("encoder_sha256") != _RAD_ENCODER_SHA256:
        raise RuntimeError("Path21 encoder pin drift")
    if _p21_bundle.get("slots") != [list(slot) for slot in _P21_SLOTS]:
        raise RuntimeError("Path21 slot recipe drift")
    if (
        float(_p21_bundle.get("crop_mm")) != 160.0
        or int(_p21_bundle.get("img")) != 224
        or int(_p21_bundle.get("slices_per_plane")) != 8
    ):
        raise RuntimeError("Path21 pixel recipe drift")

    _p21_lambda_star = {
        target: float(value)
        for target, value in _p21_gold_integration.get("lambda_star", {}).items()
    }
    if sorted(_p21_lambda_star) != sorted(TARGETS):
        raise RuntimeError("Path21 train audit lambda-star incomplete")
    _p21_bundle_lambda = {
        target: float(value) for target, value in _p21_bundle.get("lambda_star", {}).items()
    }
    if _p21_bundle_lambda != _p21_lambda_star:
        raise RuntimeError("Path21 lambda-star drift between bundle and train audit")

    # Strong-alpha gate on the honest cross-fitted gold read only.
    _p21_parent_proxy = _p21_gold_integration.get("parent_gold_monitor_per_target_auc", {})
    _p21_per_lambda = _p21_gold_integration.get("per_lambda", {})
    _p21_alpha = {}
    _p21_alpha_basis = {}
    _p21_gold_gain = {}
    for _p21_target in TARGETS:
        _p21_star = _p21_lambda_star[_p21_target]
        _p21_auc_star = _p21_per_lambda.get(f"{_p21_star:.1f}", {}).get(
            "crossfit_gold_per_target_auc", {}
        ).get(_p21_target)
        _p21_base = _p21_parent_proxy.get(_p21_target)
        if _p21_auc_star is None or _p21_base is None:
            _p21_alpha[_p21_target] = 0.0
            _p21_alpha_basis[_p21_target] = "no_crossfit_evidence"
            continue
        _p21_gain = float(_p21_auc_star) - float(_p21_base)
        _p21_gold_gain[_p21_target] = _p21_gain
        if _p21_gain > _P21_GAIN_STRONG:
            _p21_alpha[_p21_target] = _P21_ALPHA_STRONG
            _p21_alpha_basis[_p21_target] = f"gain_{_p21_gain:.4f}_gt_0.02_strong"
        elif _p21_gain >= _P21_GAIN_MODERATE:
            _p21_alpha[_p21_target] = _P21_ALPHA_MODERATE
            _p21_alpha_basis[_p21_target] = f"gain_{_p21_gain:.4f}_in_[0.005,0.02]_moderate"
        else:
            _p21_alpha[_p21_target] = 0.0
            _p21_alpha_basis[_p21_target] = f"gain_{_p21_gain:.4f}_below_0.005_parent_kept"

    _p21_variants = _p21_bundle.get("gold_fold_variants", {})
    _p21_starred = sorted({f"{_p21_lambda_star[target]:.1f}" for target in TARGETS})
    for _p21_key in _p21_starred:
        _p21_folds = _p21_variants.get(_p21_key)
        if not isinstance(_p21_folds, list) or len(_p21_folds) != 5:
            raise RuntimeError(f"Path21 lambda variant {_p21_key} missing five deployment folds")
    _p21_needed = sorted({f"{_p21_lambda_star[t]:.1f}" for t in TARGETS if _p21_alpha[t] > 0.0})
    for _p21_target in TARGETS:
        if _p21_alpha[_p21_target] > 0.0 and f"{_p21_lambda_star[_p21_target]:.1f}" not in _p21_variants:
            raise RuntimeError(f"Path21 missing lambda variant for blend target {_p21_target}")
    if not _p21_needed:
        raise RuntimeError("Path21 strong-alpha gate passed no target; keeping pure parent")

    globals().update(
        SLOTS=[tuple(slot) for slot in _P21_SLOTS],
        N_SLOT=len(_P21_SLOTS),
        CACHE_SLICES=8,
        IMG=224,
        CACHE_IMG=224,
        CROP_MM=160.0,
        SLICE_BAND=(0.05, 0.95),
        RULES=dict(RULES_LEGACY),
    )
    _p21_series = pd.read_csv(
        ROOT / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    _p21_plane = dict(zip(_p21_series.SeriesInstanceUID, _p21_series.Anatomical_Plane))
    _p21_headers = annotate(walk("test_series"))
    _p21_studies, _p21_pixels, _p21_masks = build_cache(
        pick_slots(_p21_headers, _p21_plane),
        _p21_plane,
        lat_of(_p21_headers, "test-path21 "),
        "test-path21",
    )
    _p21_positions = {str(uid): index for index, uid in enumerate(_p21_studies)}
    _p21_missing = [uid for uid in _p21_test.StudyInstanceUID if uid not in _p21_positions]
    if _p21_missing:
        raise RuntimeError(f"{len(_p21_missing)} test studies absent from path21 cache")
    _p21_order = np.asarray(
        [_p21_positions[uid] for uid in _p21_test.StudyInstanceUID], dtype=np.int64
    )
    _p21_pixels, _p21_masks = _p21_pixels[_p21_order], _p21_masks[_p21_order]
    _p21_tokens = int(np.repeat(_p21_masks[:, :, None], CACHE_SLICES, axis=2).sum())
    if _p21_tokens < int(0.55 * len(_p21_test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f"insufficient path21 test slices: {_p21_tokens}")

    _p21_encoder_path = _rad_find_file("ResNet50.pt", _RAD_ENCODER_SHA256)
    _p21_encoder = _RadEncoder()
    _p21_encoder.load_state_dict(
        torch.load(_p21_encoder_path, map_location="cpu", weights_only=True), strict=True
    )
    _p21_encoder.eval().to(_p21_device)
    for _p21_parameter in _p21_encoder.parameters():
        _p21_parameter.requires_grad_(False)
    if torch.cuda.device_count() > 1:
        _p21_encoder = nn.DataParallel(
            _p21_encoder, device_ids=list(range(torch.cuda.device_count()))
        )
    _p21_features, _p21_token_mask = _rad_encode(
        _p21_encoder, _p21_pixels, _p21_masks, _p21_device
    )
    del _p21_pixels, _p21_masks, _p21_headers
    gc.collect()

    _p21_pred = np.full((len(_p21_test), len(TARGETS)), np.nan, dtype=np.float64)
    _p21_head_count = 0
    for _p21_key in _p21_needed:
        _p21_columns = [
            index
            for index, target in enumerate(TARGETS)
            if _p21_alpha[target] > 0.0 and f"{_p21_lambda_star[target]:.1f}" == _p21_key
        ]
        if not _p21_columns:
            continue
        _p21_member = []
        for _p21_record in _p21_variants[_p21_key]:
            _p21_head = _RadHead().to(_p21_device).eval()
            _p21_head.load_state_dict(_p21_record["state_dict"], strict=True)
            _p21_member.append(
                _rad_predict_head(_p21_head, _p21_features, _p21_token_mask, _p21_device)
            )
            _p21_head_count += 1
            del _p21_head
            torch.cuda.empty_cache()
        _p21_prob = np.mean(np.stack(_p21_member), axis=0)
        _p21_pred[:, _p21_columns] = _p21_prob[:, _p21_columns]
        del _p21_member
    del _p21_features, _p21_token_mask, _p21_encoder
    gc.collect()
    torch.cuda.empty_cache()
    _p21_blend_columns = [
        index for index, target in enumerate(TARGETS) if _p21_alpha[target] > 0.0
    ]
    if not _p21_blend_columns or not np.isfinite(_p21_pred[:, _p21_blend_columns]).all():
        raise RuntimeError("Path21 non-finite blend-target predictions")

    _p21_raw = pd.DataFrame(_p21_pred, columns=TARGETS)
    _p21_raw.insert(0, "StudyInstanceUID", _p21_test.StudyInstanceUID)
    _p21_raw.to_csv(_p21_work / "submission_path21_fused_goldstar_raw.csv", index=False)

    _p21_parent_rank = _rad_rank_columns(_p21_parent_frame[TARGETS].to_numpy())
    _p21_path21_rank = _rad_rank_columns(_p21_pred[:, _p21_blend_columns])
    _p21_final = _p21_parent_frame.copy()
    for _p21_position, _p21_index in enumerate(_p21_blend_columns):
        _p21_target = TARGETS[_p21_index]
        _p21_a = float(_p21_alpha[_p21_target])
        _p21_final[_p21_target] = (
            (1.0 - _p21_a) * _p21_parent_rank[:, _p21_index]
            + _p21_a * _p21_path21_rank[:, _p21_position]
        )
    _rad_validate(_p21_final, _p21_test.StudyInstanceUID)
    _p21_final.to_csv(_p21_primary, index=False)
    _p21_roundtrip = pd.read_csv(_p21_primary, dtype={"StudyInstanceUID": str})
    _rad_validate(_p21_roundtrip, _p21_test.StudyInstanceUID)

    _p21_audit.update(
        status="PATH21_STRONG_ALPHA_APPLIED",
        alpha_map=_p21_alpha,
        alpha_basis=_p21_alpha_basis,
        gold_gain_vs_parent_monitor=_p21_gold_gain,
        lambda_star=_p21_lambda_star,
        parent_sha256=_p21_sha256(_p21_parent_preserved),
        path21_heads_sha256=_p21_heads_sha256,
        path21_train_audit_sha256=_p21_sha256(_p21_train_audit_path),
        path21_raw_sha256=_p21_sha256(_p21_work / "submission_path21_fused_goldstar_raw.csv"),
        submission_sha256=_p21_sha256(_p21_primary),
        test_studies=int(len(_p21_test)),
        tokens=int(_p21_tokens),
        heads=int(_p21_head_count),
        blend_targets=[TARGETS[i] for i in _p21_blend_columns],
        schema_exact=True,
        finite_in_range=True,
    )
    log("Path21 fused-labels lambda-star gold heads blended at strong alphas over the preserved Path13 0.920 parent")
except Exception as _p21_exc:
    _p21_error = _p21_exc
    _p21_audit["status"] = "PATH21_FALLBACK_PARENT_PRESERVED"
    _p21_audit["error"] = f"{type(_p21_exc).__name__}: {_p21_exc}"
    _p21_shutil.copy2(_p21_parent_preserved, _p21_primary)
finally:
    _p21_audit["primary_sha256"] = _p21_sha256(_p21_primary)
    _p21_audit_path.write_text(_p21_json.dumps(_p21_audit, indent=2, sort_keys=True) + "\n")

print(_p21_json.dumps(_p21_audit, indent=2, sort_keys=True))
'''
)


EXPECTED_MUTATIONS = 2
EXPECTED_REQUIRES = 2


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"{SOURCE} not found; build Path13 first")
    notebook = json.loads(SOURCE.read_text())

    fired = 0
    first = notebook["cells"][0]
    if first.get("cell_type") != "markdown":
        raise RuntimeError("Path13 cell 0 is not the expected markdown cell")
    joined = "".join(first["source"])
    joined = replace_once(joined, OLD_TITLE, NEW_TITLE, "markdown title")
    fired += 1
    joined = replace_once(joined, OLD_BODY, NEW_BODY, "markdown body")
    fired += 1
    first["source"] = [line + "\n" for line in joined.splitlines()]

    required = 0
    tail = notebook["cells"][-1]
    if tail.get("cell_type") != "code":
        raise RuntimeError("Path13 final cell is not the audit code cell")
    require_once("".join(tail["source"]), "PATH13_AMAN_SUPER_0920_VALID", "Path13 audit tail anchor")
    required += 1
    require_once(
        "".join(notebook["cells"][-2]["source"]),
        'master.rename(final)',
        "Path13 master-to-submission rename anchor",
    )
    required += 1

    if fired != EXPECTED_MUTATIONS:
        raise RuntimeError(f"expected {EXPECTED_MUTATIONS} mutations, fired {fired}")
    if required != EXPECTED_REQUIRES:
        raise RuntimeError(f"expected {EXPECTED_REQUIRES} requires, observed {required}")

    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in FINAL_CELL.splitlines()],
        }
    )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")

    # Compile-check every code cell of the generated notebook.
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), f"{OUTPUT.name}:cell{index}", "exec")

    metadata = json.loads(METADATA_SOURCE.read_text())
    metadata["id"] = "aakashkavuru/rsna-knee-path21-strong-alpha-apply"
    metadata["title"] = "RSNA Knee Path21 Strong Alpha Apply"
    metadata["code_file"] = "rsna_knee_path21_strong_alpha_apply.ipynb"
    metadata["kernel_sources"] = ["aakashkavuru/rsna-knee-path21-fused-labels-train"]
    METADATA_OUT.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells, {fired} mutations, {required} requires")
    print(f"wrote {METADATA_OUT}")


if __name__ == "__main__":
    main()
