from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path5_supervision_fusion_apply" / "rsna_knee_path5_supervision_fusion_apply.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path10_kimi_fs160_stack_apply.ipynb"


APPLY_CELL = r'''# Path 10 apply: Path6 Kimi-FS160 clean residual stacked over validated Path5.
import hashlib as _p10_hashlib
import json as _p10_json
import shutil as _p10_shutil
from pathlib import Path as _P10Path

_P10_HEADS_SHA256 = "084d48ba04277c908d86c450dc567b96a79e36268505e53a6bae7dd5e7c541f6"
_P10_PATH6_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("COR_NOFS", "Coronal", None, False),
]
_P10_REPLACE_ALPHA_MAP = {
    "ACL": 0.20,
    "MCL": 0.20,
    "Medial Meniscus": 0.35,
    "Lateral Meniscus": 0.35,
    "Medial OA": 0.20,
    "Lateral OA": 0.20,
    "PF OA": 0.20,
    "Effusion": 0.00,
    "Synovitis": 0.35,
    "Baker's": 0.10,
    "Contusion": 0.35,
    "Fracture": 0.20,
}
_P10_STACK_BETA_MAP = {
    "ACL": 0.10,
    "MCL": 0.20,
    "Medial Meniscus": 0.10,
    "Lateral Meniscus": 0.15,
    "Medial OA": 0.00,
    "Lateral OA": 0.00,
    "PF OA": 0.00,
    "Effusion": 0.00,
    "Synovitis": 0.10,
    "Baker's": 0.00,
    "Contusion": 0.20,
    "Fracture": 0.05,
}
_P10_OOF = {
    "baseline_before_today_public_score": 0.912,
    "current_best_public_score": 0.915,
    "path5_weak_oof_auc": 0.8514528222079499,
    "path5_gold_monitor_auc": 0.8574687815318786,
    "path6_weak_oof_auc": 0.8556198854004129,
    "path6_gold_monitor_auc": 0.8670964070814646,
    "path6_blend_over_e2_macro": {
        "0.10": 0.895134796140744,
        "0.20": 0.8976460098434152,
        "0.35": 0.8983071264906822,
        "0.50": 0.8948310086785702,
    },
    "selection_note": (
        "Path10 stack betas modify only targets where Path6 Kimi-FS160 beats or plausibly "
        "diversifies Path5 diagnostics; Effusion/OA/PF/Baker are preserved from Path5."
    ),
}


def _p10_sha256(path):
    return _p10_hashlib.sha256(_P10Path(path).read_bytes()).hexdigest()


_p10_work = _P10Path("/kaggle/working")
_p10_primary = _p10_work / "submission.csv"
_p10_path5_preserved = _p10_work / "submission_path5_0915_preserved.csv"
_p10_base_preserved = _p10_work / "submission_path2_robust_0912_preserved.csv"
_p10_path5_audit_path = _p10_work / "path5_supervision_apply_audit.json"
_p10_audit_path = _p10_work / "path10_kimi_fs160_stack_audit.json"
_p10_audit = {
    "status": "PATH5_0915_PRESERVED",
    "source_public_score": 0.915,
    "path6_heads_sha256": _P10_HEADS_SHA256,
    "replace_alpha_map": _P10_REPLACE_ALPHA_MAP,
    "stack_beta_map": _P10_STACK_BETA_MAP,
    "oof_diagnostics": _P10_OOF,
    "selection_rule": "target-specific Path6-on-Path5 stack from non-gold weak OOF and monitor-only gold diagnostics",
    "gold_policy": "58 official rows monitor-only; excluded from Path5/Path6 training and blend selection",
}

if not _p10_primary.is_file() or not _p10_base_preserved.is_file() or not _p10_path5_audit_path.is_file():
    raise RuntimeError("Path10 requires completed Path5 apply artifacts")
_p10_path5_receipt = _p10_json.loads(_p10_path5_audit_path.read_text())
if _p10_path5_receipt.get("status") != "PATH5_E11_SUPERVISION_FUSION_APPLIED":
    raise RuntimeError("Path10 Path5 receipt mismatch")
if _p10_sha256(_p10_primary) != _p10_path5_receipt.get("submission_sha256"):
    raise RuntimeError("Path10 Path5 primary hash drift before stacking")
_p10_shutil.copy2(_p10_primary, _p10_path5_preserved)

_p10_error = None
try:
    _p10_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if _p10_device.type != "cuda":
        raise RuntimeError("Path10 Path6 apply requires CUDA")
    _p10_heads_path = _p5_find_file_by_sha256("v52_e11_heads.pt", _P10_HEADS_SHA256)
    _p10_bundle = torch.load(_p10_heads_path, map_location="cpu", weights_only=False)
    if _p10_bundle.get("version") != "path6-kimi-fs160-clean-1":
        raise RuntimeError("Path6 payload version drift")
    if _p10_bundle.get("gold_training_or_selection_used") is not False:
        raise RuntimeError("Path6 clean-training receipt absent")
    if _p10_bundle.get("cv_grouping") != "scanner-site DICOM signature":
        raise RuntimeError("Path6 scanner-grouped receipt absent")
    if _p10_bundle.get("targets") != TARGETS:
        raise RuntimeError("Path6 target order drift")
    if len(_p10_bundle.get("folds", [])) != 5:
        raise RuntimeError("Path6 bundle does not contain five folds")

    globals().update(
        SLOTS=list(_P10_PATH6_SLOTS),
        N_SLOT=len(_P10_PATH6_SLOTS),
        CACHE_SLICES=8,
        IMG=224,
        CACHE_IMG=224,
        CROP_MM=160.0,
        SLICE_BAND=(0.05, 0.95),
    )
    _p10_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    _p10_path6, _p10_tokens, _p10_head_count = _p5_e11_test_predictions(
        _p10_bundle, _p10_test, _p10_device, "test-path10-path6-kimi-fs160"
    )

    _p10_path5_frame = pd.read_csv(_p10_path5_preserved, dtype={"StudyInstanceUID": str})
    _p10_base_frame = pd.read_csv(_p10_base_preserved, dtype={"StudyInstanceUID": str})
    _v52_validate_submission(_p10_path5_frame, _p10_test.StudyInstanceUID)
    _v52_validate_submission(_p10_base_frame, _p10_test.StudyInstanceUID)

    _p10_base_rank = _v52_rank_columns(_p10_base_frame[TARGETS].to_numpy())
    _p10_path5_rank = _v52_rank_columns(_p10_path5_frame[TARGETS].to_numpy())
    _p10_path6_rank = _v52_rank_columns(_p10_path6)

    _p10_raw = pd.DataFrame(_p10_path6, columns=TARGETS)
    _p10_raw.insert(0, "StudyInstanceUID", _p10_test.StudyInstanceUID)
    _v52_validate_submission(_p10_raw, _p10_test.StudyInstanceUID)
    _p10_raw.to_csv(_p10_work / "submission_path10_path6_raw.csv", index=False)

    _p10_replace = _p10_base_frame.copy()
    for _p10_index, _p10_target in enumerate(TARGETS):
        _p10_alpha = float(_P10_REPLACE_ALPHA_MAP[_p10_target])
        _p10_replace[_p10_target] = (
            (1.0 - _p10_alpha) * _p10_base_rank[:, _p10_index]
            + _p10_alpha * _p10_path6_rank[:, _p10_index]
        )
    _v52_validate_submission(_p10_replace, _p10_test.StudyInstanceUID)
    _p10_replace.to_csv(_p10_work / "submission_path10_path6_replace.csv", index=False)

    _p10_stack = _p10_path5_frame.copy()
    for _p10_index, _p10_target in enumerate(TARGETS):
        _p10_beta = float(_P10_STACK_BETA_MAP[_p10_target])
        if _p10_beta > 0.0:
            _p10_stack[_p10_target] = (
                (1.0 - _p10_beta) * _p10_path5_rank[:, _p10_index]
                + _p10_beta * _p10_path6_rank[:, _p10_index]
            )
    _v52_validate_submission(_p10_stack, _p10_test.StudyInstanceUID)
    _p10_stack.to_csv(_p10_primary, index=False)
    _p10_roundtrip = pd.read_csv(_p10_primary, dtype={"StudyInstanceUID": str})
    _v52_validate_submission(_p10_roundtrip, _p10_test.StudyInstanceUID)

    _p10_audit.update(
        status="PATH10_KIMI_FS160_STACK_APPLIED",
        path5_sha256=_p10_sha256(_p10_path5_preserved),
        base_sha256=_p10_sha256(_p10_base_preserved),
        path6_raw_sha256=_p10_sha256(_p10_work / "submission_path10_path6_raw.csv"),
        replace_sha256=_p10_sha256(_p10_work / "submission_path10_path6_replace.csv"),
        submission_sha256=_p10_sha256(_p10_primary),
        test_studies=int(len(_p10_test)),
        tokens=int(_p10_tokens),
        heads=int(_p10_head_count),
        recipe={
            "slots": [list(slot) for slot in _P10_PATH6_SLOTS],
            "crop_mm": 160.0,
            "slice_band": [0.05, 0.95],
            "cache_slices": 8,
            "img": 224,
        },
        schema_exact=True,
        finite_in_range=True,
    )
    log("Path10 Kimi-FS160 stack applied over validated Path5")
except Exception as _p10_exc:
    _p10_error = _p10_exc
    _p10_audit["status"] = "ERROR_PATH5_0915_PRESERVED"
    _p10_audit["error"] = f"{type(_p10_exc).__name__}: {_p10_exc}"
    _p10_shutil.copy2(_p10_path5_preserved, _p10_primary)
finally:
    _p10_audit["primary_sha256"] = _p10_sha256(_p10_primary)
    _p10_audit_path.write_text(_p10_json.dumps(_p10_audit, indent=2, sort_keys=True) + "\n")

if _p10_error is not None:
    raise RuntimeError(f"Path10 apply failed: {_p10_audit['error']}") from _p10_error
print(_p10_json.dumps(_p10_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 10 — Kimi-FS160 stack over Path5\n\n"
        "This notebook first reproduces the validated Path5 submission that scored 0.915, "
        "then applies the Path6 Kimi-FS160 clean residual only on targets where the "
        "Path6 diagnostics justify it. It also writes a Path6 replacement CSV for audit. "
        "Gold rows remain monitor-only.\n"
    )
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in APPLY_CELL.splitlines()],
        }
    )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
