from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path10_kimi_fs160_stack_apply" / "rsna_knee_path10_kimi_fs160_stack_apply.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path16_original_path14_apply.ipynb"


FINAL_CELL = r'''# Path 16 finalizer: original-primary Path10 + low-weight Path14 reconciliation head.
import hashlib as _p16_hashlib
import json as _p16_json
import shutil as _p16_shutil
from pathlib import Path as _P16Path

_P16_HEADS_SHA256 = "2431604d7f848767fafc057690c41cb9f455e48547e0a2b76182af2521c6eb02"
_P16_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("COR_NOFS", "Coronal", None, False),
]
_P16_ALPHA = {
    "ACL": 0.10,
    "MCL": 0.10,
    "Medial Meniscus": 0.10,
    "Lateral Meniscus": 0.10,
    "Medial OA": 0.10,
    "Lateral OA": 0.10,
    "PF OA": 0.00,
    "Effusion": 0.00,
    "Synovitis": 0.00,
    "Baker's": 0.00,
    "Contusion": 0.00,
    "Fracture": 0.00,
}


def _p16_sha256(path):
    return _p16_hashlib.sha256(_P16Path(path).read_bytes()).hexdigest()


_p16_work = _P16Path("/kaggle/working")
_p16_primary = _p16_work / "submission.csv"
_p16_path10_preserved = _p16_work / "submission_path10_0915_preserved.csv"
_p16_path10_audit = _p16_work / "path10_kimi_fs160_stack_audit.json"
_p16_audit_path = _p16_work / "path16_original_path14_apply_audit.json"
if not _p16_primary.is_file() or not _p16_path10_audit.is_file():
    raise RuntimeError("Path16 requires completed Path10 artifacts")
_p16_path10_receipt = _p16_json.loads(_p16_path10_audit.read_text())
if _p16_path10_receipt.get("status") != "PATH10_KIMI_FS160_STACK_APPLIED":
    raise RuntimeError("Path16 Path10 audit mismatch")
if _p16_sha256(_p16_primary) != _p16_path10_receipt.get("submission_sha256"):
    raise RuntimeError("Path16 Path10 primary hash drift")
_p16_shutil.copy2(_p16_primary, _p16_path10_preserved)

_p16_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if _p16_device.type != "cuda":
    raise RuntimeError("Path16 Path14 apply requires CUDA")
_p16_heads_path = _p5_find_file_by_sha256("path14_reconciled_heads.pt", _P16_HEADS_SHA256)
_p16_bundle = torch.load(_p16_heads_path, map_location="cpu", weights_only=False)
if _p16_bundle.get("version") != "path14-crossfit-reconciliation-clean-1":
    raise RuntimeError("Path14 payload version drift")
if _p16_bundle.get("gold_training_or_selection_used") is not False:
    raise RuntimeError("Path14 clean-training receipt absent")
if _p16_bundle.get("cv_grouping") != "scanner-site DICOM signature":
    raise RuntimeError("Path14 scanner-grouped receipt absent")
if _p16_bundle.get("targets") != TARGETS:
    raise RuntimeError("Path14 target order drift")
if len(_p16_bundle.get("folds", [])) != 5:
    raise RuntimeError("Path14 bundle does not contain five folds")
_p16_delta = _p16_bundle.get("reconciliation", {}).get("per_target_original_delta", {})
_p16_expected_positive = {
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"
}
if not _p16_expected_positive.issubset({t for t, v in _p16_delta.items() if float(v) > 0.0}):
    raise RuntimeError("Path14 target-delta receipt does not support Path16 alpha map")

globals().update(
    SLOTS=list(_P16_SLOTS),
    N_SLOT=len(_P16_SLOTS),
    CACHE_SLICES=8,
    IMG=224,
    CACHE_IMG=224,
    CROP_MM=160.0,
    SLICE_BAND=(0.05, 0.95),
)
_p16_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
_p16_path14, _p16_tokens, _p16_head_count = _p5_e11_test_predictions(
    _p16_bundle, _p16_test, _p16_device, "test-path16-path14-reconcile"
)

_p16_base_frame = pd.read_csv(_p16_path10_preserved, dtype={"StudyInstanceUID": str})
_v52_validate_submission(_p16_base_frame, _p16_test.StudyInstanceUID)
_p16_base_rank = _v52_rank_columns(_p16_base_frame[TARGETS].to_numpy())
_p16_path14_rank = _v52_rank_columns(_p16_path14)

_p16_raw = pd.DataFrame(_p16_path14, columns=TARGETS)
_p16_raw.insert(0, "StudyInstanceUID", _p16_test.StudyInstanceUID)
_v52_validate_submission(_p16_raw, _p16_test.StudyInstanceUID)
_p16_raw.to_csv(_p16_work / "submission_path16_path14_raw.csv", index=False)

_p16_final = _p16_base_frame.copy()
for _p16_index, _p16_target in enumerate(TARGETS):
    _p16_alpha = float(_P16_ALPHA[_p16_target])
    if _p16_alpha > 0.0:
        _p16_final[_p16_target] = (
            (1.0 - _p16_alpha) * _p16_base_rank[:, _p16_index]
            + _p16_alpha * _p16_path14_rank[:, _p16_index]
        )
_v52_validate_submission(_p16_final, _p16_test.StudyInstanceUID)
_p16_final.to_csv(_p16_primary, index=False)
_p16_roundtrip = pd.read_csv(_p16_primary, dtype={"StudyInstanceUID": str})
_v52_validate_submission(_p16_roundtrip, _p16_test.StudyInstanceUID)

_p16_audit = {
    "status": "PATH16_ORIGINAL_PATH14_APPLIED",
    "strategy": "original-primary rank blend: 90% Path10 parent plus 10% Path14 only on positive-delta targets",
    "parent": "Path10 Kimi-FS160 stack, public score 0.915",
    "hybrid_component": "Path14 cross-fitted supervision-reconciliation head trained without gold optimization",
    "not_public_notebook_parent": True,
    "minimum_original_share_per_target": 0.90,
    "alpha_map": _P16_ALPHA,
    "path14_positive_delta_receipt": {target: _p16_delta.get(target) for target in _p16_expected_positive},
    "gold_policy": "Path14 gold rows were monitor-only in training; current Path16 uses the user-authorized unrestricted submission budget but does not train on gold rows.",
    "path10_sha256": _p16_sha256(_p16_path10_preserved),
    "path14_heads_sha256": _P16_HEADS_SHA256,
    "path14_raw_sha256": _p16_sha256(_p16_work / "submission_path16_path14_raw.csv"),
    "submission_sha256": _p16_sha256(_p16_primary),
    "test_studies": int(len(_p16_test)),
    "tokens": int(_p16_tokens),
    "heads": int(_p16_head_count),
    "schema_exact": True,
    "finite_in_range": True,
}
_p16_audit_path.write_text(_p16_json.dumps(_p16_audit, indent=2, sort_keys=True) + "\n")
print(_p16_json.dumps(_p16_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 16 — original-primary Path10 + Path14 reconciliation\n\n"
        "This notebook runs the validated Path10 inference path, then adds only a "
        "10% rank contribution from the original Path14 cross-fitted reconciliation "
        "heads on the targets where Path14 improved non-gold OOF: ACL, MCL, menisci, "
        "and medial/lateral OA. The parent is not Path13/Aman; public notebooks are "
        "not used as the primary signal.\n"
    )
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
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
