from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontier_unrestricted_submit" / "rsna_knee_frontier_v43_unrestricted.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path3_e11_clean_apply.ipynb"


APPLY_CELL = r'''# Path 3 apply: diverse clean E11 residual over the verified 0.909 Path 2 parent.
import hashlib as _p3_hashlib
import json as _p3_json
import shutil as _p3_shutil
from pathlib import Path as _P3Path

_P3_HEADS_SHA256 = "ebf1968956a08ca2643276f4e4e8a9032347eab4b0b264891d9dd1db01b0e5f5"
_P3_ALPHA_MAP = {
    "ACL": 0.20,
    "MCL": 0.10,
    "Medial Meniscus": 0.35,
    "Lateral Meniscus": 0.35,
    "Medial OA": 0.35,
    "Lateral OA": 0.35,
    "PF OA": 0.35,
    "Effusion": 0.20,
    "Synovitis": 0.35,
    "Baker's": 0.20,
    "Contusion": 0.10,
    "Fracture": 0.10,
}
_P3_OOF = {
    "selection_rows": 4349,
    "scanner_groups": 699,
    "parent_weak_macro": 0.8526790860269484,
    "selected_weak_macro": 0.8601669925498527,
    "selected_delta": 0.007487906522904282,
    "gold_monitor_auc": 0.8536931618231117,
    "gold_training_or_selection_used": False,
}


def _p3_sha256(path):
    return _p3_hashlib.sha256(_P3Path(path).read_bytes()).hexdigest()


def _p3_e11_test_predictions(pinned, test, device, tag):
    """Five-head clean E11 prediction on non-FS series plus one FS anchor."""
    test_series = pd.read_csv(
        ROOT / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    headers = annotate(walk("test_series"))
    audit_official_sequence_metadata(headers, test_series)
    studies, pixels, slot_mask = build_cache(
        pick_slots(headers, plane), plane, lat_of(headers, f"{tag} "), tag
    )
    by_uid = {str(uid): index for index, uid in enumerate(studies)}
    missing = [uid for uid in test.StudyInstanceUID if uid not in by_uid]
    if missing:
        raise RuntimeError(f"{len(missing)} test studies absent from {tag} cache")
    order = np.asarray([by_uid[uid] for uid in test.StudyInstanceUID], dtype=np.int64)
    pixels, slot_mask = pixels[order], slot_mask[order]
    token_count = int(np.repeat(slot_mask[:, :, None], CACHE_SLICES, 2).sum())
    if token_count < int(0.55 * len(test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f"insufficient acquired {tag} test slices: {token_count}")
    features, token_mask = encode_radimagenet(pixels, slot_mask, device)
    del pixels, slot_mask, headers
    gc.collect()

    rows = np.arange(len(test), dtype=np.int64)
    predictions = []
    for record in pinned["folds"]:
        head = FoundationQueryHead().to(device)
        head.load_state_dict(record["state_dict"], strict=True)
        predictions.append(predict_head(head, features, token_mask, rows, device))
        del head
        torch.cuda.empty_cache()
    if len(predictions) != 5:
        raise RuntimeError(f"{tag} test inference did not use all five heads")
    e11_test = np.mean(np.stack(predictions), axis=0)
    if e11_test.shape != (len(test), len(TARGETS)) or not np.isfinite(e11_test).all():
        raise RuntimeError(f"invalid {tag} prediction shape/value: {e11_test.shape}")
    return e11_test, token_count, len(predictions)


_p3_work = _P3Path("/kaggle/working")
_p3_primary = _p3_work / "submission.csv"
_p3_preserved = _p3_work / "submission_path2_0909_preserved.csv"
_p3_audit_path = _p3_work / "path3_e11_apply_audit.json"
_p3_parent_receipt_path = _p3_work / "v43_runtime_audit.json"
_p3_audit = {
    "status": "PATH2_0909_PRESERVED",
    "source_public_score": 0.909,
    "alpha_map": _P3_ALPHA_MAP,
    "oof_diagnostics": _P3_OOF,
    "selection_rule": "largest conservative scanner-OOF-positive alpha per target",
    "gold_policy": "monitor only; excluded from training and blend selection",
}
if not _p3_primary.is_file() or not _p3_parent_receipt_path.is_file():
    raise RuntimeError("Path3 apply requires the completed V43 Path 2 parent")
_p3_parent_receipt = _p3_json.loads(_p3_parent_receipt_path.read_text())
if _p3_parent_receipt.get("status") != "VALID_V43_V41_PARENT_ALPHA070":
    raise RuntimeError("Path3 apply parent receipt mismatch")
if _p3_sha256(_p3_primary) != _p3_parent_receipt.get("submission_sha256"):
    raise RuntimeError("Path3 apply parent hash drift")
_p3_shutil.copy2(_p3_primary, _p3_preserved)

_p3_error = None
try:
    _p3_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if _p3_device.type != "cuda":
        raise RuntimeError("Path3 E11 apply requires CUDA")
    _p3_heads_path = find_input_file("v52_e11_heads.pt")
    if _p3_sha256(_p3_heads_path) != _P3_HEADS_SHA256:
        raise RuntimeError("Path3 E11 heads hash drift")
    _p3_bundle = torch.load(_p3_heads_path, map_location="cpu", weights_only=False)
    if _p3_bundle.get("version") != "path3-e11-radimagenet-diverse-clean-1":
        raise RuntimeError("Path3 E11 payload version drift")
    if _p3_bundle.get("gold_training_or_selection_used") is not False:
        raise RuntimeError("Path3 E11 clean-training receipt absent")
    if _p3_bundle.get("cv_grouping") != "scanner-site DICOM signature":
        raise RuntimeError("Path3 E11 scanner-grouped receipt absent")
    if _p3_bundle.get("targets") != TARGETS:
        raise RuntimeError("Path3 E11 target order drift")
    if len(_p3_bundle.get("folds", [])) != 5:
        raise RuntimeError("Path3 E11 bundle does not contain five folds")

    globals().update(
        SLOTS=list(E11_SLOTS),
        N_SLOT=len(E11_SLOTS),
        CACHE_SLICES=int(E11_CACHE_SLICES),
        IMG=int(E11_IMG),
        CACHE_IMG=int(E11_IMG),
        CROP_MM=float(E11_CROP_MM),
    )
    _p3_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    _p3_e11, _p3_tokens, _p3_head_count = _p3_e11_test_predictions(
        _p3_bundle, _p3_test, _p3_device, "test-path3-e11"
    )
    _p3_base = pd.read_csv(_p3_preserved, dtype={"StudyInstanceUID": str})
    _v52_validate_submission(_p3_base, _p3_test.StudyInstanceUID)
    _p3_base_rank = _v52_rank_columns(_p3_base[TARGETS].to_numpy())
    _p3_e11_rank = _v52_rank_columns(_p3_e11)
    _p3_candidate = _p3_base.copy()
    for _p3_index, _p3_target in enumerate(TARGETS):
        _p3_alpha = float(_P3_ALPHA_MAP[_p3_target])
        _p3_candidate[_p3_target] = (
            (1.0 - _p3_alpha) * _p3_base_rank[:, _p3_index]
            + _p3_alpha * _p3_e11_rank[:, _p3_index]
        )
    _v52_validate_submission(_p3_candidate, _p3_test.StudyInstanceUID)

    _p3_raw = pd.DataFrame(_p3_e11, columns=TARGETS)
    _p3_raw.insert(0, "StudyInstanceUID", _p3_test.StudyInstanceUID)
    _v52_validate_submission(_p3_raw, _p3_test.StudyInstanceUID)
    _p3_raw.to_csv(_p3_work / "submission_path3_e11_raw.csv", index=False)
    _p3_candidate.to_csv(_p3_primary, index=False)
    _p3_roundtrip = pd.read_csv(_p3_primary, dtype={"StudyInstanceUID": str})
    _v52_validate_submission(_p3_roundtrip, _p3_test.StudyInstanceUID)
    _p3_audit.update(
        status="PATH3_E11_CLEAN_APPLIED",
        heads_sha256=_P3_HEADS_SHA256,
        parent_sha256=_p3_sha256(_p3_preserved),
        submission_sha256=_p3_sha256(_p3_primary),
        test_studies=int(len(_p3_test)),
        tokens=int(_p3_tokens),
        heads=int(_p3_head_count),
        schema_exact=True,
        finite_in_range=True,
    )
    log("Path3 clean E11 target-specific residual applied over the 0.909 parent")
except Exception as _p3_exc:
    _p3_error = _p3_exc
    _p3_audit["status"] = "ERROR_PATH2_0909_PRESERVED"
    _p3_audit["error"] = f"{type(_p3_exc).__name__}: {_p3_exc}"
    _p3_shutil.copy2(_p3_preserved, _p3_primary)
finally:
    _p3_audit["primary_sha256"] = _p3_sha256(_p3_primary)
    _p3_audit_path.write_text(_p3_json.dumps(_p3_audit, indent=2, sort_keys=True) + "\n")

if _p3_error is not None:
    raise RuntimeError(f"Path3 apply failed: {_p3_audit['error']}") from _p3_error
print(_p3_json.dumps(_p3_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 3 — clean diverse E11 application\n\n"
        "This notebook reproduces the verified Path 2 0.909 parent, then applies "
        "the scanner-grouped clean E11 arm trained on 4,349 non-gold studies. "
        "Target-specific weights are fixed from non-gold scanner OOF evidence.\n"
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
