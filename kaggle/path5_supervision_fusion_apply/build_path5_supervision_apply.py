from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontier_path2_robust_submit" / "rsna_knee_path2_robust.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path5_supervision_fusion_apply.ipynb"


APPLY_CELL = r'''# Path 5 apply: supervision-fusion clean E11 residual over the 0.912 Path2 robust base.
import hashlib as _p5_hashlib
import json as _p5_json
import shutil as _p5_shutil
from pathlib import Path as _P5Path

_P5_HEADS_SHA256 = "2e4cfb2a3dd73316fd0d2aafa2debac6efb9f16af7f2daebd04a9edb30d1cb0c"
_P5_ALPHA_MAP = {
    "ACL": 0.20,
    "MCL": 0.10,
    "Medial Meniscus": 0.35,
    "Lateral Meniscus": 0.35,
    "Medial OA": 0.20,
    "Lateral OA": 0.20,
    "PF OA": 0.35,
    "Effusion": 0.05,
    "Synovitis": 0.35,
    "Baker's": 0.20,
    "Contusion": 0.20,
    "Fracture": 0.20,
}
_P5_OOF = {
    "baseline_public_score": 0.912,
    "path2_robust_sha256": "030aebf5e487c572124a58c429cb7b5261de7d9fd14a6c30f48489db9ca1a49f",
    "selection_rows": 4349,
    "scanner_groups": 705,
    "e2_base_masked_weak_macro": 0.8911629859652598,
    "path5_raw_masked_weak_macro": 0.8514528222079499,
    "path5_gold_monitor_auc": 0.8574687815318786,
    "selection_note": "Path5 alphas are fixed from non-gold masked weak OOF; 58 gold rows monitor-only",
}


def _p5_sha256(path):
    return _p5_hashlib.sha256(_P5Path(path).read_bytes()).hexdigest()


def _p5_find_file_by_sha256(name, digest):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            path = _P5Path(root) / name
            if _p5_sha256(path) == digest:
                return path
            hits.append(str(path))
    raise FileNotFoundError(f"{name} with sha256 {digest} not found; candidates={hits[:8]}")


def _p5_e11_test_predictions(pinned, test, device, tag):
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


_p5_work = _P5Path("/kaggle/working")
_p5_primary = _p5_work / "submission.csv"
_p5_base = _p5_work / "submission_path2_robust_0912_preserved.csv"
_p5_audit_path = _p5_work / "path5_supervision_apply_audit.json"
_p5_base_audit_path = _p5_work / "path2_robust_runtime_audit.json"
_p5_audit = {
    "status": "PATH2_ROBUST_0912_PRESERVED",
    "source_public_score": 0.912,
    "alpha_map": _P5_ALPHA_MAP,
    "oof_diagnostics": _P5_OOF,
    "selection_rule": "target-specific non-gold masked weak-OOF-positive alpha map",
    "gold_policy": "monitor only; excluded from training and blend selection",
}
if not _p5_primary.is_file() or not _p5_base_audit_path.is_file():
    raise RuntimeError("Path5 apply requires the completed Path2 robust base")
_p5_base_receipt = _p5_json.loads(_p5_base_audit_path.read_text())
if _p5_base_receipt.get("status") != "VALID_PATH2_ROBUST_PER_TARGET_V1":
    raise RuntimeError("Path5 apply Path2 robust receipt mismatch")
if _p5_sha256(_p5_primary) != _p5_base_receipt.get("submission_sha256"):
    raise RuntimeError("Path5 apply base hash drift")
_p5_shutil.copy2(_p5_primary, _p5_base)

_p5_error = None
try:
    _p5_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if _p5_device.type != "cuda":
        raise RuntimeError("Path5 E11 apply requires CUDA")
    _p5_heads_path = _p5_find_file_by_sha256("v52_e11_heads.pt", _P5_HEADS_SHA256)
    _p5_bundle = torch.load(_p5_heads_path, map_location="cpu", weights_only=False)
    if _p5_bundle.get("version") != "path5-e11-supervision-fusion-clean-1":
        raise RuntimeError("Path5 E11 payload version drift")
    if _p5_bundle.get("gold_training_or_selection_used") is not False:
        raise RuntimeError("Path5 E11 clean-training receipt absent")
    if _p5_bundle.get("cv_grouping") != "scanner-site DICOM signature":
        raise RuntimeError("Path5 E11 scanner-grouped receipt absent")
    if _p5_bundle.get("targets") != TARGETS:
        raise RuntimeError("Path5 E11 target order drift")
    if len(_p5_bundle.get("folds", [])) != 5:
        raise RuntimeError("Path5 E11 bundle does not contain five folds")

    globals().update(
        SLOTS=list(E11_SLOTS),
        N_SLOT=len(E11_SLOTS),
        CACHE_SLICES=int(E11_CACHE_SLICES),
        IMG=int(E11_IMG),
        CACHE_IMG=int(E11_IMG),
        CROP_MM=float(E11_CROP_MM),
    )
    _p5_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    _p5_e11, _p5_tokens, _p5_head_count = _p5_e11_test_predictions(
        _p5_bundle, _p5_test, _p5_device, "test-path5-e11"
    )
    _p5_base_frame = pd.read_csv(_p5_base, dtype={"StudyInstanceUID": str})
    _v52_validate_submission(_p5_base_frame, _p5_test.StudyInstanceUID)
    _p5_base_rank = _v52_rank_columns(_p5_base_frame[TARGETS].to_numpy())
    _p5_e11_rank = _v52_rank_columns(_p5_e11)
    _p5_candidate = _p5_base_frame.copy()
    for _p5_index, _p5_target in enumerate(TARGETS):
        _p5_alpha = float(_P5_ALPHA_MAP[_p5_target])
        _p5_candidate[_p5_target] = (
            (1.0 - _p5_alpha) * _p5_base_rank[:, _p5_index]
            + _p5_alpha * _p5_e11_rank[:, _p5_index]
        )
    _v52_validate_submission(_p5_candidate, _p5_test.StudyInstanceUID)

    _p5_raw = pd.DataFrame(_p5_e11, columns=TARGETS)
    _p5_raw.insert(0, "StudyInstanceUID", _p5_test.StudyInstanceUID)
    _v52_validate_submission(_p5_raw, _p5_test.StudyInstanceUID)
    _p5_raw.to_csv(_p5_work / "submission_path5_e11_raw.csv", index=False)
    _p5_candidate.to_csv(_p5_primary, index=False)
    _p5_roundtrip = pd.read_csv(_p5_primary, dtype={"StudyInstanceUID": str})
    _v52_validate_submission(_p5_roundtrip, _p5_test.StudyInstanceUID)
    _p5_audit.update(
        status="PATH5_E11_SUPERVISION_FUSION_APPLIED",
        heads_sha256=_P5_HEADS_SHA256,
        base_sha256=_p5_sha256(_p5_base),
        submission_sha256=_p5_sha256(_p5_primary),
        test_studies=int(len(_p5_test)),
        tokens=int(_p5_tokens),
        heads=int(_p5_head_count),
        schema_exact=True,
        finite_in_range=True,
    )
    log("Path5 supervision-fusion E11 residual applied over the 0.912 Path2 robust base")
except Exception as _p5_exc:
    _p5_error = _p5_exc
    _p5_audit["status"] = "ERROR_PATH2_ROBUST_0912_PRESERVED"
    _p5_audit["error"] = f"{type(_p5_exc).__name__}: {_p5_exc}"
    _p5_shutil.copy2(_p5_base, _p5_primary)
finally:
    _p5_audit["primary_sha256"] = _p5_sha256(_p5_primary)
    _p5_audit_path.write_text(_p5_json.dumps(_p5_audit, indent=2, sort_keys=True) + "\n")

if _p5_error is not None:
    raise RuntimeError(f"Path5 apply failed: {_p5_audit['error']}") from _p5_error
print(_p5_json.dumps(_p5_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 5 — supervision-fusion E11 application\n\n"
        "This notebook reproduces the Path2 robust 0.912 base, then applies the "
        "Path5 clean supervision-fusion E11 residual. The 58 official labels were "
        "monitor-only in training and not used for alpha selection.\n"
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
