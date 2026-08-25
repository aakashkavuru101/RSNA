from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path13_aman_super_0920" / "rsna_knee_path13_aman_super_0920.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path19_path13_path14gold_apply.ipynb"


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
NEW_TITLE = "# Path 19 — Path13 super-0920 parent + Path14 reconciliation heads (gold-evidence targets)"

OLD_BODY = "Fork of `amanatar/rsna-knee-super-ensemble-0920` with an appended audit gate."
NEW_BODY = (
    "Exact clone of the Path13 fork of `amanatar/rsna-knee-super-ensemble-0920` "
    "(public LB 0.920), including its appended audit gate, plus one final stage: a "
    "per-target rank blend of the Path14 supervision-reconciliation (label-consistency) "
    "heads on the three targets where the gold-58 re-eval showed Path14 beating the "
    "Path6 parent — Effusion 0.15, Lateral OA 0.10, PF OA 0.10 (gold AUC deltas "
    "+0.035 / +0.017 / +0.015). All other targets stay at alpha 0.0; gold showed "
    "Path14 regresses Lateral Meniscus and Medial OA despite silver claims. Heads "
    "are located by filename plus pinned sha256 from the "
    "`rsna-knee-path14-crossfit-reconcile-train` kernel output. Any guard failure "
    "restores the preserved Path13 submission byte-for-byte."
)


FINAL_CELL = r'''# Path 19 finalizer: Path13 Aman super-0920 parent + Path14 supervision-reconciliation heads on gold-evidence targets.
import hashlib as _p19_hashlib
import json as _p19_json
import shutil as _p19_shutil
from pathlib import Path as _P19Path

_P19_HEADS_NAME = "path14_reconciled_heads.pt"
_P19_HEADS_SHA256 = "2431604d7f848767fafc057690c41cb9f455e48547e0a2b76182af2521c6eb02"
_P19_BUNDLE_VERSION = "path14-crossfit-reconciliation-clean-1"
_P19_ALPHA = {
    "ACL": 0.00,
    "MCL": 0.00,
    "Medial Meniscus": 0.00,
    "Lateral Meniscus": 0.00,
    "Medial OA": 0.00,
    "Lateral OA": 0.10,
    "PF OA": 0.10,
    "Effusion": 0.15,
    "Synovitis": 0.00,
    "Baker's": 0.00,
    "Contusion": 0.00,
    "Fracture": 0.00,
}
_P19_GOLD_REEVAL_DELTA = {
    "Effusion": 0.034782608695652195,
    "Lateral OA": 0.01740812379110268,
    "PF OA": 0.015444015444015413,
}
_P19_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("COR_NOFS", "Coronal", None, False),
]


def _p19_sha256(path):
    return _p19_hashlib.sha256(_P19Path(path).read_bytes()).hexdigest()


def _p19_find_file_by_sha256(name, digest):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            path = _P19Path(root) / name
            if _p19_sha256(path) == digest:
                return path
            hits.append(str(path))
    raise FileNotFoundError(f"{name} with sha256 {digest} not found; candidates={hits[:8]}")


_p19_work = _P19Path("/kaggle/working")
_p19_primary = _p19_work / "submission.csv"
_p19_parent_preserved = _p19_work / "submission_path13_0920_preserved.csv"
_p19_parent_audit_path = _p19_work / "path13_aman_super_0920_audit.json"
_p19_audit_path = _p19_work / "path19_path13_path14gold_apply_audit.json"
_p19_audit = {
    "status": "PATH19_PENDING",
    "strategy": "Path13 Aman super-0920 parent preserved, then per-target rank blend of Path14 supervision-reconciliation heads on three gold-evidence targets (label-consistency probe)",
    "parent": "Path13 Aman super-0920, public score 0.920",
    "alpha_map": _P19_ALPHA,
    "gold_reeval_delta_receipt": _P19_GOLD_REEVAL_DELTA,
    "path14_heads_sha256": _P19_HEADS_SHA256,
    "gold_usage": "none",
    "selection_evidence": "gold-58 re-eval (.codex_work/path14_output/gold_reeval.json)",
    "test_studies_expected": 3,
}
if not _p19_primary.is_file() or not _p19_parent_audit_path.is_file():
    raise RuntimeError("Path19 requires completed Path13 artifacts")
_p19_shutil.copy2(_p19_primary, _p19_parent_preserved)

_p19_error = None
try:
    _p19_parent_receipt = _p19_json.loads(_p19_parent_audit_path.read_text())
    if _p19_parent_receipt.get("status") != "PATH13_AMAN_SUPER_0920_VALID":
        raise RuntimeError("Path19 Path13 audit mismatch")
    if _p19_sha256(_p19_primary) != _p19_parent_receipt.get("submission_sha256"):
        raise RuntimeError("Path19 Path13 primary hash drift")
    _p19_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if _p19_device.type != "cuda":
        raise RuntimeError("Path19 Path14 apply requires CUDA")
    _p19_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    _p19_parent_frame = pd.read_csv(_p19_parent_preserved, dtype={"StudyInstanceUID": str})
    _rad_validate(_p19_parent_frame, _p19_test.StudyInstanceUID)
    if len(_p19_test) != 3 or len(_p19_parent_frame) != 3:
        raise RuntimeError("Path19 parent row count drift")

    _p19_heads_path = _p19_find_file_by_sha256(_P19_HEADS_NAME, _P19_HEADS_SHA256)
    _p19_bundle = torch.load(_p19_heads_path, map_location="cpu", weights_only=False)
    if _p19_bundle.get("version") != _P19_BUNDLE_VERSION:
        raise RuntimeError("Path14 payload version drift")
    if _p19_bundle.get("gold_training_or_selection_used") is not False:
        raise RuntimeError("Path14 clean-training receipt absent")
    if _p19_bundle.get("cv_grouping") != "scanner-site DICOM signature":
        raise RuntimeError("Path14 scanner-grouped receipt absent")
    if _p19_bundle.get("targets") != TARGETS:
        raise RuntimeError("Path14 target order drift")
    if _p19_bundle.get("encoder_sha256") != _RAD_ENCODER_SHA256:
        raise RuntimeError("Path14 encoder pin drift")
    if _p19_bundle.get("slots") != [list(slot) for slot in _P19_SLOTS]:
        raise RuntimeError("Path14 slot recipe drift")
    if (
        float(_p19_bundle.get("crop_mm")) != 160.0
        or int(_p19_bundle.get("img")) != 224
        or int(_p19_bundle.get("slices_per_plane")) != 8
    ):
        raise RuntimeError("Path14 pixel recipe drift")
    _p19_folds = _p19_bundle.get("folds")
    if not isinstance(_p19_folds, list) or len(_p19_folds) != 5:
        raise RuntimeError("Path14 bundle does not contain five folds")
    for _p19_record in _p19_folds:
        if not isinstance(_p19_record.get("state_dict"), dict):
            raise RuntimeError("Path14 fold record missing state_dict")

    globals().update(
        SLOTS=[tuple(slot) for slot in _P19_SLOTS],
        N_SLOT=len(_P19_SLOTS),
        CACHE_SLICES=8,
        IMG=224,
        CACHE_IMG=224,
        CROP_MM=160.0,
        SLICE_BAND=(0.05, 0.95),
        RULES=dict(RULES_LEGACY),
    )
    _p19_series = pd.read_csv(
        ROOT / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    _p19_plane = dict(zip(_p19_series.SeriesInstanceUID, _p19_series.Anatomical_Plane))
    _p19_headers = annotate(walk("test_series"))
    _p19_studies, _p19_pixels, _p19_masks = build_cache(
        pick_slots(_p19_headers, _p19_plane),
        _p19_plane,
        lat_of(_p19_headers, "test-path19-path14 "),
        "test-path19-path14",
    )
    _p19_positions = {str(uid): index for index, uid in enumerate(_p19_studies)}
    _p19_missing = [uid for uid in _p19_test.StudyInstanceUID if uid not in _p19_positions]
    if _p19_missing:
        raise RuntimeError(f"{len(_p19_missing)} test studies absent from path19 cache")
    _p19_order = np.asarray(
        [_p19_positions[uid] for uid in _p19_test.StudyInstanceUID], dtype=np.int64
    )
    _p19_pixels, _p19_masks = _p19_pixels[_p19_order], _p19_masks[_p19_order]
    _p19_tokens = int(np.repeat(_p19_masks[:, :, None], CACHE_SLICES, axis=2).sum())
    if _p19_tokens < int(0.55 * len(_p19_test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f"insufficient path19 test slices: {_p19_tokens}")

    _p19_encoder_path = _rad_find_file("ResNet50.pt", _RAD_ENCODER_SHA256)
    _p19_encoder = _RadEncoder()
    _p19_encoder.load_state_dict(
        torch.load(_p19_encoder_path, map_location="cpu", weights_only=True), strict=True
    )
    _p19_encoder.eval().to(_p19_device)
    for _p19_parameter in _p19_encoder.parameters():
        _p19_parameter.requires_grad_(False)
    if torch.cuda.device_count() > 1:
        _p19_encoder = nn.DataParallel(
            _p19_encoder, device_ids=list(range(torch.cuda.device_count()))
        )
    _p19_features, _p19_token_mask = _rad_encode(
        _p19_encoder, _p19_pixels, _p19_masks, _p19_device
    )
    del _p19_pixels, _p19_masks, _p19_headers
    gc.collect()

    _p19_member = []
    _p19_head_count = 0
    for _p19_record in _p19_folds:
        _p19_head = _RadHead().to(_p19_device).eval()
        _p19_head.load_state_dict(_p19_record["state_dict"], strict=True)
        _p19_member.append(
            _rad_predict_head(_p19_head, _p19_features, _p19_token_mask, _p19_device)
        )
        _p19_head_count += 1
        del _p19_head
        torch.cuda.empty_cache()
    _p19_pred = np.mean(np.stack(_p19_member), axis=0).astype(np.float64)
    del _p19_member, _p19_features, _p19_token_mask, _p19_encoder
    gc.collect()
    torch.cuda.empty_cache()
    if _p19_pred.shape != (len(_p19_test), len(TARGETS)) or not np.isfinite(_p19_pred).all():
        raise RuntimeError("Path19 non-finite Path14 predictions")

    _p19_raw = pd.DataFrame(_p19_pred, columns=TARGETS)
    _p19_raw.insert(0, "StudyInstanceUID", _p19_test.StudyInstanceUID)
    _rad_validate(_p19_raw, _p19_test.StudyInstanceUID)
    _p19_raw.to_csv(_p19_work / "submission_path19_path14_raw.csv", index=False)

    _p19_parent_rank = _rad_rank_columns(_p19_parent_frame[TARGETS].to_numpy())
    _p19_path14_rank = _rad_rank_columns(_p19_pred)
    _p19_final = _p19_parent_frame.copy()
    for _p19_index, _p19_target in enumerate(TARGETS):
        _p19_alpha = float(_P19_ALPHA[_p19_target])
        if _p19_alpha > 0.0:
            _p19_final[_p19_target] = (
                (1.0 - _p19_alpha) * _p19_parent_rank[:, _p19_index]
                + _p19_alpha * _p19_path14_rank[:, _p19_index]
            )
    _rad_validate(_p19_final, _p19_test.StudyInstanceUID)
    _p19_final.to_csv(_p19_primary, index=False)
    _p19_roundtrip = pd.read_csv(_p19_primary, dtype={"StudyInstanceUID": str})
    _rad_validate(_p19_roundtrip, _p19_test.StudyInstanceUID)

    _p19_audit.update(
        status="PATH19_PATH13_PATH14GOLD_APPLIED",
        parent_sha256=_p19_sha256(_p19_parent_preserved),
        path14_raw_sha256=_p19_sha256(_p19_work / "submission_path19_path14_raw.csv"),
        submission_sha256=_p19_sha256(_p19_primary),
        test_studies=int(len(_p19_test)),
        tokens=int(_p19_tokens),
        heads=int(_p19_head_count),
        schema_exact=True,
        finite_in_range=True,
    )
    log("Path19 Path14 reconciliation heads blended over the preserved Path13 0.920 parent")
except Exception as _p19_exc:
    _p19_error = _p19_exc
    _p19_audit["status"] = "PATH19_FALLBACK_PARENT_PRESERVED"
    _p19_audit["error"] = f"{type(_p19_exc).__name__}: {_p19_exc}"
    _p19_shutil.copy2(_p19_parent_preserved, _p19_primary)
finally:
    _p19_audit["primary_sha256"] = _p19_sha256(_p19_primary)
    _p19_audit_path.write_text(_p19_json.dumps(_p19_audit, indent=2, sort_keys=True) + "\n")

print(_p19_json.dumps(_p19_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"{SOURCE} not found; build Path13 first")
    notebook = json.loads(SOURCE.read_text())

    first = notebook["cells"][0]
    if first.get("cell_type") != "markdown":
        raise RuntimeError("Path13 cell 0 is not the expected markdown cell")
    joined = "".join(first["source"])
    joined = replace_once(joined, OLD_TITLE, NEW_TITLE, "markdown title")
    joined = replace_once(joined, OLD_BODY, NEW_BODY, "markdown body")
    first["source"] = [line + "\n" for line in joined.splitlines()]

    tail = notebook["cells"][-1]
    if tail.get("cell_type") != "code":
        raise RuntimeError("Path13 final cell is not the audit code cell")
    require_once("".join(tail["source"]), "PATH13_AMAN_SUPER_0920_VALID", "Path13 audit tail anchor")
    require_once(
        "".join(notebook["cells"][-2]["source"]),
        'master.rename(final)',
        "Path13 master-to-submission rename anchor",
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
