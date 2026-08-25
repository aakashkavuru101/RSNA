from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path13_aman_super_0920" / "rsna_knee_path13_aman_super_0920.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path18_path13_goldstar_apply.ipynb"


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
NEW_TITLE = "# Path 18 — Path13 super-0920 parent + Path17 λ* gold heads"

OLD_BODY = "Fork of `amanatar/rsna-knee-super-ensemble-0920` with an appended audit gate."
NEW_BODY = (
    "Exact clone of the Path13 fork of `amanatar/rsna-knee-super-ensemble-0920` "
    "(public LB 0.920), including its appended audit gate, plus one final stage: a "
    "per-target rank blend of the Path17 gold-crossfit deployment heads at each "
    "target's λ* on the seven targets where Path17 λ* cross-fitted gold AUC beat the "
    "Path6 parent's gold monitor by >0.01 — Baker's 0.15, Contusion 0.15, Fracture "
    "0.15, Lateral Meniscus 0.15, Lateral OA 0.10, Medial Meniscus 0.10, Medial OA "
    "0.10. Heads are located by filename plus pinned sha256 from the "
    "`rsna-knee-path17-gold-crossfit-train` kernel output. Any guard failure "
    "restores the preserved Path13 submission byte-for-byte."
)


FINAL_CELL = r'''# Path 18 finalizer: Path13 Aman super-0920 parent + Path17 per-target lambda-star gold heads.
import hashlib as _p18_hashlib
import json as _p18_json
import shutil as _p18_shutil
from pathlib import Path as _P18Path

_P18_HEADS_NAME = "path17_gold_heads.pt"
_P18_HEADS_SHA256 = "cf8862992064a3075d9a5f788abecf78d80a334e5af275c9d9cbd5c41d28a355"
_P18_OOF_SHA256 = "c65083f6558f848d9bde135147d646ceb97b609e600fb4680f54d42b19b71cce"
_P18_BUNDLE_VERSION = "path17-gold-crossfit-1"
_P18_LAMBDA_STAR = {
    "ACL": 2.0,
    "Baker's": 2.0,
    "Contusion": 8.0,
    "Effusion": 4.0,
    "Fracture": 4.0,
    "Lateral Meniscus": 1.0,
    "Lateral OA": 4.0,
    "MCL": 1.0,
    "Medial Meniscus": 1.0,
    "Medial OA": 4.0,
    "PF OA": 2.0,
    "Synovitis": 4.0,
}
_P18_ALPHA = {
    "ACL": 0.00,
    "MCL": 0.00,
    "Medial Meniscus": 0.10,
    "Lateral Meniscus": 0.15,
    "Medial OA": 0.10,
    "Lateral OA": 0.10,
    "PF OA": 0.00,
    "Effusion": 0.00,
    "Synovitis": 0.00,
    "Baker's": 0.15,
    "Contusion": 0.15,
    "Fracture": 0.15,
}
_P18_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("COR_NOFS", "Coronal", None, False),
]


def _p18_sha256(path):
    return _p18_hashlib.sha256(_P18Path(path).read_bytes()).hexdigest()


def _p18_find_file_by_sha256(name, digest):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            path = _P18Path(root) / name
            if _p18_sha256(path) == digest:
                return path
            hits.append(str(path))
    raise FileNotFoundError(f"{name} with sha256 {digest} not found; candidates={hits[:8]}")


_p18_work = _P18Path("/kaggle/working")
_p18_primary = _p18_work / "submission.csv"
_p18_parent_preserved = _p18_work / "submission_path13_0920_preserved.csv"
_p18_parent_audit_path = _p18_work / "path13_aman_super_0920_audit.json"
_p18_audit_path = _p18_work / "path18_path13_goldstar_apply_audit.json"
_p18_audit = {
    "status": "PATH18_PENDING",
    "strategy": "Path13 Aman super-0920 parent preserved, then per-target rank blend of Path17 lambda-star gold-crossfit deployment heads on seven gold-evidence targets",
    "parent": "Path13 Aman super-0920, public score 0.920",
    "alpha_map": _P18_ALPHA,
    "lambda_star": _P18_LAMBDA_STAR,
    "path17_heads_sha256": _P18_HEADS_SHA256,
    "path17_oof_sha256": _P18_OOF_SHA256,
    "gold_usage": "crossfit",
    "gold_policy": "Path17 component trained with all 58 gold rows overridden under the GOLD_INTEGRATION_PLAN.md section 2 cross-fit protocol; alpha map selected on cross-fitted gold OOF only",
    "test_studies_expected": 3,
}
if not _p18_primary.is_file() or not _p18_parent_audit_path.is_file():
    raise RuntimeError("Path18 requires completed Path13 artifacts")
_p18_parent_receipt = _p18_json.loads(_p18_parent_audit_path.read_text())
if _p18_parent_receipt.get("status") != "PATH13_AMAN_SUPER_0920_VALID":
    raise RuntimeError("Path18 Path13 audit mismatch")
if _p18_sha256(_p18_primary) != _p18_parent_receipt.get("submission_sha256"):
    raise RuntimeError("Path18 Path13 primary hash drift")
_p18_shutil.copy2(_p18_primary, _p18_parent_preserved)

_p18_error = None
try:
    _p18_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if _p18_device.type != "cuda":
        raise RuntimeError("Path18 Path17 apply requires CUDA")
    _p18_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    _p18_parent_frame = pd.read_csv(_p18_parent_preserved, dtype={"StudyInstanceUID": str})
    _rad_validate(_p18_parent_frame, _p18_test.StudyInstanceUID)
    if len(_p18_test) != 3 or len(_p18_parent_frame) != 3:
        raise RuntimeError("Path18 parent row count drift")

    _p18_heads_path = _p18_find_file_by_sha256(_P18_HEADS_NAME, _P18_HEADS_SHA256)
    _p18_bundle = torch.load(_p18_heads_path, map_location="cpu", weights_only=False)
    if _p18_bundle.get("version") != _P18_BUNDLE_VERSION:
        raise RuntimeError("Path17 payload version drift")
    if _p18_bundle.get("gold_usage") != "crossfit":
        raise RuntimeError("Path17 cross-fit receipt absent")
    if _p18_bundle.get("cv_grouping") != "scanner-site DICOM signature":
        raise RuntimeError("Path17 scanner-grouped receipt absent")
    if _p18_bundle.get("targets") != TARGETS:
        raise RuntimeError("Path17 target order drift")
    if _p18_bundle.get("encoder_sha256") != _RAD_ENCODER_SHA256:
        raise RuntimeError("Path17 encoder pin drift")
    if _p18_bundle.get("slots") != [list(slot) for slot in _P18_SLOTS]:
        raise RuntimeError("Path17 slot recipe drift")
    if (
        float(_p18_bundle.get("crop_mm")) != 160.0
        or int(_p18_bundle.get("img")) != 224
        or int(_p18_bundle.get("slices_per_plane")) != 8
    ):
        raise RuntimeError("Path17 pixel recipe drift")
    _p18_bundle_lambda = {
        target: float(value) for target, value in _p18_bundle.get("lambda_star", {}).items()
    }
    if _p18_bundle_lambda != {target: float(value) for target, value in _P18_LAMBDA_STAR.items()}:
        raise RuntimeError("Path17 lambda-star drift")
    _p18_variants = _p18_bundle.get("gold_fold_variants", {})
    _p18_needed = sorted({f"{_P18_LAMBDA_STAR[target]:.1f}" for target in TARGETS})
    for _p18_key in _p18_needed:
        _p18_folds = _p18_variants.get(_p18_key)
        if not isinstance(_p18_folds, list) or len(_p18_folds) != 5:
            raise RuntimeError(f"Path17 lambda variant {_p18_key} missing five deployment folds")
    for _p18_target, _p18_alpha in _P18_ALPHA.items():
        if _p18_alpha > 0.0 and f"{_P18_LAMBDA_STAR[_p18_target]:.1f}" not in _p18_variants:
            raise RuntimeError(f"Path17 missing lambda variant for blend target {_p18_target}")

    globals().update(
        SLOTS=[tuple(slot) for slot in _P18_SLOTS],
        N_SLOT=len(_P18_SLOTS),
        CACHE_SLICES=8,
        IMG=224,
        CACHE_IMG=224,
        CROP_MM=160.0,
        SLICE_BAND=(0.05, 0.95),
        RULES=dict(RULES_LEGACY),
    )
    _p18_series = pd.read_csv(
        ROOT / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    _p18_plane = dict(zip(_p18_series.SeriesInstanceUID, _p18_series.Anatomical_Plane))
    _p18_headers = annotate(walk("test_series"))
    _p18_studies, _p18_pixels, _p18_masks = build_cache(
        pick_slots(_p18_headers, _p18_plane),
        _p18_plane,
        lat_of(_p18_headers, "test-path18-path17 "),
        "test-path18-path17",
    )
    _p18_positions = {str(uid): index for index, uid in enumerate(_p18_studies)}
    _p18_missing = [uid for uid in _p18_test.StudyInstanceUID if uid not in _p18_positions]
    if _p18_missing:
        raise RuntimeError(f"{len(_p18_missing)} test studies absent from path18 cache")
    _p18_order = np.asarray(
        [_p18_positions[uid] for uid in _p18_test.StudyInstanceUID], dtype=np.int64
    )
    _p18_pixels, _p18_masks = _p18_pixels[_p18_order], _p18_masks[_p18_order]
    _p18_tokens = int(np.repeat(_p18_masks[:, :, None], CACHE_SLICES, axis=2).sum())
    if _p18_tokens < int(0.55 * len(_p18_test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f"insufficient path18 test slices: {_p18_tokens}")

    _p18_encoder_path = _rad_find_file("ResNet50.pt", _RAD_ENCODER_SHA256)
    _p18_encoder = _RadEncoder()
    _p18_encoder.load_state_dict(
        torch.load(_p18_encoder_path, map_location="cpu", weights_only=True), strict=True
    )
    _p18_encoder.eval().to(_p18_device)
    for _p18_parameter in _p18_encoder.parameters():
        _p18_parameter.requires_grad_(False)
    if torch.cuda.device_count() > 1:
        _p18_encoder = nn.DataParallel(
            _p18_encoder, device_ids=list(range(torch.cuda.device_count()))
        )
    _p18_features, _p18_token_mask = _rad_encode(
        _p18_encoder, _p18_pixels, _p18_masks, _p18_device
    )
    del _p18_pixels, _p18_masks, _p18_headers
    gc.collect()

    _p18_pred = np.full((len(_p18_test), len(TARGETS)), np.nan, dtype=np.float64)
    _p18_head_count = 0
    for _p18_key in _p18_needed:
        _p18_columns = [
            index
            for index, target in enumerate(TARGETS)
            if f"{_P18_LAMBDA_STAR[target]:.1f}" == _p18_key
        ]
        if not _p18_columns:
            continue
        _p18_member = []
        for _p18_record in _p18_variants[_p18_key]:
            _p18_head = _RadHead().to(_p18_device).eval()
            _p18_head.load_state_dict(_p18_record["state_dict"], strict=True)
            _p18_member.append(
                _rad_predict_head(_p18_head, _p18_features, _p18_token_mask, _p18_device)
            )
            _p18_head_count += 1
            del _p18_head
            torch.cuda.empty_cache()
        _p18_prob = np.mean(np.stack(_p18_member), axis=0)
        _p18_pred[:, _p18_columns] = _p18_prob[:, _p18_columns]
        del _p18_member
    del _p18_features, _p18_token_mask, _p18_encoder
    gc.collect()
    torch.cuda.empty_cache()
    if _p18_pred.shape != (len(_p18_test), len(TARGETS)) or not np.isfinite(_p18_pred).all():
        raise RuntimeError("Path18 non-finite Path17 predictions")

    _p18_raw = pd.DataFrame(_p18_pred, columns=TARGETS)
    _p18_raw.insert(0, "StudyInstanceUID", _p18_test.StudyInstanceUID)
    _rad_validate(_p18_raw, _p18_test.StudyInstanceUID)
    _p18_raw.to_csv(_p18_work / "submission_path18_path17_goldstar_raw.csv", index=False)

    _p18_parent_rank = _rad_rank_columns(_p18_parent_frame[TARGETS].to_numpy())
    _p18_path17_rank = _rad_rank_columns(_p18_pred)
    _p18_final = _p18_parent_frame.copy()
    for _p18_index, _p18_target in enumerate(TARGETS):
        _p18_alpha = float(_P18_ALPHA[_p18_target])
        if _p18_alpha > 0.0:
            _p18_final[_p18_target] = (
                (1.0 - _p18_alpha) * _p18_parent_rank[:, _p18_index]
                + _p18_alpha * _p18_path17_rank[:, _p18_index]
            )
    _rad_validate(_p18_final, _p18_test.StudyInstanceUID)
    _p18_final.to_csv(_p18_primary, index=False)
    _p18_roundtrip = pd.read_csv(_p18_primary, dtype={"StudyInstanceUID": str})
    _rad_validate(_p18_roundtrip, _p18_test.StudyInstanceUID)

    _p18_audit.update(
        status="PATH18_PATH13_GOLDSTAR_APPLIED",
        parent_sha256=_p18_sha256(_p18_parent_preserved),
        path17_raw_sha256=_p18_sha256(_p18_work / "submission_path18_path17_goldstar_raw.csv"),
        submission_sha256=_p18_sha256(_p18_primary),
        test_studies=int(len(_p18_test)),
        tokens=int(_p18_tokens),
        heads=int(_p18_head_count),
        schema_exact=True,
        finite_in_range=True,
    )
    log("Path18 Path17 lambda-star gold heads blended over the preserved Path13 0.920 parent")
except Exception as _p18_exc:
    _p18_error = _p18_exc
    _p18_audit["status"] = "PATH18_FALLBACK_PARENT_PRESERVED"
    _p18_audit["error"] = f"{type(_p18_exc).__name__}: {_p18_exc}"
    _p18_shutil.copy2(_p18_parent_preserved, _p18_primary)
finally:
    _p18_audit["primary_sha256"] = _p18_sha256(_p18_primary)
    _p18_audit_path.write_text(_p18_json.dumps(_p18_audit, indent=2, sort_keys=True) + "\n")

print(_p18_json.dumps(_p18_audit, indent=2, sort_keys=True))
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
