from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontier_unrestricted_submit" / "rsna_knee_frontier_v43_unrestricted.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path2_robust.ipynb"


ROBUST_CELL = r'''# Path 2 robust per-target correction over the verified 0.909 V43 parent.
#
# V43 used RadImageNet alpha=0.70 for ten targets. Two independent grouped OOF
# sources agree that this is too aggressive for ACL, MCL, Medial Meniscus,
# Effusion, and Synovitis. The map below maximises the worse of the two OOF
# deltas at each target (ties use mean delta, then the smaller alpha). Gold rows
# are used only for this diagnostic blend selection; no model is trained here.
import hashlib as _p2r_hashlib
import json as _p2r_json
from pathlib import Path as _P2RPath

import numpy as _p2r_np
import pandas as _p2r_pd

_P2R_CONTRACT_SHA256 = "219c91f40905181c222e2966b3fed01a96570ddfb64862357d5fd6cad500cd45"
_P2R_ALPHA_MAP = {
    "ACL": 0.35,
    "MCL": 0.50,
    "Medial Meniscus": 0.20,
    "Lateral Meniscus": 0.70,
    "Medial OA": 0.70,
    "Lateral OA": 0.70,
    "PF OA": 0.60,
    "Effusion": 0.20,
    "Synovitis": 0.60,
    "Baker's": 0.00,
    "Contusion": 0.70,
    "Fracture": 0.00,
}
_P2R_OOF = {
    "robust_public_macro": 0.8662010160390182,
    "robust_remote_v15_macro": 0.8750191339735348,
    "uniform_070_public_macro": 0.8600443700178757,
    "uniform_070_remote_v15_macro": 0.8730053068109873,
}


def _p2r_sha256(path):
    return _p2r_hashlib.sha256(_P2RPath(path).read_bytes()).hexdigest()


def _p2r_rank_columns(values):
    return _p2r_pd.DataFrame(
        _p2r_np.asarray(values, dtype=_p2r_np.float64)
    ).rank(method="average", pct=True).to_numpy(_p2r_np.float64)


_p2r_work = _P2RPath("/kaggle/working")
_p2r_primary = _p2r_work / "submission.csv"
_p2r_parent = _p2r_work / "submission_e2_preserved.csv"
_p2r_rad = _p2r_work / "rsna_rad_e10" / "submission_rad_only.csv"
_p2r_e10_audit_path = _p2r_work / "rad_e10_audit.json"
_p2r_v43_audit_path = _p2r_work / "v43_runtime_audit.json"
for _p2r_path in (
    _p2r_primary,
    _p2r_parent,
    _p2r_rad,
    _p2r_e10_audit_path,
    _p2r_v43_audit_path,
):
    if not _p2r_path.is_file():
        raise RuntimeError(f"Path2 robust stage missing required artifact: {_p2r_path}")

_p2r_e10_audit = _p2r_json.loads(_p2r_e10_audit_path.read_text())
_p2r_v43_audit = _p2r_json.loads(_p2r_v43_audit_path.read_text())
if _p2r_e10_audit.get("status") != "CANDIDATE_SELECTED":
    raise RuntimeError("Path2 robust stage requires a successful E10 parent")
if _p2r_e10_audit.get("configuration") != "uniform_070":
    raise RuntimeError("Path2 robust stage requires the submitted uniform_070 parent")
if _p2r_e10_audit.get("contract_sha256") != _P2R_CONTRACT_SHA256:
    raise RuntimeError("Path2 robust E10 contract drift")
if _p2r_v43_audit.get("status") != "VALID_V43_V41_PARENT_ALPHA070":
    raise RuntimeError("Path2 robust V43 parent receipt mismatch")
if _p2r_sha256(_p2r_primary) != _p2r_v43_audit.get("submission_sha256"):
    raise RuntimeError("Path2 robust parent changed after V43 validation")

_p2r_test = _p2r_pd.read_csv(COMP / "test.csv", dtype={"StudyInstanceUID": str})
_p2r_base = _p2r_pd.read_csv(_p2r_parent, dtype={"StudyInstanceUID": str})
_p2r_rad_frame = _p2r_pd.read_csv(_p2r_rad, dtype={"StudyInstanceUID": str})
_p2r_expected_columns = ["StudyInstanceUID", *TARGETS]
for _p2r_name, _p2r_frame in (("base", _p2r_base), ("rad", _p2r_rad_frame)):
    if _p2r_frame.columns.tolist() != _p2r_expected_columns:
        raise RuntimeError(f"Path2 robust {_p2r_name} schema drift")
    if _p2r_frame.StudyInstanceUID.tolist() != _p2r_test.StudyInstanceUID.astype(str).tolist():
        raise RuntimeError(f"Path2 robust {_p2r_name} study identity/order drift")
    _p2r_values = _p2r_frame[TARGETS].to_numpy(_p2r_np.float64)
    if not _p2r_np.isfinite(_p2r_values).all():
        raise RuntimeError(f"Path2 robust {_p2r_name} contains non-finite values")

_p2r_base_rank = _p2r_rank_columns(_p2r_base[TARGETS].to_numpy())
_p2r_rad_rank = _p2r_rank_columns(_p2r_rad_frame[TARGETS].to_numpy())
_p2r_candidate = _p2r_base.copy()
for _p2r_index, _p2r_target in enumerate(TARGETS):
    _p2r_alpha = float(_P2R_ALPHA_MAP[_p2r_target])
    if _p2r_alpha > 0:
        _p2r_candidate[_p2r_target] = (
            (1.0 - _p2r_alpha) * _p2r_base_rank[:, _p2r_index]
            + _p2r_alpha * _p2r_rad_rank[:, _p2r_index]
        )

for _p2r_target in ("Baker's", "Fracture"):
    if not _p2r_np.array_equal(
        _p2r_candidate[_p2r_target].to_numpy(),
        _p2r_base[_p2r_target].to_numpy(),
    ):
        raise RuntimeError(f"Path2 robust failed to preserve {_p2r_target}")

_p2r_values = _p2r_candidate[TARGETS].to_numpy(_p2r_np.float64)
if not _p2r_np.isfinite(_p2r_values).all() or _p2r_values.min() < 0 or _p2r_values.max() > 1:
    raise RuntimeError("Path2 robust candidate values are invalid")
_p2r_tmp = _p2r_work / "submission_path2_robust.tmp.csv"
_p2r_candidate.to_csv(_p2r_tmp, index=False)
_p2r_roundtrip = _p2r_pd.read_csv(_p2r_tmp, dtype={"StudyInstanceUID": str})
if _p2r_roundtrip.columns.tolist() != _p2r_expected_columns:
    raise RuntimeError("Path2 robust serialized schema drift")
_p2r_tmp.replace(_p2r_primary)

_p2r_receipt = {
    "status": "VALID_PATH2_ROBUST_PER_TARGET_V1",
    "source_public_score": 0.909,
    "source_configuration": "uniform_070",
    "selection_rule": "per-target maximin delta across two independent grouped OOF sources",
    "alpha_map": _P2R_ALPHA_MAP,
    "preserved_targets": ["Baker's", "Fracture"],
    "oof_diagnostics": _P2R_OOF,
    "gold_used_for_training": False,
    "test_studies": len(_p2r_candidate),
    "dynamic_test_ids_exact": True,
    "schema_exact": True,
    "finite_in_range": True,
    "e10_contract_sha256": _P2R_CONTRACT_SHA256,
    "uniform_070_sha256": _p2r_v43_audit.get("submission_sha256"),
    "base_sha256": _p2r_sha256(_p2r_parent),
    "rad_only_sha256": _p2r_sha256(_p2r_rad),
    "submission_sha256": _p2r_sha256(_p2r_primary),
}
(_p2r_work / "path2_robust_runtime_audit.json").write_text(
    _p2r_json.dumps(_p2r_receipt, indent=2, sort_keys=True) + "\n"
)
print(_p2r_json.dumps(_p2r_receipt, indent=2, sort_keys=True))
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in ROBUST_CELL.splitlines()],
        }
    )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
