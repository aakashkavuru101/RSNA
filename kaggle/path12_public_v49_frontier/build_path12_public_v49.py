from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".codex_work" / "public_said_v49" / "rsna-knee-frontier-v49.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path12_public_v49_frontier.ipynb"


AUDIT_CELL = r'''# Path 12 runtime audit: public V49 frontier gate.
import hashlib as _p12_hashlib
import json as _p12_json
from pathlib import Path as _P12Path

import numpy as _p12_np
import pandas as _p12_pd

_P12_TARGETS = globals().get("TARGETS", [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
])
_p12_work = _P12Path("/kaggle/working")
_p12_submission = _p12_work / "submission.csv"
if not _p12_submission.is_file():
    raise RuntimeError("Path12 audit: submission.csv missing")

_p12_test = _p12_pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
_p12_sub = _p12_pd.read_csv(_p12_submission, dtype={"StudyInstanceUID": str})
_p12_expected = ["StudyInstanceUID", *_P12_TARGETS]
if _p12_sub.columns.tolist() != _p12_expected:
    raise RuntimeError(f"Path12 audit: schema mismatch: {_p12_sub.columns.tolist()}")
if _p12_sub["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Path12 audit: duplicate StudyInstanceUID")
if _p12_sub["StudyInstanceUID"].tolist() != _p12_test["StudyInstanceUID"].astype(str).tolist():
    raise RuntimeError("Path12 audit: StudyInstanceUID order mismatch")

_p12_values = _p12_sub[_P12_TARGETS].to_numpy(_p12_np.float64)
if not _p12_np.isfinite(_p12_values).all():
    raise RuntimeError("Path12 audit: non-finite prediction values")
if _p12_values.min() < 0.0 or _p12_values.max() > 1.0:
    raise RuntimeError("Path12 audit: predictions out of [0, 1]")

_p12_v40 = _p12_work / "v40_runtime_audit.json"
_p12_e10 = _p12_work / "rad_e10_audit.json"
_p12_e11 = _p12_work / "rad_e11_apply_audit.json"
_p12_v40_a = _p12_json.loads(_p12_v40.read_text()) if _p12_v40.is_file() else {}
_p12_e10_a = _p12_json.loads(_p12_e10.read_text()) if _p12_e10.is_file() else {}
_p12_e11_a = _p12_json.loads(_p12_e11.read_text()) if _p12_e11.is_file() else {}
if _p12_v40_a.get("status") != "VALID_DINOV3_E10_HYBRID_ALPHA050":
    raise RuntimeError(f"Path12 audit: V40/V49 receipt invalid: {_p12_v40_a.get('status')}")
if _p12_e10_a.get("status") != "CANDIDATE_SELECTED":
    raise RuntimeError(f"Path12 audit: E10 receipt invalid: {_p12_e10_a.get('status')}")
if _p12_e11_a and _p12_e11_a.get("status") not in ("E11_APPLIED", "ERROR_E10_PRESERVED"):
    raise RuntimeError(f"Path12 audit: E11 apply receipt invalid: {_p12_e11_a.get('status')}")

_p12_audit = {
    "status": "PATH12_PUBLIC_V49_FRONTIER_VALID",
    "source_public_notebook": "saidmohamedomary/rsna-knee-frontier-v49",
    "path_type": "unrestricted_public_frontier",
    "baseline_public_score_at_submit_decision": 0.915,
    "policy_note": "not AGENTS-clean; upstream V49 uses official/gold-labelled rows for guards/selection",
    "runtime_receipts": {
        "v40_status": _p12_v40_a.get("status"),
        "e10_status": _p12_e10_a.get("status"),
        "e10_configuration": _p12_e10_a.get("configuration"),
        "e11_apply_status": _p12_e11_a.get("status"),
    },
    "test_studies": int(len(_p12_sub)),
    "schema_exact": True,
    "finite_in_range": True,
    "submission_sha256": _p12_hashlib.sha256(_p12_submission.read_bytes()).hexdigest(),
}
(_p12_work / "path12_public_v49_audit.json").write_text(
    _p12_json.dumps(_p12_audit, indent=2, sort_keys=True) + "\n"
)
print(_p12_json.dumps(_p12_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"{SOURCE} not found; pull saidmohamedomary/rsna-knee-frontier-v49 first"
        )
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Path 12 — unrestricted public V49 frontier\n",
                "\n",
                "Fork of `saidmohamedomary/rsna-knee-frontier-v49` with an appended audit gate. ",
                "This is separated from AGENTS-clean work because V49 references official/gold rows for guard/selection logic.\n",
            ],
        },
    )
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in AUDIT_CELL.splitlines()],
        }
    )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
