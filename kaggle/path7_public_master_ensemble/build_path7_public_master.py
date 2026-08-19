from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".codex_work" / "public_sakhawat_enhanced_ensemble" / "rsna-knee-enhanced-ensemble.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path7_public_master_ensemble.ipynb"


AUDIT_CELL = r'''# Path 7 runtime audit: public master ensemble submission gate.
import hashlib as _p7_hashlib
import json as _p7_json
from pathlib import Path as _P7Path

import numpy as _p7_np
import pandas as _p7_pd

_P7_TARGETS = globals().get("TARGETS", [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
])
_p7_work = _P7Path("/kaggle/working")
_p7_submission = _p7_work / "submission.csv"
if not _p7_submission.is_file():
    raise RuntimeError("Path7 audit: submission.csv missing")

_p7_test = _p7_pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
_p7_sub = _p7_pd.read_csv(_p7_submission, dtype={"StudyInstanceUID": str})
_p7_expected = ["StudyInstanceUID", *_P7_TARGETS]
if _p7_sub.columns.tolist() != _p7_expected:
    raise RuntimeError(f"Path7 audit: schema mismatch: {_p7_sub.columns.tolist()}")
if _p7_sub["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Path7 audit: duplicate StudyInstanceUID")
if _p7_sub["StudyInstanceUID"].tolist() != _p7_test["StudyInstanceUID"].astype(str).tolist():
    raise RuntimeError("Path7 audit: StudyInstanceUID order mismatch")

_p7_values = _p7_sub[_P7_TARGETS].to_numpy(_p7_np.float64)
if not _p7_np.isfinite(_p7_values).all():
    raise RuntimeError("Path7 audit: non-finite prediction values")
if _p7_values.min() < 0.0 or _p7_values.max() > 1.0:
    raise RuntimeError("Path7 audit: predictions out of [0, 1]")

_p7_audit = {
    "status": "PATH7_PUBLIC_MASTER_VALID",
    "source_public_notebook": "sakhawathossen/rsna-knee-enhanced-ensemble",
    "path_type": "unrestricted_public_artifact",
    "baseline_public_score_at_submit_decision": 0.912,
    "policy_note": "not AGENTS-clean; public notebook/artifact path references official 58-label rows internally",
    "test_studies": int(len(_p7_sub)),
    "schema_exact": True,
    "finite_in_range": True,
    "submission_sha256": _p7_hashlib.sha256(_p7_submission.read_bytes()).hexdigest(),
}
(_p7_work / "path7_public_master_audit.json").write_text(
    _p7_json.dumps(_p7_audit, indent=2, sort_keys=True) + "\n"
)
print(_p7_json.dumps(_p7_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"{SOURCE} not found; pull sakhawathossen/rsna-knee-enhanced-ensemble first"
        )
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Path 7 — unrestricted public master ensemble\n",
                "\n",
                "Fork of `sakhawathossen/rsna-knee-enhanced-ensemble`, kept as an unrestricted public-artifact candidate. ",
                "This is intentionally separate from AGENTS-clean work because the upstream public notebook references the 58 official annotation rows internally.\n",
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
