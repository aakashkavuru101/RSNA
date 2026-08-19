from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".codex_work" / "public_mattia_dinov3_ensembled" / "bend-the-knee-to-dinov3-ensembled.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path8_public_dinov3_target.ipynb"


AUDIT_CELL = r'''# Path 8 runtime audit: public DINOv3/RadImageNet target recipe gate.
import hashlib as _p8_hashlib
import json as _p8_json
from pathlib import Path as _P8Path

import numpy as _p8_np
import pandas as _p8_pd

_P8_TARGETS = globals().get("TARGETS", [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
])
_p8_work = _P8Path("/kaggle/working")
_p8_submission = _p8_work / "submission.csv"
if not _p8_submission.is_file():
    raise RuntimeError("Path8 audit: submission.csv missing")

_p8_test = _p8_pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
_p8_sub = _p8_pd.read_csv(_p8_submission, dtype={"StudyInstanceUID": str})
_p8_expected = ["StudyInstanceUID", *_P8_TARGETS]
if _p8_sub.columns.tolist() != _p8_expected:
    raise RuntimeError(f"Path8 audit: schema mismatch: {_p8_sub.columns.tolist()}")
if _p8_sub["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Path8 audit: duplicate StudyInstanceUID")
if _p8_sub["StudyInstanceUID"].tolist() != _p8_test["StudyInstanceUID"].astype(str).tolist():
    raise RuntimeError("Path8 audit: StudyInstanceUID order mismatch")

_p8_values = _p8_sub[_P8_TARGETS].to_numpy(_p8_np.float64)
if not _p8_np.isfinite(_p8_values).all():
    raise RuntimeError("Path8 audit: non-finite prediction values")
if _p8_values.min() < 0.0 or _p8_values.max() > 1.0:
    raise RuntimeError("Path8 audit: predictions out of [0, 1]")

_p8_audit = {
    "status": "PATH8_PUBLIC_DINOV3_TARGET_VALID",
    "source_public_notebook": "mattiaangeli/bend-the-knee-to-dinov3-ensembled",
    "path_type": "unrestricted_public_artifact",
    "baseline_public_score_at_submit_decision": 0.912,
    "policy_note": "not AGENTS-clean; public notebook/artifact path references official 58-label rows internally",
    "test_studies": int(len(_p8_sub)),
    "schema_exact": True,
    "finite_in_range": True,
    "submission_sha256": _p8_hashlib.sha256(_p8_submission.read_bytes()).hexdigest(),
}
(_p8_work / "path8_public_dinov3_audit.json").write_text(
    _p8_json.dumps(_p8_audit, indent=2, sort_keys=True) + "\n"
)
print(_p8_json.dumps(_p8_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"{SOURCE} not found; pull mattiaangeli/bend-the-knee-to-dinov3-ensembled first"
        )
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Path 8 — unrestricted public DINOv3 target recipe\n",
                "\n",
                "Fork of `mattiaangeli/bend-the-knee-to-dinov3-ensembled`, kept as an unrestricted public-artifact candidate. ",
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
