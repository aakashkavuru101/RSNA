from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".codex_work" / "public_aman_super_0920_meta" / "rsna-knee-super-ensemble-0920.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path13_aman_super_0920.ipynb"


AUDIT_CELL = r'''# Path 13 runtime audit: Aman Super Ensemble 0920 gate.
import hashlib as _p13_hashlib
import json as _p13_json
from pathlib import Path as _P13Path

import numpy as _p13_np
import pandas as _p13_pd

_P13_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
_p13_work = _P13Path("/kaggle/working")
_p13_submission = _p13_work / "submission.csv"
if not _p13_submission.is_file():
    raise RuntimeError("Path13 audit: submission.csv missing")

_p13_comp_candidates = [
    _P13Path("/kaggle/input/rsna-knee-abnormality-detection"),
    _P13Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
]
_p13_comp = next((p for p in _p13_comp_candidates if (p / "test.csv").is_file()), None)
if _p13_comp is None:
    raise RuntimeError("Path13 audit: competition test.csv missing")

_p13_test = _p13_pd.read_csv(_p13_comp / "test.csv", dtype={"StudyInstanceUID": str})
_p13_sub = _p13_pd.read_csv(_p13_submission, dtype={"StudyInstanceUID": str})
_p13_expected = ["StudyInstanceUID", *_P13_TARGETS]
if _p13_sub.columns.tolist() != _p13_expected:
    raise RuntimeError(f"Path13 audit: schema mismatch: {_p13_sub.columns.tolist()}")
if _p13_sub["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Path13 audit: duplicate StudyInstanceUID")
if _p13_sub["StudyInstanceUID"].tolist() != _p13_test["StudyInstanceUID"].astype(str).tolist():
    raise RuntimeError("Path13 audit: StudyInstanceUID order mismatch")

_p13_values = _p13_sub[_P13_TARGETS].to_numpy(_p13_np.float64)
if not _p13_np.isfinite(_p13_values).all():
    raise RuntimeError("Path13 audit: non-finite prediction values")
if _p13_values.min() < 0.0 or _p13_values.max() > 1.0:
    raise RuntimeError("Path13 audit: predictions out of [0, 1]")

_p13_audit = {
    "status": "PATH13_AMAN_SUPER_0920_VALID",
    "source_public_notebook": "amanatar/rsna-knee-super-ensemble-0920",
    "path_type": "unrestricted_public_super_ensemble",
    "baseline_public_score_at_build": 0.915,
    "strategy": "Aman 0920 V48/E13 adaptive RadImageNet finalizer; structurally different from Path5/10 Kimi plateau and Path12 fixed V49 E10/E11.",
    "test_studies": int(len(_p13_sub)),
    "schema_exact": True,
    "finite_in_range": True,
    "submission_sha256": _p13_hashlib.sha256(_p13_submission.read_bytes()).hexdigest(),
}
(_p13_work / "path13_aman_super_0920_audit.json").write_text(
    _p13_json.dumps(_p13_audit, indent=2, sort_keys=True) + "\n"
)
print(_p13_json.dumps(_p13_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"{SOURCE} not found; pull amanatar/rsna-knee-super-ensemble-0920 first")
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Path 13 — Aman Super Ensemble 0920\n",
                "\n",
                "Fork of `amanatar/rsna-knee-super-ensemble-0920` with an appended audit gate.\n",
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
