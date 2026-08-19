from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".codex_work" / "public_sakhawat_enhanced_ensemble" / "rsna-knee-enhanced-ensemble.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path9_public_master_b3.ipynb"


AUDIT_CELL = r'''# Path 9 runtime audit: public master ensemble with required B3 diversity.
import hashlib as _p9_hashlib
import json as _p9_json
from pathlib import Path as _P9Path

import numpy as _p9_np
import pandas as _p9_pd

_P9_TARGETS = globals().get("TARGETS", [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
])
_p9_work = _P9Path("/kaggle/working")
_p9_submission = _p9_work / "submission.csv"
if not _p9_submission.is_file():
    raise RuntimeError("Path9 audit: submission.csv missing")

_p9_b3_alpha = float(globals().get("_b3_alpha", 0.0))
_p9_legacy_alpha = float(globals().get("_legacy_alpha", 0.0))
_p9_parent_alpha = float(globals().get("_parent_alpha", 0.0))
if _p9_b3_alpha <= 0.0:
    raise RuntimeError("Path9 audit: B3 package was not used; refusing B3 candidate")

_p9_test = _p9_pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
_p9_sub = _p9_pd.read_csv(_p9_submission, dtype={"StudyInstanceUID": str})
_p9_expected = ["StudyInstanceUID", *_P9_TARGETS]
if _p9_sub.columns.tolist() != _p9_expected:
    raise RuntimeError(f"Path9 audit: schema mismatch: {_p9_sub.columns.tolist()}")
if _p9_sub["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Path9 audit: duplicate StudyInstanceUID")
if _p9_sub["StudyInstanceUID"].tolist() != _p9_test["StudyInstanceUID"].astype(str).tolist():
    raise RuntimeError("Path9 audit: StudyInstanceUID order mismatch")

_p9_values = _p9_sub[_P9_TARGETS].to_numpy(_p9_np.float64)
if not _p9_np.isfinite(_p9_values).all():
    raise RuntimeError("Path9 audit: non-finite prediction values")
if _p9_values.min() < 0.0 or _p9_values.max() > 1.0:
    raise RuntimeError("Path9 audit: predictions out of [0, 1]")

_p9_audit = {
    "status": "PATH9_PUBLIC_MASTER_B3_VALID",
    "source_public_notebook": "sakhawathossen/rsna-knee-enhanced-ensemble",
    "b3_dataset": "prvsiyan/rsna-knee-b3-v47-public-deployment",
    "path_type": "unrestricted_public_artifact",
    "baseline_public_score_at_submit_decision": 0.912,
    "policy_note": "not AGENTS-clean; public notebook/artifact path references official 58-label rows internally",
    "blend": {
        "parent_alpha": _p9_parent_alpha,
        "legacy_alpha": _p9_legacy_alpha,
        "b3_alpha": _p9_b3_alpha,
    },
    "b3_oof_reference": {
        "exact_public_macro_auc": 0.8434849072528627,
        "global_nested_macro_auc": 0.8463056657160722,
        "target_nested_macro_auc": 0.8548132525326522,
    },
    "test_studies": int(len(_p9_sub)),
    "schema_exact": True,
    "finite_in_range": True,
    "submission_sha256": _p9_hashlib.sha256(_p9_submission.read_bytes()).hexdigest(),
}
(_p9_work / "path9_public_master_b3_audit.json").write_text(
    _p9_json.dumps(_p9_audit, indent=2, sort_keys=True) + "\n"
)
print(_p9_json.dumps(_p9_audit, indent=2, sort_keys=True))
'''


B3_ENV_CELL = r'''# Path 9 B3 mount hint.
# Kaggle mounts dataset sources under /kaggle/input/datasets/<owner>/<slug> in
# this environment. The upstream public notebook also supports an explicit path.
import os as _p9_b3_os

_p9_b3_os.environ.setdefault(
    "KNEE_B3_DIR",
    "/kaggle/input/datasets/prvsiyan/rsna-knee-b3-v47-public-deployment",
)
print("KNEE_B3_DIR =", _p9_b3_os.environ["KNEE_B3_DIR"])
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
                "# Path 9 — unrestricted public master ensemble with B3\n",
                "\n",
                "Fork of `sakhawathossen/rsna-knee-enhanced-ensemble` with the public B3 V47 deployment package attached. ",
                "The final audit refuses the run if B3 is not actually used.\n",
            ],
        },
    )
    notebook["cells"].insert(
        1,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in B3_ENV_CELL.splitlines()],
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
