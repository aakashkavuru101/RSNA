from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path10_kimi_fs160_stack_apply" / "rsna_knee_path10_kimi_fs160_stack_apply.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path11_kimi_fs160_replace_apply.ipynb"


FINAL_CELL = r'''# Path 11 finalizer: submit Path6 replacement over 0.912 base instead of Path6-on-Path5 stack.
import hashlib as _p11_hashlib
import json as _p11_json
import shutil as _p11_shutil
from pathlib import Path as _P11Path

_p11_work = _P11Path("/kaggle/working")
_p11_replace = _p11_work / "submission_path10_path6_replace.csv"
_p11_primary = _p11_work / "submission.csv"
_p11_path10_audit = _p11_work / "path10_kimi_fs160_stack_audit.json"
if not _p11_replace.is_file() or not _p11_path10_audit.is_file():
    raise RuntimeError("Path11 requires completed Path10 replacement artifact")
_p11_receipt = _p11_json.loads(_p11_path10_audit.read_text())
if _p11_receipt.get("status") != "PATH10_KIMI_FS160_STACK_APPLIED":
    raise RuntimeError("Path11 Path10 audit mismatch")
_p11_shutil.copy2(_p11_replace, _p11_primary)

_p11_test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
_p11_frame = pd.read_csv(_p11_primary, dtype={"StudyInstanceUID": str})
_v52_validate_submission(_p11_frame, _p11_test.StudyInstanceUID)

_p11_audit = {
    "status": "PATH11_KIMI_FS160_REPLACE_APPLIED",
    "source_public_score": 0.915,
    "strategy": "Path6 Kimi-FS160 replacement over Path2 0.912 base; not stacked on Path5",
    "why": "fallback if Path10 stack dilutes Path5 despite stronger Path6 diagnostics",
    "path10_audit_status": _p11_receipt.get("status"),
    "path6_heads_sha256": _p11_receipt.get("path6_heads_sha256"),
    "replace_alpha_map": _p11_receipt.get("replace_alpha_map"),
    "gold_policy": _p11_receipt.get("gold_policy"),
    "test_studies": int(len(_p11_frame)),
    "schema_exact": True,
    "finite_in_range": True,
    "submission_sha256": _p11_hashlib.sha256(_p11_primary.read_bytes()).hexdigest(),
}
(_p11_work / "path11_kimi_fs160_replace_audit.json").write_text(
    _p11_json.dumps(_p11_audit, indent=2, sort_keys=True) + "\n"
)
print(_p11_json.dumps(_p11_audit, indent=2, sort_keys=True))
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 11 — Kimi-FS160 replacement fallback\n\n"
        "This notebook runs the Path10 pipeline but finalizes the Path6 replacement "
        "candidate as `submission.csv`, instead of the Path6-on-Path5 stack. It is "
        "a controlled fallback for the last slot if Path10's stack blend dilutes Path5.\n"
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
