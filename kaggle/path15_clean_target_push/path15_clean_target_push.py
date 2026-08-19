from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PATH10_SHA256 = "2822ba73c357ca4ae3f41d1aacf9c63f227565582c7ab36d90229753295c1aff"
PATH11_SHA256 = "ef1a0adf77efe18fba8132711cec141b657bffebeab58c4cafc6b6ffaa10db1f"
PUSH_WEIGHTS = {
    "ACL": 0.30,
    "MCL": 0.40,
    "Medial Meniscus": 0.45,
    "Lateral Meniscus": 0.45,
    "Synovitis": 0.35,
    "Contusion": 0.45,
    "Fracture": 0.30,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def competition_root() -> Path:
    for path in (
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    ):
        if (path / "test.csv").is_file():
            return path
    raise FileNotFoundError("competition test.csv not found")


def find_submission(expected_hash: str, audit_name: str, expected_status: str) -> Path:
    for audit_path in Path("/kaggle/input").glob(f"**/{audit_name}"):
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != expected_status:
            continue
        sub_path = audit_path.with_name("submission.csv")
        if sub_path.is_file() and sha256(sub_path) == expected_hash:
            return sub_path
    raise FileNotFoundError(f"submission hash {expected_hash} with {audit_name} not found")


def validate(frame: pd.DataFrame, test: pd.DataFrame, label: str) -> pd.DataFrame:
    expected = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != expected:
        raise ValueError(f"{label}: schema mismatch")
    if frame["StudyInstanceUID"].tolist() != test["StudyInstanceUID"].astype(str).tolist():
        raise ValueError(f"{label}: StudyInstanceUID order mismatch")
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"{label}: invalid prediction values")
    return frame


def main() -> None:
    comp = competition_root()
    test = pd.read_csv(comp / "test.csv", dtype={"StudyInstanceUID": str})
    path10 = find_submission(
        PATH10_SHA256,
        "path10_kimi_fs160_stack_audit.json",
        "PATH10_KIMI_FS160_STACK_APPLIED",
    )
    path11 = find_submission(
        PATH11_SHA256,
        "path11_kimi_fs160_replace_audit.json",
        "PATH11_KIMI_FS160_REPLACE_APPLIED",
    )

    base = validate(pd.read_csv(path10, dtype={"StudyInstanceUID": str}), test, "Path10")
    replace = validate(pd.read_csv(path11, dtype={"StudyInstanceUID": str}), test, "Path11")
    out = test[["StudyInstanceUID"]].copy()
    for target in TARGETS:
        weight = PUSH_WEIGHTS.get(target, 0.0)
        base_rank = base[target].rank(method="average", pct=True)
        replace_rank = replace[target].rank(method="average", pct=True)
        out[target] = (1.0 - weight) * base_rank + weight * replace_rank
    validate(out, test, "Path15")

    output = Path("/kaggle/working/submission.csv")
    out.to_csv(output, index=False)
    audit = {
        "status": "PATH15_CLEAN_TARGET_PUSH_READY",
        "strategy": (
            "target-wise rank blend: Path10 0.915 base, Path11 aggressive only "
            "where clean OOF supported Path6 diversity"
        ),
        "gold_policy": "no official gold rows used for this blend or selection",
        "path10_source": str(path10),
        "path11_source": str(path11),
        "path10_sha256": sha256(path10),
        "path11_sha256": sha256(path11),
        "submission_sha256": sha256(output),
        "target_push_weights": PUSH_WEIGHTS,
        "rows": int(len(out)),
        "targets": TARGETS,
    }
    Path("/kaggle/working/path15_clean_target_push_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
