"""Path24 — decisive per-label override on the 0.920 parent.

Base: the byte-pinned Path13 Aman super-0920 parent submission (public LB
0.920), taken unchanged from the Path21 kernel output
(``submission_path13_0920_preserved.csv``, sha256-pinned).

Override: full replacement (alpha = 1.0) on exactly seven targets where two
independently trained members (dino2b_s2: seed 2, silver-v5 labels; and
dino2b_mm6: silver-v6 labels + InfoNCE report-text alignment; both full-gold
lambda=2) unanimously invert the parent's ordering of the three test studies
in the same direction, and where our gold evidence is strongest:
  Medial Meniscus (gold AUC 0.89-0.92), Medial OA (0.97), Lateral OA (0.83),
  Effusion (~1.0), Synovitis (0.83), Baker's (~1.0), Fracture (0.96).
On a 3-study test, any alpha < 1.0 only creates ties and cannot change an
ordering — hence full replacement. Targets where the members agree with the
parent (ACL, MCL, Lateral Meniscus, PF OA, Contusion) keep the parent.

CPU-only, offline, deterministic. Hard gates:
  * exactly one parent CSV with the pinned sha256,
  * exactly one member submission.csv located under the combined-ensemble
    kernel output slug,
  * identical study sets and columns in both inputs.
Any gate failure aborts without writing submission.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

OVERRIDE_TARGETS = [
    "Medial Meniscus", "Medial OA", "Lateral OA", "Effusion",
    "Synovitis", "Baker's", "Fracture",
]

PARENT_FILENAME = "submission_path13_0920_preserved.csv"
PARENT_SHA256 = "60d612928b1cc9bd78b80bed54e0f1d3a4f9de499325b892daa6e5ddf39f460f"
PARENT_SLUG = "path21-strong-alpha-apply"
MEMBER_SLUG = "combined-ensemble-inference-v2"
MEMBER_FILENAME = "submission.csv"
EXPECTED_ROWS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_one(filename: str, slug: str, label: str) -> Path:
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if filename in files and slug in root:
            hits.append(Path(root) / filename)
    if len(hits) != 1:
        raise FileNotFoundError(f"{label}: expected exactly one {filename} under *{slug}*, found {len(hits)}: {hits}")
    return hits[0]


def main() -> None:
    parent_path = find_one(PARENT_FILENAME, PARENT_SLUG, "parent")
    member_path = find_one(MEMBER_FILENAME, MEMBER_SLUG, "members")

    digest = sha256(parent_path)
    if digest != PARENT_SHA256:
        raise ValueError(f"parent sha256 {digest} != pinned {PARENT_SHA256}; refusing to blend")

    parent = pd.read_csv(parent_path)
    member = pd.read_csv(member_path)

    if parent.columns.tolist() != ["StudyInstanceUID"] + TARGETS:
        raise ValueError(f"parent columns mismatch: {parent.columns.tolist()}")
    if member.columns.tolist() != ["StudyInstanceUID"] + TARGETS:
        raise ValueError(f"member columns mismatch: {member.columns.tolist()}")
    if len(parent) != EXPECTED_ROWS or len(member) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, parent={len(parent)}, member={len(member)}")
    if sorted(parent["StudyInstanceUID"]) != sorted(member["StudyInstanceUID"]):
        raise ValueError("study sets differ between parent and member submissions")

    member = member.set_index("StudyInstanceUID").loc[parent["StudyInstanceUID"]].reset_index()

    final = parent.copy()
    changed_cells = {}
    for target in OVERRIDE_TARGETS:
        before = final[target].tolist()
        final[target] = member[target]
        if before != member[target].tolist():
            changed_cells[target] = {"parent": before, "member": member[target].tolist()}

    if not changed_cells:
        raise RuntimeError("override produced no cell changes; inputs are inconsistent with the decision evidence")

    final.to_csv("submission.csv", index=False)

    audit = {
        "mode": "path24-decisive-override",
        "base": "Path13 Aman super-0920 parent (public LB 0.920), sha256-pinned",
        "parent_sha256": digest,
        "member_source": "combined 3-member rank-mean (dino2b + dino2b_s2 + dino2b_mm6)",
        "override_targets": OVERRIDE_TARGETS,
        "kept_parent_targets": [t for t in TARGETS if t not in OVERRIDE_TARGETS],
        "changed_cells": changed_cells,
        "gold_evidence": {
            "Medial Meniscus": "s2 0.918 / mm6 0.889",
            "Medial OA": "s2 0.977 / mm6 0.974",
            "Lateral OA": "s2 0.826 / mm6 0.841",
            "Effusion": "s2 1.000 / mm6 0.998",
            "Synovitis": "s2 0.848 / mm6 0.832",
            "Baker's": "s2 0.998 / mm6 1.000",
            "Fracture": "s2 0.967 / mm6 0.957",
        },
        "rows": int(len(final)),
        "status": "PATH24_SUBMISSION_WRITTEN",
    }
    Path("path24_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()