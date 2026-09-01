"""Label engine v6 — fusion driver.

Waits for the new extractor corpus CSVs (ox + qwen, produced by
13_label_engine_v6_corpus.py) to reach full coverage, then runs the proven
08_label_fusion cross-fit protocol over public teachers + the new extractors
to emit silver_labels_v6.csv + audit. Falls back to public-teacher-only fusion
if the new extractors are not ready yet (--allow-public-only).

Usage:
    .venv/bin/python notebooks/14_label_fusion_v6.py [--allow-public-only]

The incumbent (first --source) is the strongest public teacher; per-label
winners are chosen on gold AUC under the GOLD_INTEGRATION_PLAN §2 cross-fit
protocol with the 0.02 noise floor.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_ENGINE = PROJECT_ROOT / ".codex_work" / "label_engine_v6"
FUSION_SCRIPT = PROJECT_ROOT / "notebooks" / "08_label_fusion.py"
GOLD_LABELS = PROJECT_ROOT / "input" / "gold_labels.csv"
GOLD_FOLDS = PROJECT_ROOT / "input" / "gold_folds.csv"
OUTPUT = PROJECT_ROOT / "input" / "silver_labels_v6.csv"
AUDIT = PROJECT_ROOT / "input" / "silver_labels_v6_fusion_audit.json"

PUBLIC_SOURCES = [
    "flight=.codex_work/public_datasets/flight_hybrid_labels/report_labels_v4hybrid.csv",
    "steven_v4=.codex_work/public_datasets/steven_labels/llm_labels_v4_blend.csv",
    "steven_v2=.codex_work/public_datasets/steven_labels/llm_labels_v2.csv",
    "pilkwang=.codex_work/public_datasets/pilkwang_labels/report_labels_v2.csv",
    "lixin=.codex_work/public_datasets/lixin_sol56_labels/report_labels_gpt56sol.csv",
]

NEW_SOURCES = [
    ("ox", LABEL_ENGINE / "corpus_ox-alpha-free.csv"),
    ("qwen", LABEL_ENGINE / "corpus_qwen_qwen3-30b-a3b-instruct-2507-4bit.csv"),
    ("gpt4o_mini", LABEL_ENGINE / "corpus_gpt-4o-mini.csv"),
]

TOTAL_SILVER = 4349


def ready(path: Path) -> bool:
    if not path.is_file():
        return False
    import pandas as pd
    try:
        df = pd.read_csv(path, usecols=["StudyInstanceUID"])
        return len(df) >= TOTAL_SILVER - 5
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-public-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sources = list(PUBLIC_SOURCES)
    included = []
    for name, path in NEW_SOURCES:
        if ready(path):
            sources.append(f"{name}={path}")
            included.append(name)
        else:
            print(f"[skip] {name}: corpus not ready at {path}")
    if not included and not args.allow_public_only:
        print("No new extractor corpus ready; pass --allow-public-only to fuse public teachers only.")
        sys.exit(0)

    cmd = [
        sys.executable, str(FUSION_SCRIPT),
        "--gold-labels", str(GOLD_LABELS),
        "--gold-folds", str(GOLD_FOLDS),
        "--output", str(OUTPUT),
        "--audit", str(AUDIT),
    ]
    for s in sources:
        cmd += ["--source", s]
    if args.force:
        cmd += ["--force"]

    print("Running fusion with sources:")
    for s in sources:
        print("  ", s)
    print("New extractors included:", included or "none")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
