"""Build input/silver_labels_mv1.csv — a gold-free hard majority vote across
7 label sources, deliberately decorrelated from the flight-dominant
silver_labels_v5 fusion (which was per-label source-selected on gold AUCs).

Sources (all soft-score CSVs hardened at 0.5 into 0/1 votes):
  flight, steven_v2, steven_v4, pilkwang, lixin  (5 public teachers)
  inhouse_v4  (input/silver_labels_v4.csv, fused in-house line, 158 studies)
  lf          (input/rulebased_labels_v1.csv; abstains outside
               Effusion/Synovitis/Baker's/Lateral OA)

Vote rule per (study, label):
  vote = 1 if source value > 0.5, 0 if < 0.5, abstain if == 0.5 (missing /
  "not addressed" sentinel, including cells a source never produced).
  mean of non-abstaining votes > 0.5 -> 1, < 0.5 -> 0; a tie at exactly
  0.5 -> 0.5 (kept missing — no coin flips). All-abstain -> 0.5.

GOLD IS NOT USED TO BUILD THIS ARTIFACT (gold_usage: none). Gold is read
only AFTER the label set is frozen to report per-label AUCs of mv1 vs v5 in
the audit (recorded as gold_usage: evaluation_only).

Conventions match notebooks/08_label_fusion.py (atomic writes, sha256 audit,
0.5=missing preserved, --force no-overwrite guard); loader/eval helpers are
imported from it so the missing-cell semantics stay identical.

Deterministic: no RNG anywhere.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

fusion = importlib.import_module("08_label_fusion")  # noqa: E402
from src.labels.extractor import LABELS  # noqa: E402

MISSING = fusion.MISSING  # 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Fixed, gold-free source list. Order is irrelevant to a plain majority vote
# but is recorded for provenance.
SOURCES: list[tuple[str, Path]] = [
    ("flight", PROJECT_ROOT / ".codex_work/public_datasets/flight_hybrid_labels/report_labels_v4hybrid.csv"),
    ("steven_v2", PROJECT_ROOT / ".codex_work/public_datasets/steven_labels/llm_labels_v2.csv"),
    ("steven_v4", PROJECT_ROOT / ".codex_work/public_datasets/steven_labels/llm_labels_v4_blend.csv"),
    ("pilkwang", PROJECT_ROOT / ".codex_work/public_datasets/pilkwang_labels/report_labels_v2.csv"),
    ("lixin", PROJECT_ROOT / ".codex_work/public_datasets/lixin_sol56_labels/labels_llm_gpt56sol.csv"),
    ("inhouse_v4", PROJECT_ROOT / "input/silver_labels_v4.csv"),
    ("lf", PROJECT_ROOT / "input/rulebased_labels_v1.csv"),
]


def harden(values: np.ndarray) -> np.ndarray:
    """Soft score -> hard vote: 1.0 / 0.0, NaN for abstain (== 0.5 or NaN)."""
    out = np.full(values.shape, np.nan)
    out[values > MISSING] = 1.0
    out[values < MISSING] = 0.0
    return out


def majority_vote(mats: dict[str, pd.DataFrame], uids: list[str]) -> tuple[pd.DataFrame, dict]:
    """Hard majority vote per (uid, label). Returns the wide mv1 frame and a
    stats dict (per-label vote-count histogram + tie/all-abstain counts)."""
    aligned = {name: m.set_index("StudyInstanceUID")[LABELS].reindex(uids)
               for name, m in mats.items()}
    fused = pd.DataFrame(MISSING, index=pd.Index(uids, name="StudyInstanceUID"),
                         columns=LABELS, dtype=float)
    stats: dict[str, dict] = {}
    for label in LABELS:
        votes = np.column_stack([
            harden(aligned[s][label].to_numpy(dtype=float)) for s in mats])
        n_votes = np.isfinite(votes).sum(axis=1)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(votes, axis=1)
        col = np.full(len(uids), MISSING)
        has = n_votes > 0
        col[has & (mean > MISSING)] = 1.0
        col[has & (mean < MISSING)] = 0.0
        # mean == 0.5 exactly (tie) and all-abstain stay 0.5.
        fused[label] = col
        stats[label] = {
            "n_cells_addressed": int(has.sum()),
            "n_ties_0p5_mean": int((has & (mean == MISSING)).sum()),
            "n_all_abstain": int((~has).sum()),
            "vote_count_hist": {str(k): int((n_votes == k).sum())
                                for k in range(len(mats) + 1)
                                if (n_votes == k).any()},
        }
    return fused.reset_index(), stats


def prevalence(df: pd.DataFrame, hardened: bool) -> dict[str, float]:
    """Positive prevalence per label among addressed cells (!= 0.5)."""
    out = {}
    for label in LABELS:
        col = df[label].to_numpy(dtype=float)
        if hardened:
            col = harden(col)
        addressed = np.isfinite(col)
        out[label] = (float(np.nanmean(col)) if addressed.any()
                      else float("nan"))
    return out


def disagreement(mv1: pd.DataFrame, v5: pd.DataFrame) -> dict[str, dict]:
    """Per-label disagreement between mv1 and v5, both hardened at 0.5, on
    cells where BOTH are addressed. Also counts cells addressed by only one."""
    a = mv1.set_index("StudyInstanceUID")[LABELS]
    b = v5.set_index("StudyInstanceUID")[LABELS]
    out = {}
    for label in LABELS:
        va = harden(a[label].to_numpy(dtype=float))
        vb = harden(b[label].to_numpy(dtype=float))
        both = np.isfinite(va) & np.isfinite(vb)
        out[label] = {
            "n_both_addressed": int(both.sum()),
            "n_disagree": int((both & (va != vb)).sum()),
            "rate": (float((both & (va != vb)).mean()) if both.any()
                     else float("nan")),
            "n_mv1_only": int((np.isfinite(va) & ~np.isfinite(vb)).sum()),
            "n_v5_only": int((~np.isfinite(va) & np.isfinite(vb)).sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-labels", type=Path,
                        default=PROJECT_ROOT / "input" / "gold_labels.csv",
                        help="read-only; used ONLY for post-build evaluation")
    parser.add_argument("--v5", type=Path,
                        default=PROJECT_ROOT / "input" / "silver_labels_v5.csv",
                        help="incumbent fused labels, for comparison only")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "input" / "silver_labels_mv1.csv")
    parser.add_argument("--audit", type=Path,
                        default=PROJECT_ROOT / "input" / "silver_labels_mv1_audit.json")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output/audit file; existing "
                             "artifacts are never silently replaced")
    args = parser.parse_args()

    for path in (args.output, args.audit):
        if path.exists() and not args.force:
            parser.error(f"{path} already exists — pass --force to overwrite")

    # 1. Load the 7 sources with the exact 08_label_fusion semantics
    #    (absent rows/labels and NaN cells -> 0.5 abstain).
    mats: dict[str, pd.DataFrame] = {}
    for name, path in SOURCES:
        mats[name] = fusion.load_source(name, path)
        LOGGER.info("source %s: %d studies from %s", name, len(mats[name]), path)

    uids = fusion.ordered_union_uid(list(mats.values()))
    LOGGER.info("uid union: %d studies", len(uids))

    # 2. Hard majority vote — no gold involved anywhere above this line.
    mv1, vote_stats = majority_vote(mats, uids)

    # Sanity guard before writing.
    values = mv1[LABELS].to_numpy(dtype=float)
    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise RuntimeError("mv1 labels outside [0,1] — refusing to write")
    fusion.atomic_write_csv(mv1, args.output)
    LOGGER.info("wrote %s (%d studies)", args.output, len(mv1))

    # 3. Comparison vs v5 (comparison only — v5 is not an input to the vote).
    v5 = fusion.load_source("v5", args.v5)
    prev_mv1 = prevalence(mv1, hardened=False)
    prev_v5 = prevalence(v5, hardened=True)
    dis = disagreement(mv1, v5)
    coverage = {label: round(float((mv1[label] != MISSING).mean()), 4)
                for label in LABELS}
    for label in LABELS:
        LOGGER.info(
            "%s: coverage=%.3f prev mv1=%.3f v5=%.3f delta=%+.3f "
            "disagree=%s/%s (%.3f)",
            label, coverage[label], prev_mv1[label], prev_v5[label],
            prev_mv1[label] - prev_v5[label],
            dis[label]["n_disagree"], dis[label]["n_both_addressed"],
            dis[label]["rate"] if not np.isnan(dis[label]["rate"]) else float("nan"))

    # 4. Post-build gold evaluation (evaluation_only; labels already frozen).
    gold = pd.read_csv(args.gold_labels, dtype={"StudyInstanceUID": str})
    gold[LABELS] = gold[LABELS].astype(float)
    gold_rows = pd.Index(gold["StudyInstanceUID"])
    auc_mv1 = fusion.source_gold_aucs(mv1, gold, gold_rows)
    auc_v5 = fusion.source_gold_aucs(v5, gold, gold_rows)
    for name, aucs in (("mv1", auc_mv1), ("v5", auc_v5)):
        valid = [a for a in aucs.values() if not np.isnan(a)]
        aucs["macro"] = float(np.mean(valid)) if valid else float("nan")
    LOGGER.info("gold AUC macro: mv1=%.4f v5=%.4f", auc_mv1["macro"], auc_v5["macro"])

    # 5. Audit JSON.
    def clean(d: dict) -> dict:
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else round(v, 4))
                for k, v in d.items()}

    audit = {
        "script": Path(__file__).name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gold_usage": "evaluation_only",
        "gold_usage_note": ("mv1 is a plain hard majority vote of 7 fixed "
                            "sources; no source, weight, or threshold was "
                            "chosen using gold. Gold was read only after "
                            "silver_labels_mv1.csv was written, to report AUCs."),
        "vote_rule": ("per source: >0.5 -> 1, <0.5 -> 0, ==0.5 (or absent "
                      "row/label) -> abstain; per (study,label): mean of "
                      "non-abstaining votes >0.5 -> 1, <0.5 -> 0, tie at "
                      "exactly 0.5 -> 0.5 (missing preserved); all-abstain "
                      "-> 0.5"),
        "sources": [
            {"name": name, "path": str(path), "sha256": fusion.sha256_file(path),
             "n_studies": int(len(mats[name]))}
            for name, path in SOURCES],
        "vote_stats_per_label": vote_stats,
        "coverage_fraction_nonmissing": coverage,
        "prevalence": {
            "mv1": clean(prev_mv1),
            "v5_hardened": clean(prev_v5),
            "delta_mv1_minus_v5": clean(
                {l: prev_mv1[l] - prev_v5[l] for l in LABELS}),
        },
        "disagreement_vs_v5": {
            l: {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in d.items()} for l, d in dis.items()},
        "gold_auc": {"mv1": clean(auc_mv1), "v5": clean(auc_v5),
                     "min_class_count": fusion.MIN_CLASS_COUNT,
                     "note": ("mv1 scores are hard 0/1 (ties degrade AUC "
                              "resolution); v5 scores are soft. Addressed "
                              "cells only (0.5 excluded), per 08 convention.")},
        "comparison_input": {"v5": {"path": str(args.v5),
                                    "sha256": fusion.sha256_file(args.v5)}},
        "inputs": {"gold_labels": {"path": str(args.gold_labels),
                                   "sha256": fusion.sha256_file(args.gold_labels)}},
        "output": {"path": str(args.output),
                   "sha256": fusion.sha256_file(args.output),
                   "n_rows": int(len(mv1))},
    }
    fusion.atomic_write_json(audit, args.audit)
    LOGGER.info("wrote %s", args.audit)
    LOGGER.info("Done; majority-voted %d sources -> %d studies", len(mats), len(mv1))


if __name__ == "__main__":
    main()
