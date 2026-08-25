"""Fuse multiple silver-label extraction sources into input/silver_labels_v4.csv.

Per-label source selection is done under the gold cross-fit protocol of
GOLD_INTEGRATION_PLAN.md §2:

- SELECTION EVIDENCE (gold_usage: crossfit): for each gold fold k, the best
  source per label is chosen by gold AUC computed on folds != k only, with the
  noise-floor rule that the incumbent (first --source) keeps the label unless a
  challenger beats it by > --margin (default 0.02, the §2 noise floor). The
  cross-fitted policy AUC is the honest estimate of the fusion policy.
- DEPLOYMENT ARTIFACT (gold_usage: full): the same selection repeated on all
  58 gold rows gives the per-label winning-source map used to build
  silver_labels_v4.csv. The recipe is frozen by the cross-fitted evidence
  before this full-gold map is computed.

Missing cells are 0.5 ("not addressed", masked in loss downstream as
labels != 0.5). A real value is NEVER written over a cell that is 0.5 in every
source: the fallback chain only replaces 0.5 with a non-0.5 cell.

Sources are wide CSVs (StudyInstanceUID + 12 label columns) or long-form
detailed CSVs (StudyInstanceUID|uid, finding, value[, score|soft_score],
evidence); long-form files are pivoted with the assembler's value mapping.
--override name=path sources are applied cell-level after fusion: any non-0.5
override cell replaces the fused cell (for targeted-reextract patches).

Deterministic: no RNG anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labels.assembler import SILENCE_INFORMATIVE
from src.labels.extractor import LABELS

MISSING = 0.5  # "not addressed" sentinel — masked in loss, never a measurement
MIN_CLASS_COUNT = 5  # skip AUC for a label with <5 positives or <5 negatives

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def atomic_write_json(payload: dict, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=float))
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected name=path, got {spec!r}")
    name, _, raw = spec.partition("=")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"empty source name in {spec!r}")
    return name, Path(raw)


# --------------------------------------------------------------------------- #
# Source loading (wide or long-form detailed)
# --------------------------------------------------------------------------- #

VALUE_MAP = {"present": 1.0, "absent": 0.0, "uncertain": MISSING,
             "laterality_ambiguous": MISSING}


def _value_to_score(value: str, finding: str) -> float:
    """Assembler mapping: not_addressed uses per-finding silence semantics."""
    if value == "not_addressed":
        return SILENCE_INFORMATIVE.get(finding, MISSING)
    return VALUE_MAP.get(value, np.nan)


def _pivot_detailed(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Long-form detailed CSV -> wide soft-label matrix."""
    df = raw.rename(columns={"uid": "StudyInstanceUID"})
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    bad = set(df["finding"]) - set(LABELS)
    if bad:
        LOGGER.warning("source %s: dropping unknown findings %s", name, sorted(bad))
        df = df[df["finding"].isin(LABELS)]
    score_col = next((c for c in ("soft_score", "score")
                      if c in df.columns and df[c].notna().any()), None)
    if score_col is not None:
        df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        df["_score"] = np.nan
    fallback = df["_score"].isna()
    if fallback.any():
        df.loc[fallback, "_score"] = [
            _value_to_score(v, f)
            for v, f in zip(df.loc[fallback, "value"], df.loc[fallback, "finding"])
        ]
    n_dupes = int(df.duplicated(subset=["StudyInstanceUID", "finding"]).sum())
    if n_dupes:
        LOGGER.warning("source %s: %d duplicate (uid, finding) rows, keeping first",
                       name, n_dupes)
        df = df.drop_duplicates(subset=["StudyInstanceUID", "finding"], keep="first")
    wide = df.pivot(index="StudyInstanceUID", columns="finding", values="_score")
    return wide.reset_index()


def load_source(name: str, path: Path) -> pd.DataFrame:
    """Load one source as a wide DataFrame: StudyInstanceUID + LABELS columns."""
    if not path.is_file():
        raise FileNotFoundError(f"source {name}: {path} does not exist")
    header = pd.read_csv(path, nrows=0)
    if "finding" in header.columns:
        df = _pivot_detailed(name, pd.read_csv(path, dtype={"StudyInstanceUID": str,
                                                            "uid": str}))
    else:
        df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in df.columns:
        raise ValueError(f"source {name}: no StudyInstanceUID column in {path}")
    missing_cols = [label for label in LABELS if label not in df.columns]
    if missing_cols:
        LOGGER.warning("source %s: missing label columns %s (treated as 0.5)",
                       name, missing_cols)
    df = df[["StudyInstanceUID"] + [c for c in LABELS if c in df.columns]].copy()
    for label in LABELS:
        if label not in df.columns:
            df[label] = np.nan
        df[label] = pd.to_numeric(df[label], errors="coerce")
    n_dupes = int(df.duplicated(subset=["StudyInstanceUID"]).sum())
    if n_dupes:
        LOGGER.warning("source %s: %d duplicate UIDs, keeping first", name, n_dupes)
        df = df.drop_duplicates(subset=["StudyInstanceUID"], keep="first")
    values = df[LABELS].to_numpy(dtype=float)
    out_of_range = np.isfinite(values) & ((values < 0.0) | (values > 1.0))
    if out_of_range.any():
        LOGGER.warning("source %s: %d cells outside [0,1], clipping",
                       name, int(out_of_range.sum()))
        df[LABELS] = np.clip(values, 0.0, 1.0)
    # Cells a source never produced (absent row/label) are 0.5 = not addressed.
    df[LABELS] = df[LABELS].fillna(MISSING)
    return df[["StudyInstanceUID"] + LABELS]


# --------------------------------------------------------------------------- #
# Gold evaluation harness
# --------------------------------------------------------------------------- #

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC, or NaN when the eval set has <5 positives or <5 negatives."""
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if (y_true == 1).sum() < MIN_CLASS_COUNT or (y_true == 0).sum() < MIN_CLASS_COUNT:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def source_gold_aucs(matrix: pd.DataFrame, gold: pd.DataFrame,
                     rows: pd.Index) -> dict[str, float]:
    """Per-label AUC of one source vs gold binary on the given gold rows.

    A row only enters a label's eval set if the source *addresses* that cell
    (value != 0.5); a bare 0.5 carries no ranking signal.
    """
    gold_idx = gold.set_index("StudyInstanceUID")
    mat_idx = matrix.set_index("StudyInstanceUID")
    common = [uid for uid in rows if uid in mat_idx.index and uid in gold_idx.index]
    aucs = {}
    for label in LABELS:
        y_true = gold_idx.loc[common, label].to_numpy(dtype=float)
        y_score = mat_idx.loc[common, label].to_numpy(dtype=float)
        addressed = np.isfinite(y_score) & (y_score != MISSING)
        aucs[label] = safe_auc(y_true[addressed], y_score[addressed])
    return aucs


def choose_winner(aucs_by_source: dict[str, dict[str, float]], label: str,
                  sources: list[str], margin: float) -> str:
    """Incumbent (first source) keeps the label unless a challenger beats its
    AUC by more than `margin`. NaN (unevaluable) always loses."""
    incumbent = sources[0]
    incumbent_auc = aucs_by_source[incumbent][label]
    best_challenger, best_auc = None, float("nan")
    for challenger in sources[1:]:
        auc = aucs_by_source[challenger][label]
        if np.isnan(auc):
            continue
        if best_challenger is None or auc > best_auc:
            best_challenger, best_auc = challenger, auc
    if best_challenger is None:
        return incumbent
    if np.isnan(incumbent_auc) or best_auc > incumbent_auc + margin:
        return best_challenger
    return incumbent


def rank_sources(aucs_by_source: dict[str, dict[str, float]], label: str,
                 sources: list[str]) -> list[str]:
    """Fallback order: AUC descending, NaN last, CLI order breaks ties."""
    return sorted(sources,
                  key=lambda s: (np.isnan(aucs_by_source[s][label]),
                                 -np.nan_to_num(aucs_by_source[s][label], nan=-1.0),
                                 sources.index(s)))


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

def ordered_union_uid(dfs: list[pd.DataFrame]) -> list[str]:
    seen, uids = set(), []
    for df in dfs:
        for uid in df["StudyInstanceUID"]:
            if uid not in seen:
                seen.add(uid)
                uids.append(uid)
    return uids


def fuse_frames(mats: dict[str, pd.DataFrame], uids: list[str],
                rankings: dict[str, list[str]]) -> pd.DataFrame:
    """Per label, take the top-ranked source's cell; fall back to the next
    ranked source with a non-0.5 cell; keep 0.5 when every source is 0.5."""
    aligned = {name: m.set_index("StudyInstanceUID")[LABELS]
               for name, m in mats.items()}
    uid_arr = np.asarray(uids)
    fused = pd.DataFrame(MISSING, index=pd.Index(uids, name="StudyInstanceUID"),
                         columns=LABELS, dtype=float)
    for label in LABELS:
        cols = np.column_stack([
            aligned[s][label].reindex(uids).to_numpy(dtype=float)
            for s in rankings[label]])
        addressed = np.isfinite(cols) & (cols != MISSING)
        has = addressed.any(axis=1)
        if has.any():
            sel = cols[has]
            first = addressed[has].argmax(axis=1)
            fused.loc[uid_arr[has], label] = sel[np.arange(len(sel)), first]
    return fused.reset_index()


def apply_overrides(fused: pd.DataFrame, overrides: dict[str, pd.DataFrame],
                    audit_counts: dict) -> pd.DataFrame:
    """Cell-level override: any non-0.5 override cell replaces the fused cell."""
    fused = fused.set_index("StudyInstanceUID")
    for name, df in overrides.items():
        cells = 0
        for uid, row in df.set_index("StudyInstanceUID").iterrows():
            if uid not in fused.index:
                fused.loc[uid, LABELS] = MISSING
            for label in LABELS:
                value = row[label]
                if pd.notna(value) and float(value) != MISSING:
                    fused.loc[uid, label] = float(value)
                    cells += 1
        audit_counts[name] = cells
        LOGGER.info("override %s: patched %d cells", name, cells)
    return fused.reset_index()


def synovitis_impute(fused: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Targeted imputation: Synovitis = Effusion * 0.5 ONLY on 0.5 cells.

    Measured to lift Synovitis gold AUC 0.678 -> 0.790; blanket imputation
    (overwriting addressed cells) is worse, so addressed cells are untouched.
    """
    mask = fused["Synovitis"] == MISSING
    out = fused.copy()
    out.loc[mask, "Synovitis"] = fused.loc[mask, "Effusion"] * 0.5
    return out, int(mask.sum())


# --------------------------------------------------------------------------- #
# Cross-fit harness
# --------------------------------------------------------------------------- #

def crossfit_policy_scores(mats: dict[str, pd.DataFrame], gold: pd.DataFrame,
                           folds: pd.Series, sources: list[str],
                           margin: float) -> tuple[pd.DataFrame, dict]:
    """For each gold fold k: select winners/rankings on folds != k, then fuse
    fold k's gold rows through the same fallback chain. Returns per-(uid,
    label) scores (NaN where no ranked source addresses the cell) and the
    per-fold winner maps."""
    gold_uids = gold["StudyInstanceUID"]
    aligned = {s: mats[s].set_index("StudyInstanceUID")[LABELS] for s in sources}
    per_fold_winners: dict[int, dict[str, str]] = {}
    scores = pd.DataFrame(np.nan, index=gold_uids, columns=LABELS, dtype=float)
    for fold in sorted(folds.unique()):
        train_rows = folds[folds != fold].index
        aucs = {s: source_gold_aucs(mats[s], gold, train_rows) for s in sources}
        winners = {label: choose_winner(aucs, label, sources, margin)
                   for label in LABELS}
        rankings = {label: rank_sources(aucs, label, sources) for label in LABELS}
        per_fold_winners[int(fold)] = winners
        for uid in folds[folds == fold].index:
            for label in LABELS:
                for s in rankings[label]:
                    value = aligned[s][label].get(uid, np.nan)
                    if pd.notna(value) and value != MISSING:
                        scores.loc[uid, label] = value
                        break
    return scores, per_fold_winners


def policy_aucs(scores: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float]:
    """Per-label + macro AUC of fused scores vs gold, aligned on UID. Macro is
    the mean over labels with an evaluable AUC; n_labels records how many."""
    gold_idx = gold.set_index("StudyInstanceUID")
    aligned = scores.reindex(gold_idx.index)
    aucs = {label: safe_auc(gold_idx[label].to_numpy(dtype=float),
                            aligned[label].to_numpy(dtype=float))
            for label in LABELS}
    valid = [a for a in aucs.values() if not np.isnan(a)]
    aucs["macro"] = float(np.mean(valid)) if valid else float("nan")
    aucs["n_labels"] = float(len(valid))
    return aucs


def paired_macros(aucs_a: dict[str, float], aucs_b: dict[str, float]) -> dict:
    """Macros over labels evaluable in BOTH variants, so the imputation effect
    is compared on the same label set."""
    both = [label for label in LABELS
            if not np.isnan(aucs_a[label]) and not np.isnan(aucs_b[label])]
    return {
        "labels": both,
        "without_imputation": float(np.mean([aucs_a[l] for l in both])) if both else None,
        "with_imputation": float(np.mean([aucs_b[l] for l in both])) if both else None,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", type=parse_named_path,
                        required=True, metavar="name=path",
                        help="candidate silver source (wide or detailed CSV); "
                             "repeatable. The FIRST source is the incumbent that "
                             "wins ties within --margin.")
    parser.add_argument("--override", action="append", type=parse_named_path,
                        default=[], metavar="name=path",
                        help="cell-level override source (e.g. targeted reextract "
                             "detailed CSV); non-0.5 cells replace fused cells")
    parser.add_argument("--gold-labels", type=Path,
                        default=PROJECT_ROOT / "input" / "gold_labels.csv")
    parser.add_argument("--gold-folds", type=Path,
                        default=PROJECT_ROOT / "input" / "gold_folds.csv")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "input" / "silver_labels_v4.csv")
    parser.add_argument("--audit", type=Path,
                        default=PROJECT_ROOT / "input" / "silver_labels_v4_fusion_audit.json")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="AUC margin a challenger must beat the incumbent by "
                             "(the GOLD_INTEGRATION_PLAN §2 noise floor)")
    parser.add_argument("--synovitis-impute", dest="synovitis_impute",
                        action="store_true", default=True)
    parser.add_argument("--no-synovitis-impute", dest="synovitis_impute",
                        action="store_false")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output/audit file; existing "
                             "fusion artifacts are never silently replaced")
    args = parser.parse_args()

    if args.margin < 0:
        parser.error("--margin must be non-negative")
    for path in (args.output, args.audit):
        if path.exists() and not args.force:
            parser.error(f"{path} already exists — pass --force to overwrite")

    sources = [name for name, _ in args.source]
    if len(sources) != len(set(sources)):
        parser.error("duplicate source names")
    override_names = [name for name, _ in args.override]
    if set(override_names) & set(sources):
        parser.error("override names must not collide with source names")
    if len(override_names) != len(set(override_names)):
        parser.error("duplicate override names")

    # 1. Load sources, align on StudyInstanceUID, report coverage.
    mats: dict[str, pd.DataFrame] = {}
    for name, path in args.source:
        mats[name] = load_source(name, path)
        LOGGER.info("source %s: %d studies from %s", name, len(mats[name]), path)
    overrides: dict[str, pd.DataFrame] = {}
    for name, path in args.override:
        overrides[name] = load_source(name, path)
        LOGGER.info("override %s: %d studies from %s", name, len(overrides[name]), path)

    gold = pd.read_csv(args.gold_labels, dtype={"StudyInstanceUID": str})
    gold[LABELS] = gold[LABELS].astype(float)
    fold_df = pd.read_csv(args.gold_folds, dtype={"StudyInstanceUID": str})
    fold_col = next(c for c in fold_df.columns if c != "StudyInstanceUID")
    folds = fold_df.set_index("StudyInstanceUID")[fold_col].astype(int)
    gold = gold[gold["StudyInstanceUID"].isin(folds.index)].reset_index(drop=True)
    LOGGER.info("gold: %d studies across %d folds", len(gold), folds.nunique())
    gold_uids = set(gold["StudyInstanceUID"])

    coverage = {}
    for name, df in ([*mats.items()]
                     + [(f"override:{n}", d) for n, d in overrides.items()]):
        coverage[name] = {
            label: round(float((df[label] != MISSING).mean()), 4) for label in LABELS}
        nonzero = {k: v for k, v in coverage[name].items() if v > 0}
        LOGGER.info("coverage %s: %s", name, nonzero or "all missing")

    # 2. Single-source gold AUCs on all 58 gold rows (deployment evidence).
    all_gold_rows = pd.Index(gold["StudyInstanceUID"])
    single_aucs = {s: source_gold_aucs(mats[s], gold, all_gold_rows) for s in sources}
    for s in sources:
        valid = [a for a in single_aucs[s].values() if not np.isnan(a)]
        single_aucs[s]["macro"] = float(np.mean(valid)) if valid else float("nan")
        LOGGER.info("single-source gold AUC %s: macro=%s", s,
                    round(single_aucs[s]["macro"], 4)
                    if not np.isnan(single_aucs[s]["macro"]) else "NaN")

    # 3. Cross-fitted selection: the honest estimate of the fusion policy.
    cf_scores, per_fold_winners = crossfit_policy_scores(
        mats, gold, folds, sources, args.margin)
    cf_aucs = policy_aucs(cf_scores, gold)
    cf_imputed, n_cf_imputed = synovitis_impute(
        cf_scores.fillna(MISSING).reset_index())
    cf_aucs_imputed = policy_aucs(
        cf_imputed.set_index("StudyInstanceUID"), gold)
    cf_paired = paired_macros(cf_aucs, cf_aucs_imputed)
    LOGGER.info("cross-fitted policy macro AUC=%.4f (synovitis-imputed=%.4f)",
                cf_aucs["macro"], cf_aucs_imputed["macro"])
    LOGGER.info("paired macro on shared labels (n=%d): without=%s with=%s",
                len(cf_paired["labels"]),
                cf_paired["without_imputation"], cf_paired["with_imputation"])
    for fold, winners in per_fold_winners.items():
        LOGGER.info("fold %d winners: %s", fold, winners)

    # 4. Deployment artifact: selection on all 58 gold rows, then fuse.
    deploy_map = {label: choose_winner(single_aucs, label, sources, args.margin)
                  for label in LABELS}
    deploy_rankings = {label: rank_sources(single_aucs, label, sources)
                       for label in LABELS}
    LOGGER.info("deployment source map: %s", deploy_map)

    uids = ordered_union_uid(list(mats.values()) + list(overrides.values()))
    fused = fuse_frames(mats, uids, deploy_rankings)
    override_counts: dict[str, int] = {}
    if overrides:
        fused = apply_overrides(fused, overrides, override_counts)
    fused[LABELS] = fused[LABELS].round(4)

    # 5. Synovitis imputation toggle (deployment).
    n_imputed = 0
    if args.synovitis_impute:
        fused, n_imputed = synovitis_impute(fused)
        LOGGER.info("synovitis imputation: %d cells set to Effusion*0.5", n_imputed)

    # Sanity guards before writing.
    values = fused[LABELS].to_numpy(dtype=float)
    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise RuntimeError("fused labels outside [0,1] — refusing to write")
    atomic_write_csv(fused, args.output)
    LOGGER.info("wrote %s (%d studies)", args.output, len(fused))

    # 6. Audit JSON.
    def clean(d: dict) -> dict:
        return {k: (None if np.isnan(v) else round(v, 4)) for k, v in d.items()}

    audit = {
        "script": Path(__file__).name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gold_usage": {
            "selection_evidence": "crossfit",
            "deployment_artifact": "full",
            "note": ("Per-label winners/rankings were validated cross-fitted "
                     "(chosen on folds != k, applied to fold k); the deployment "
                     "map written to silver_labels_v4.csv is the same rule "
                     "re-fit on all 58 gold rows after the recipe was frozen."),
        },
        "margin": args.margin,
        "min_class_count_for_auc": MIN_CLASS_COUNT,
        "synovitis_impute": bool(args.synovitis_impute),
        "sources": [
            {"name": name, "path": str(path), "sha256": sha256_file(path),
             "n_studies": int(len(mats[name])),
             "gold_overlap": int(mats[name]["StudyInstanceUID"].isin(gold_uids).sum())}
            for name, path in args.source],
        "overrides": [
            {"name": name, "path": str(path), "sha256": sha256_file(path),
             "n_studies": int(len(overrides[name])),
             "cells_overridden": override_counts.get(name, 0)}
            for name, path in args.override],
        "coverage_fraction_nonmissing": coverage,
        "single_source_gold_auc": {s: clean(single_aucs[s]) for s in sources},
        "crossfit": {
            "per_fold_winners": {str(k): v for k, v in per_fold_winners.items()},
            "policy_auc": clean(cf_aucs),
            "policy_auc_synovitis_imputed": clean(cf_aucs_imputed),
            "paired_macro_shared_labels": cf_paired,
            "synovitis_cells_imputed_in_harness": n_cf_imputed,
        },
        "deployment_source_map": {
            label: {"winner": deploy_map[label],
                    "ranking": deploy_rankings[label],
                    "source_aucs": clean({s: single_aucs[s][label] for s in sources})}
            for label in LABELS},
        "synovitis_imputation_deployment": {
            "applied": bool(args.synovitis_impute),
            "cells_imputed": n_imputed,
            "rule": "Synovitis = Effusion * 0.5 only where fused Synovitis == 0.5",
        },
        "inputs": {
            "gold_labels": {"path": str(args.gold_labels),
                            "sha256": sha256_file(args.gold_labels)},
            "gold_folds": {"path": str(args.gold_folds),
                           "sha256": sha256_file(args.gold_folds)},
        },
        "output": {"path": str(args.output), "sha256": sha256_file(args.output),
                   "n_rows": int(len(fused))},
    }
    atomic_write_json(audit, args.audit)
    LOGGER.info("wrote %s", args.audit)
    LOGGER.info("Done; fused %d sources + %d overrides -> %d studies",
                len(mats), len(overrides), len(fused))


if __name__ == "__main__":
    main()
