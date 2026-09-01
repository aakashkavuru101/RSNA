"""Consensus validator — rebuild the validation signal.

The silver-label OOF is noise-capped (~0.80 for every model) and cannot rank
candidates. This script fuses the OOF predictions of many strong members into
a consensus score, calibrates per-target member weights on the 58 gold
studies, validates the consensus itself on gold, and emits:

  .codex_work/consensus_validator/consensus_labels.csv
      StudyInstanceUID + 12 consensus soft-labels + per-study confidence
  .codex_work/consensus_validator/consensus_report.json
      per-member gold AUCs, consensus gold AUC, weights

The consensus labels are a *pseudo-gold* validation target: use them for
blend selection, member weighting, and (later) distillation — never for
training the members they were computed from (they'd leak).

Usage:
    .venv/bin/python notebooks/16_consensus_validator.py [--include path=csv ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / ".codex_work" / "consensus_validator"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

DEFAULT_OOFS = {
    "path3": ".codex_work/path3_e11_clean_train_v2_output/rsna_rad_e11/v52_e11_oof.csv",
    "path5": ".codex_work/path5_supervision_fusion_train_v1_output/rsna_rad_e11/v52_e11_oof.csv",
    "path6": ".codex_work/path6_kimi_fs160_train_v1_output/rsna_rad_e11/v52_e11_oof.csv",
    "path14": ".codex_work/path14_output/rsna_rad_e11/path14_reconciled_oof.csv",
    "path17": ".codex_work/path17_output/rsna_rad_e11/path17_gold_oof.csv",
    "path23": ".codex_work/path23_log/path23_diverse_oof.csv",
}


def rank_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df[TARGETS].rank(method="average", pct=True)


def load_oof(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.is_file():
        return None
    df = pd.read_csv(p, dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in df.columns:
        return None
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        return None
    df = df.drop_duplicates(subset=["StudyInstanceUID"], keep="first")
    df[TARGETS] = df[TARGETS].astype(float)
    return df.set_index("StudyInstanceUID")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", action="append", default=[],
                        help="extra name=path OOF csvs (new members)")
    parser.add_argument("--gold-labels", type=Path,
                        default=PROJECT_ROOT / "input" / "gold_labels.csv")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = dict(DEFAULT_OOFS)
    for spec in args.include:
        name, _, path = spec.partition("=")
        sources[name.strip()] = path.strip()

    oofs: dict[str, pd.DataFrame] = {}
    for name, path in sources.items():
        df = load_oof(path)
        if df is not None:
            oofs[name] = df
            print(f"loaded {name}: {len(df)} studies")
        else:
            print(f"skip {name}: not found/incomplete ({path})")
    if len(oofs) < 2:
        raise SystemExit("need at least two OOF sources")

    gold = pd.read_csv(args.gold_labels, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = gold[gold[TARGETS].notna().any(axis=1)]
    print(f"gold studies: {len(gold)}")

    # Per-target member weights from gold AUC (clipped at 0.5 so broken members
    # get zero weight; softmax over the clipped margins).
    common = sorted(set.intersection(*(set(df.index) for df in oofs.values())))
    print(f"common studies across members: {len(common)}")

    weights: dict[str, dict[str, float]] = {}
    member_gold_auc: dict[str, dict[str, float | None]] = {}
    consensus = pd.DataFrame(index=pd.Index(common, name="StudyInstanceUID"),
                             columns=TARGETS, dtype=float)
    for t in TARGETS:
        y_gold = gold[t].reindex(common)
        mask = y_gold.notna() & np.isfinite(y_gold)
        aucs = {}
        for name, df in oofs.items():
            scores = df[t].reindex(common)
            valid = mask & scores.notna() & np.isfinite(scores)
            if valid.sum() >= 10 and y_gold[valid].nunique() == 2:
                aucs[name] = float(roc_auc_score(
                    y_gold[valid].astype(int), scores[valid]
                ))
            else:
                aucs[name] = None
        member_gold_auc[t] = aucs
        margins = {n: max(a - 0.5, 0.0) if a is not None else 0.0 for n, a in aucs.items()}
        total = sum(margins.values())
        if total <= 0:
            w = {n: 1.0 / len(oofs) for n in oofs}
        else:
            w = {n: m / total for n, m in margins.items()}
        weights[t] = w
        stacked = np.zeros(len(common))
        coverage = np.zeros(len(common))
        for name, df in oofs.items():
            vals = rank_cols(df).reindex(common)[t].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            stacked += np.where(finite, w[name] * vals, 0.0)
            coverage += np.where(finite, w[name], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            consensus[t] = np.where(coverage > 0, stacked / np.maximum(coverage, 1e-9), np.nan)

    # Validate the consensus itself on gold vs each member.
    consensus_gold_auc = {}
    per_member_macro = {}
    for t in TARGETS:
        y_gold = gold[t].reindex(common)
        scores = consensus[t]
        valid = y_gold.notna() & np.isfinite(y_gold) & scores.notna() & np.isfinite(scores)
        if valid.sum() >= 10 and y_gold[valid].nunique() == 2:
            consensus_gold_auc[t] = float(roc_auc_score(
                y_gold[valid].astype(int), scores[valid]
            ))
        else:
            consensus_gold_auc[t] = None
    for name in oofs:
        vals = [
            member_gold_auc[t][name] for t in TARGETS
            if member_gold_auc[t][name] is not None
        ]
        per_member_macro[name] = float(np.mean(vals)) if vals else None
    cons_vals = [v for v in consensus_gold_auc.values() if v is not None]
    consensus_macro = float(np.mean(cons_vals)) if cons_vals else None

    # Per-study confidence: agreement between members (std of ranks).
    rank_stack = np.stack([rank_cols(df).reindex(common).to_numpy() for df in oofs.values()])
    agreement = 1.0 - np.nanstd(rank_stack, axis=0).mean(axis=1)
    consensus["confidence"] = np.clip(np.nan_to_num(agreement, nan=0.0), 0, 1)

    consensus.reset_index().to_csv(OUT_DIR / "consensus_labels.csv", index=False)
    report = {
        "n_members": len(oofs),
        "n_studies": len(common),
        "member_gold_macro_auc": per_member_macro,
        "consensus_gold_macro_auc": consensus_macro,
        "consensus_gold_per_target": consensus_gold_auc,
        "member_gold_per_target": member_gold_auc,
        "weights": weights,
        "note": (
            "Consensus = gold-AUC-weighted rank average of member OOFs. Use as "
            "the selection signal; never train members on it (leakage)."
        ),
    }
    (OUT_DIR / "consensus_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("n_members", "n_studies", "member_gold_macro_auc",
                       "consensus_gold_macro_auc")}, indent=2))
    print(f"wrote {OUT_DIR / 'consensus_labels.csv'}")


if __name__ == "__main__":
    main()
