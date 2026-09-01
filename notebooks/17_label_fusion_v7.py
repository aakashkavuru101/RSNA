"""silver_labels_v7 — audit-remediated fusion (2026-08-31).

Two changes against the v6 recipe, both traced to the label-pipeline audit:

  1. WHOLE-COLUMN winners. v5/v6's fuse_frames spliced per-cell fallbacks from
     differently-calibrated sources into one column, corrupting the ranking
     (PF OA: pure steven_v2 column 0.9455 gold AUC vs spliced column 0.9009).
     v7 ships the winning source's entire column per label, keeping its 0.5
     abstentions as 0.5 (training already downweights 0.5 cells).
  2. The margin-aware winner map (choose_winner, ±0.02 noise floor, incumbent
     = flight) is now actually USED for deployment — v6 computed it, audited
     it, and fused with raw AUC-argmax rankings instead.

Protocol (GOLD_INTEGRATION_PLAN.md §2): winner selection is validated
cross-fitted on input/gold_folds.csv, then the deployment map is re-fit on all
58 gold rows once the recipe is frozen. gold_usage: selection=crossfit,
deployment=full.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "label_fusion", ROOT / "notebooks" / "08_label_fusion.py")
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)

LABELS = lf.LABELS
MISSING = lf.MISSING
SOURCES = {
    "flight": ROOT / ".codex_work/public_datasets/flight_hybrid_labels/report_labels_v4hybrid.csv",
    "steven_v4": ROOT / ".codex_work/public_datasets/steven_labels/llm_labels_v4_blend.csv",
    "steven_v2": ROOT / ".codex_work/public_datasets/steven_labels/llm_labels_v2.csv",
    "pilkwang": ROOT / ".codex_work/public_datasets/pilkwang_labels/report_labels_v2.csv",
    "lixin": ROOT / ".codex_work/public_datasets/lixin_sol56_labels/report_labels_gpt56sol.csv",
}
MARGIN = 0.02
OUT_CSV = ROOT / "input" / "silver_labels_v7.csv"
OUT_AUDIT = ROOT / "input" / "silver_labels_v7_fusion_audit.json"


def whole_column_scores(mats, gold, folds, sources):
    """Cross-fit harness for the WHOLE-COLUMN policy: per fold k, winners are
    chosen on folds != k, and fold k's gold rows are scored with the winner's
    raw column value (0.5 abstentions included — that is what ships)."""
    gold_uids = gold["StudyInstanceUID"]
    aligned = {s: mats[s].set_index("StudyInstanceUID")[LABELS] for s in sources}
    per_fold_winners = {}
    scores = pd.DataFrame(np.nan, index=gold_uids, columns=LABELS, dtype=float)
    for fold in sorted(folds.unique()):
        train_rows = folds[folds != fold].index
        aucs = {s: lf.source_gold_aucs(mats[s], gold, train_rows) for s in sources}
        winners = {label: lf.choose_winner(aucs, label, list(sources), MARGIN)
                   for label in LABELS}
        per_fold_winners[int(fold)] = winners
        for uid in folds[folds == fold].index:
            for label in LABELS:
                value = aligned[winners[label]][label].get(uid, np.nan)
                scores.loc[uid, label] = value if pd.notna(value) else np.nan
    return scores, per_fold_winners


def main() -> None:
    t0 = time.time()
    gold = pd.read_csv(ROOT / "input" / "gold_labels.csv")
    folds_df = pd.read_csv(ROOT / "input" / "gold_folds.csv")
    folds = folds_df.set_index("StudyInstanceUID")["gold_fold"]
    folds = folds.loc[gold["StudyInstanceUID"]]

    mats = {name: lf.load_source(name, path) for name, path in SOURCES.items()}
    for name, m in mats.items():
        print(f"source {name}: {len(m)} studies")

    # --- cross-fitted evidence for the whole-column margin-aware policy ------
    scores, per_fold_winners = whole_column_scores(mats, gold, folds, SOURCES)
    imputed = scores.copy()
    mask = imputed["Synovitis"] == MISSING
    imputed.loc[mask, "Synovitis"] = imputed.loc[mask, "Effusion"] * 0.5
    cf_plain = lf.policy_aucs(scores, gold)
    cf_imp = lf.policy_aucs(imputed, gold)
    print("\ncross-fit (whole-column, margin 0.02):")
    print(f"  plain     macro: {cf_plain['macro']:.4f} ({int(cf_plain['n_labels'])} labels)")
    print(f"  syn-imput macro: {cf_imp['macro']:.4f} ({int(cf_imp['n_labels'])} labels)")

    # --- deployment: winners re-fit on all 58 gold (recipe frozen above) -----
    all_rows = folds.index
    aucs_full = {s: lf.source_gold_aucs(mats[s], gold, all_rows) for s in SOURCES}
    deploy_winners = {label: lf.choose_winner(aucs_full, label, list(SOURCES), MARGIN)
                      for label in LABELS}
    print("\ndeployment winners:", deploy_winners)

    uids = lf.ordered_union_uid(list(mats.values()))
    aligned = {s: mats[s].set_index("StudyInstanceUID")[LABELS] for s in SOURCES}
    fused = pd.DataFrame(MISSING, index=pd.Index(uids, name="StudyInstanceUID"),
                         columns=LABELS, dtype=float)
    for label in LABELS:
        column = aligned[deploy_winners[label]][label].reindex(uids)
        fused[label] = column.fillna(MISSING).to_numpy(dtype=float)
    fused = fused.reset_index()
    fused, n_syn = lf.synovitis_impute(fused)
    print(f"synovitis imputed cells: {n_syn}")

    # --- full-gold read of the deployed file (comparison, not selection) -----
    deploy_aucs = lf.policy_aucs(fused.set_index("StudyInstanceUID"), gold)
    macro_deploy = float(deploy_aucs["macro"])
    print(f"\ndeployed v7 full-gold macro: {macro_deploy:.4f} (v6 was 0.9006)")
    print("per-target:", {k: round(v, 4) for k, v in deploy_aucs.items()})

    lf.atomic_write_csv(fused, OUT_CSV)
    lf.atomic_write_json({
        "script": "notebooks/17_label_fusion_v7.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gold_usage": {
            "selection_evidence": "crossfit",
            "deployment_artifact": "full",
            "note": "whole-column margin-aware winners; per-fold winner maps "
                    "validated cross-fitted, deployment map re-fit on all 58; "
                    "written to silver_labels_v7.csv",
        },
        "margin": MARGIN,
        "policy": "whole-column (no per-cell splice)",
        "sources": {n: str(p.relative_to(ROOT)) for n, p in SOURCES.items()},
        "crossfit": {
            "per_fold_winners": per_fold_winners,
            "policy_auc": {k: (None if np.isnan(v) else round(v, 4)) for k, v in cf_plain.items()},
            "policy_auc_synovitis_imputed": {k: (None if np.isnan(v) else round(v, 4)) for k, v in cf_imp.items()},
            "macro_plain": round(float(cf_plain["macro"]), 4),
            "macro_synovitis_imputed": round(float(cf_imp["macro"]), 4),
        },
        "deployment_winners": deploy_winners,
        "deployed_full_gold_macro": round(macro_deploy, 4),
        "deployed_full_gold_per_target": {k: round(v, 4) for k, v in deploy_aucs.items()},
        "synovitis_cells_imputed": n_syn,
        "elapsed_seconds": round(time.time() - t0, 1),
    }, OUT_AUDIT)
    print(f"\nwrote {OUT_CSV.name} + audit in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
