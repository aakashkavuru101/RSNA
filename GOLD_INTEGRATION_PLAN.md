# Gold Integration Plan — breaking the 0.915 plateau

**Date:** 2026-08-19
**Status:** Approved direction (supersedes the "gold never used for training" policy in
`AGENTS.md`; see §6 for the replacement protocol)
**Context:** Public LB stuck at 0.915 (Path5/Path10 lineage). Top of LB is 0.952 with a
0.937+ pack. Repo audits attribute the gap to (a) silver-label quality (supervision ceiling),
(b) competitors' use of the 58 gold labels for label override/selection, which our clean
policy forbade. Test set is **3 studies** (confirmed in path13/path16 apply audits,
`test_studies: 3`), which makes per-label ranking quality and LB-read strategy dominant.

---

## 1. Why gold integration is now safe to do

The original ban existed to preserve one honest validation anchor. It did its job: we have
a full battery of models trained gold-free, each with an **unbiased gold-monitor read**
(heads were trained on cached DINO features with `reconciled_w[gold] = 0.0`, so every
existing `gold_monitor_auc` is untainted). We are not giving that up — we are adding a
cross-fitted protocol so gold can inform training/selection without ever being evaluated
on data it trained on.

Competition rules permit it: gold labels are part of the official training data. Public
0.93-class artifacts already do this (E11/E10 contracts override weak labels with the 58
official labels; see `.codex_work/supervision_audit/supervision_research_2026-08-17.md`).

## 2. Validation honesty protocol (the contract)

1. Split the 58 gold studies into 5 folds, stratified by per-label prevalence, fixed and
   saved as an artifact (`input/gold_folds.csv`). This split is immutable once created.
2. Any model or blend that **consumes** gold in training or weight selection must be
   evaluated only on gold folds it never saw (cross-fitted gold OOF).
3. Final submission candidates may train/select on all 58 gold rows — but only after the
   candidate recipe is frozen via cross-fitted evidence.
4. Silver scanner-grouped OOF remains the primary metric for anything that does not touch
   gold. Gold OOF (cross-fitted) is the primary metric for anything that does.
5. Noise floor: n=58, some labels have <10 positives. Treat gold-OOF differences < ±0.02
   as unmeasurable. Prefer per-target decisions over macro averages.
6. Every artifact records `gold_usage: none | crossfit | full` in its audit JSON.

## 3. Path17 — gold-integrated head training (cheap, do first)

Heads train on cached DINO features (`FoundationQueryHead`, feature: global_average_pool),
so gold experiments cost minutes of GPU, not hours. Design:

- **Gold override:** replace silver cells with binary gold labels for the 58 studies,
  weight λ relative to silver rows. Sweep λ ∈ {1, 2, 4, 8} and gold oversampling ×{1, 4}.
- **Cross-fit:** for each gold fold k, train on silver + gold folds ≠ k, predict fold k.
  Produces an honest 58-study gold OOF per (λ, oversample) setting.
- **Selection:** per-target, pick the λ with the best cross-fitted gold AUC, subject to the
  ±0.02 noise floor and non-regression on silver OOF (≥ −0.005 per target vs Path6 parent,
  mirroring the Path14 gate but with gold as the gain metric instead of silver).
- **Output:** `path17_gold_heads.pt`, `path17_gold_oof.csv` (cross-fitted gold OOF +
  silver OOF), audit JSON with per-target λ map.
- Build as `kaggle/path17_gold_crossfit_train/build_path17.py` mutating the proven Path6
  FS160 notebook (same pattern as `build_path14.py`).

## 4. Per-target blend selection on cross-fitted gold OOF

Current blends (Path10/15/16) were weight-selected on silver OOF, which is blind. Redo
selection on the honest gold reads:

- **Candidate pool:** Path5 heads, Path6/FS160, Path10 stack, Path11 replace, Path14
  reconciled, Path17 gold heads, plus hash-pinned public teacher OOFs already vetted.
- **Method:** per-target constrained greedy over blend weights, objective = cross-fitted
  gold OOF AUC per target, with a minimum-parent-share constraint (≥0.7 on the 0.915
  parent) to bound variance, and silver-OOF non-regression as a sanity check.
- **Final blend trains on all 58 gold** (full-gold variant of the winning recipe) for
  submission; selection evidence stays cross-fitted.

## 5. Label pipeline v2 — disagreement-targeted re-extraction

The 0.778 in-house extractor is the weakest link and the biggest remaining ceiling-raiser.
Targeted, not blanket:

- Re-extract only the high-disagreement cells: Synovitis (39.4% teacher disagreement),
  Contusion (15.4%), Medial Meniscus (15.3%). Case list already exists:
  `.codex_work/supervision_audit/high_disagreement_cases.csv`.
- Multi-pass LLM with self-consistency (k=5 votes), positivity-threshold system prompt,
  evidence-span output (existing `src/labels/extractor.py` schema).
- Gate: re-extracted cells must improve gold AUC on those three targets without
  regressing others (>±0.02 noise floor). Then retrain heads (Path17 recipe) on the
  updated silver matrix.

## 6. AGENTS.md amendment

Replace the "Do-Not-Touch" gold line and warning #2's implication with:

> `input/gold_labels.csv` (and the 58 labeled rows in `train.csv`) may be used for
> training and blend selection ONLY under the cross-fit protocol in
> `GOLD_INTEGRATION_PLAN.md` §2. Never evaluate a candidate on gold folds it trained on.
> Every artifact must record `gold_usage: none | crossfit | full`.

`models/*/oof/` remains never-overwrite; raw competition data remains read-only.

## 7. Submission strategy under a 3-study test set

- Each LB read is highly informative (per-label ordering of 3 studies), but the public
  split is ~30% of 3 — expect violent private shake-up. Favor candidates that win across
  ALL evidence (silver OOF + gold OOF + LB), never LB alone.
- Submission order (5/day budget): (1) Path15 clean target push, (2) Path16
  original+Path14, (3) Path13 Aman super-0920 — all already built, unscored. Then
  Path17-based blends as they arrive.
- Keep a submission ledger: file sha256 → score → hypothesis confirmed/refuted.
- Final picks: best cross-fitted gold OOF full-gold ensemble + the most decorrelated
  runner-up. Expect the public LB to mislead; trust the protocol.

## 8. Sequencing

| # | Workstream | Where | Cost |
|---|---|---|---|
| 0 | Submit Path15/16/13 for LB reads | Kaggle web/API | 3 submissions |
| 1 | Path14 gold re-eval (running) | local | minutes |
| 2 | `input/gold_folds.csv` + gold-OOF harness script | local | minutes |
| 3 | Path17 build script + kernel push | local → Kaggle GPU | ~1 GPU-h |
| 4 | Per-target blend re-selection | local | minutes |
| 5 | Disagreement re-extraction v2 | local (LLM API) | ~$1 |
| 6 | Full-gold final training + submission | Kaggle GPU | ~2 GPU-h |
