# Verifier v1 — "Path to 0.95-0.96" analysis deliverable

## Acceptance criteria (all must pass)
1. CURRENT-LEADERBOARD: Current public-leaderboard state fetched live (top score, score spread, team count) with source + date; no fabricated numbers.
2. GAP-DIAGNOSIS: Explicit verdict on "knowledge gap vs model gap" with reasoning grounded in competition facts (12 labels, 58/4407 labeled studies, report-text supervision, macro-AUC metric).
3. QUANTITATIVE-CEILING: Analysis of what limits macro AUC (label noise, per-label difficulty, test-set properties), with per-label difficulty ranking tied to the radiology guide already produced.
4. ROADMAP: Concrete, ordered, actionable roadmap to 0.95-0.96 (data/labels, model, training, ensembling, validation), each item with expected gain rationale.
5. RISK-CONTROL: Validation/CV strategy vs public LB shake-up risk (group leakage by site/scanner, laterality derivation).
6. SOURCES: Key claims carry sources (Kaggle pages/discussions, literature); uncertainty flagged.
7. FORMAT: Markdown deliverable in /mnt/agents/output/, human-readable name, ≤ reasonable length, standalone.

## Check method
- Script checks 1,4,5,7 structurally (sections present, leaderboard numbers cited).
- Manual review for 2,3,6.
