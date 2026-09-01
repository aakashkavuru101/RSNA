# Gap Analysis — RSNA Knee Abnormality Detection
**Date:** 2026-09-01 · **Standing:** public LB **0.936**, rank **272 / ~2,700** · **Target:** 0.949–0.952 · **Final deadline:** 2026-10-22

This document is the complete evidence record of where the score comes from, where it doesn't, what every experiment proved, and the quantified campaign to ≥0.948. Sources: three forensic audits of this repo (2026-08-31), the label showdown against gold, dissection of both public 0.936 pipelines, LB distribution analysis, thread 735304 testimony, and eight scored submissions of our own.

---

## 1. Executive summary

1. **The leaderboard is a fork economy.** Two public Apache-2.0 notebooks (Roman Tamrazov "DINOsaur", Anhad Mahajan "Take Care Of Your Knee") define the public frontier. 427 teams sit tied at 0.936; 147 teams are now above it and climbing as newer public versions (internally "V5.x") spread. Riding forks can never produce a lead — everyone holds the same cards.
2. **Labels are solved; stop investing there.** Our fused silver labels (v6: 0.9006 gold macro; v7 cross-fit 0.8945) are at parity with the best public sources (flight 0.8991) against a measured per-label oracle of 0.9071. Nobody's supervision is meaningfully better than ~0.90.
3. **The measured gap is single-model quality, then folds.** The 0.944–0.950 tier is built from one strong single model (0.93–0.94 LB) five-folded (+0.008–0.009), not from ensembles of public notebooks. Direct testimony (thread 735304): Scott Willis (5th) — *single model, single fold 0.938 → 0.947 with 5 folds*, a small efficiency-class model; Tucker Arrants — single 0.934→0.942 @224px; Tom Aindow — single 0.915 = DINOv2 @392px, 150 mm crop, random 32-slice bags.
4. **Our own line converts 0.80 internal AUC into 0.876 LB** (Path33 calibration, 2026-09-01). The conversion deficit against the frontier's craft was traced to concrete pipeline decisions — fixed slice subsets, 130 mm crops, deleted head components — most now fixed, the rest scheduled.
5. **Fusion of public artifacts is a measured dead end.** Two-spine fusion scored 0.928 (worse than either 0.936 parent — silent stage degradation at hidden-test scale); weak-member overlay scored 0.935 (−0.001). Both were single-submission experiments, both are closed.

---

## 2. Submission ledger (the record)

| Date (UTC) | Path | Artifact | Score | Verdict |
|---|---|---|---|---|
| 08-12→08-24 | 1–21 | pre-audit era: DINOv2 pipelines, Aman-0920 blends | 0.500→**0.920** | plateaued on obsolete public parent |
| 08-30 | 21v3 | Aman parent re-run | 0.920 | old lineage ceiling confirmed |
| 08-30 | own ×3 | first own-line ensembles | 0.867 ×3 | members too weak/correlated |
| 08-30 | 24 | frozen-CSV per-target override | **blank** | 3-row submission vs ~1,300-study rerun; slot refunded |
| 08-31 | 25 | verbatim DINOsaur V4 fork | **0.936** | frontier banked; rank 877→~270 |
| 08-31 | 26 | two-spine fusion (DINOsaur × Take-Care) | 0.928 | **negative result**: arm B degraded silently at rerun scale |
| 08-31 | 28 | spine + our fx members @12% on 6 targets | 0.935 | **negative result**: 0.795-OOF members subtract even bounded |
| 09-01 | 33 | pure own stack (4 own families, no public weights) | 0.876 | **calibration**: own line's honest LB position |

Mechanics that cost us early: the visible test set is a 3-study placeholder while the hidden rerun is ~1,300 studies (public LB ≈ 30% ≈ 390); any kernel recombining frozen attached CSVs emits 3 rows and scores blank (Paths 15×2, 24). Kaggle keeps the best score — experiments can never lower standing.

---

## 3. Label pipeline — findings and closure

**Showdown on the 58 gold studies (macro AUC):** ours v6 **0.9006** > flight 0.8991 > steven_v4b 0.8927 > steven_v2 0.8876 > ours v5 0.8836 > pilkwang 0.8700 > sol56 0.8351 > in-house extractor 0.808. Oracle (best source per label) 0.9071.

Audit findings, all verified against artifacts:
- **v5 self-sabotage:** 94 override cells from a 10-study in-house re-extraction (Lateral OA eval 0.222 — below chance) were spliced as hard 0/1, bypassing the cross-fit harness: −0.017 macro. Nine kernels trained on v5. v6's entire gain is removing them.
- **The v6 "upgrade" was a no-op by construction:** the gpt-4o-mini corpus excluded all gold rows → NaN gold AUC → ranked last → contributed **0 of 52,176 cells**. The strongest in-house extractor (ox-alpha-free, 0.8840) died at 12/4,349 reports on provider 503s; qwen stalled at 17% (87 h ETA); the recorded root-cause note contradicts its own log.
- **Machinery ignored its own safety rails:** the margin-aware winner map (±0.02 noise floor) was computed, audited, and never used — fusion ran on raw AUC-argmax, deciding Fracture's 4,407 labels on a 0.0012 gap. Per-cell fallback spliced differently-calibrated LLMs into single columns.
- **v7 (2026-08-31)** implements whole-column margin-aware winners: cross-fit 0.8868→**0.8945** (honest +0.008); deployed full-gold 0.8991 ≈ v6. Conclusion: **supervision is saturated at ~0.90**; the celebrated "PF OA splice −0.045" partly conflated eval populations. Persistent supervision-limited targets everywhere: Synovitis ~0.79, Lateral OA ~0.83, Effusion/Lateral Meniscus ~0.88.

**Closed.** Remaining label headroom ≈ +0.01 macro at best; it is not the competitive lever.

---

## 4. Model recipe — the conversion gap, itemized

Public recipes convert 0.887-class labels into 0.915–0.942 single models (**image models exceed their teachers**). Our pre-fix members converted 0.90-class labels into 0.787 OOF. The audited causes:

| # | Defect | Evidence | Status |
|---|---|---|---|
| 1 | Fork deleted the parent's slot priors + focal top-k pooling | per-target signature: ACL 0.699/MCL 0.710 cratered, diffuse targets fine | **fixed** (fx1) |
| 2 | Interleaved pseudo-RGB: channels ~7 slices apart | fix existed in `localized_dino_train`, was reverted | **fixed** (fx1) |
| 3 | Undertrained: batch 3 vs 8, epochs 8 vs 10, unfreeze 4 vs 6; val curves rising at stop | fold curves monotone to last epoch | **fixed** (fx1/v3) |
| 4 | Sagittal hflip TTA anatomically invalid + phantom label swap | sagittal M↔L is through-plane | **removed** |
| 5 | Scoring kernel averaged raw sigmoids across scanner-calibrated folds | train kernel rank-normalizes; submit kernel didn't (~0.011 OOF bias) | **fixed** |
| 6 | Laterality from a single slice of one series: 49.7% resolved | poisons all 4 M/L targets | **fixed** (all-series median) |
| 7 | InfoNCE multimodal branch: temperature inverted (÷14.29) + detached | mm6 OOF 0.7873 vs 0.7871 plain — a full GPU run bought 0.0002 | **abandoned** |
| 8 | **Fixed 6–9 slice subsets reused every epoch** (RAM artifact, not a decision) | top recipes: random 16–32-slice bags per epoch | **fixed in v3** (random window starts) |
| 9 | **130 mm crop cuts edge anatomy** | Tom's public recipe: 150 mm @392px (0.383 mm/px) | **fixed in v3** (150 mm); 392px variant pending |

**Measured payoffs so far:** fixes 1–6 → OOF 0.787→0.7953 (+0.008, 5-fold). v3 (fixes 8–9, @224px): fold-0 OOF **0.8005 in 1.8 h** (⅓ the compute) with the val curve still rising at epoch 14, and the focal targets moving exactly as predicted: ACL 0.718→**0.800**, MCL 0.687→**0.749**, Fracture 0.860→**0.877**, Baker's 0.869→**0.883**.

**RaptorX v1** (our whole-volume CoAtNet, 32-slice routed volume, per-target window attention, our labels): holdout 0.786 macro — but **Lateral Meniscus 0.802**, the best LM any of our models has produced (everything else ~0.73). Keeps a per-target seat in future fusion. v2 needs a schedule fix (SWA averaged a decayed tail; peak was 0.792 at epoch ~5).

**Calibration (Path33):** equal rank-mean of all four own families = **0.876 LB**. Mapping: ~0.80 scanner-OOF ≈ 0.876 LB for this line. To stand at ~0.93 LB alone, the line needs roughly ≥0.85 scanner-OOF — the 5-fold + resolution program's target.

---

## 5. The frontier, dissected (what 0.936 actually is)

Both public 0.936 notebooks are forks of one four-stage spine:
1. DINOv2-small slot-transformer, **20–24 member** multi-config ensemble (training loop public in Take-Care; anatomical slot priors, focal pooling, physical-mm crops, per-member per-target weight vectors);
2. DINOv3 ViT-S/16 5-fold, rank-blended at 0.45 (weights public, loop not);
3. RadImageNet ResNet-50 frozen heads ×3 slot layouts + an **88-feature LB-fit linear calibrator** (rank branches, target-group means, and 12 DICOM protocol counts from `test_series.csv`) applied at 0.40 to 7 targets;
4. CoAtNet-384 "Raptor" whole-volume window model — **the largest single weight (~0.5) and the only stage with no public training code** — fine-tuned on a 4,349-study soft-label corpus; its author's ablation: growing that corpus 3,155→4,349 was worth **+0.013** on their gold gate;
5. Per-target correlation-damped percentile-rank fusion end to end (nothing probability-averaged).

Strategic readings: (a) the one defensible moat in the public stack is the Raptor-class arm — which is why our RaptorX exists; (b) both "independent" 0.936s share the spine, capping fork-fusion upside at ±0.002 (Path26 confirmed worse); (c) the V5.x line (per-window focal evidence on the Raptor arm) is already spreading — the public frontier will keep inflating without us doing anything, and the only counter is arms it doesn't contain.

---

## 6. Leaderboard dynamics

- 08-30: 27 teams ≥0.944; 363 tied at 0.936; we were 877th at 0.920.
- 09-01: **81 teams ≥0.940, 147 above 0.936, 427 tied at 0.936**; we are 272nd at 0.936.
- Low-entry high scores prove public-artifact sufficiency up to ~0.938 (single-submission 0.938 exists) — and its ceiling: every team ≥0.944 has double-digit submissions behind a private edge.
- Public LB = ~30% of ~1,300 studies. The 0.936 tie block rises and falls **together** at private shakeup; only original arms decouple us. Final selection (2 submissions, Oct 22) should follow the cross-fit gold protocol, not public-LB decimals.

---

## 7. The campaign to ≥0.948 (quantified)

**Arithmetic:** target single-model 0.93-class (per §4's mapping: ≥0.85 scanner-OOF) → 5-fold +0.008–0.009 → 0.94-class own stack → per-target cross-fitted fusion with the spine → 0.945–0.949. Every step has a measured precedent (Scott: 0.938 single-fold → 0.947).

**Phase 1 — done (08-31→09-01):** recipe fixes validated (+0.008); bag-sampling + 150 mm validated on fold-0 (+ focal-target jumps at ⅓ compute); RaptorX v1 trained (LM specialist); own-line LB calibration measured (0.876); v3 epoch-headroom probe (e20) in flight.
**Phase 2 — week of 09-06 (30 h GPU):** member v3 **5-fold** at the e20-informed config (~12–15 h); v3-hi variant @392px/150 mm, reduced slots to fit RAM (~2 folds, ~8 h); RaptorX v2 with fixed schedule if budget allows (~5 h). Gate: v3 5-fold scanner-OOF ≥0.83; hi-res fold beats 224 twin by >noise.
**Phase 3 — week of 09-13:** scale the winner into a multi-config own stack (2 resolutions × 2 seeds × best family) → target own-stack ≥0.90 LB standalone; begin per-target cross-fitted fusion with the spine (selection on gold folds per `GOLD_INTEGRATION_PLAN.md` §2, never on in-sample gold, never on 3-study orderings).
**Phase 4 — late September:** corpus-grown RaptorX v3 (the +0.013 lever the public author demonstrated), efficiency-track candidate, final-submission selection: best cross-fitted ensemble + most decorrelated runner-up.

**Budget:** 30 h GPU/week (T4×2), resets Saturday 00:00 UTC; ~26 h consumed this week across 6 trainings + commits. 5 submissions/day (UTC); slots are experiments — spend on information, never on unvalidated blends.

**Risks:** (i) public frontier inflation outpacing Phase 2–3 (mitigation: the spine is always available as fusion scaffold; our arms add on top); (ii) T4 ceiling vs top teams' hardware (mitigation: Scott's small-model proof; 224px craft first, resolution second); (iii) private-LB shakeup (mitigation: scanner-grouped CV + cross-fit gold as selection truth; tie-block decoupling via original arms is itself shakeup insurance).

---

## 8. Standing decisions

1. No fusion-of-public-forks submissions (measured: ≤0, twice).
2. No member overlays below ~0.85 OOF (measured: −0.001).
3. No decisions from in-sample gold or 3-study orderings (Path24 post-mortem).
4. Every submission kernel regenerates predictions dynamically and asserts loudly; frozen-CSV recombination is banned; silent fail-safes are banned.
5. Public forks are benchmarks and scaffolds only (Path30/V5.4 sits committed, unsubmitted, as a measuring stick); originality is the strategy, per team decision 2026-08-31.
6. Kaggle CLI OAuth tokens expire hourly and the CLI doesn't persist refreshes — re-persist via `kaggle auth print-access-token` capture when API calls start failing.

---

## Appendix A — Own artifact inventory

| Artifact | Where | Honest metric |
|---|---|---|
| silver_labels v5/v6/v7 (+ audits) | `input/` (local), Kaggle dataset `rsna-knee-silver-labels-v6` | v6 0.9006 full-gold; v7 0.8945 cross-fit |
| fx1 base/small (5 folds each) | kernels `path27-member-fixed-train`, `path29-member-small-train` | OOF 0.7953 / 0.7915 |
| member v3 fold-0 (+e20 probe) | `path32-member-v3-train`, `path34-member-v3e20-train` | fold-0 OOF 0.8005 @1.8 h |
| RaptorX v1 SWA | `path31-raptorx-train` | holdout 0.786; LM 0.802 |
| Own-stack submission | `path33-own-stack-submit` | **LB 0.876** |
| Frontier forks (scaffolds) | `path25-dinosaur-v4-parent` (LB 0.936), `path30-dinosaur-v54` (committed, unsubmitted) | reference only |
| Label engine v6/v7 + consensus tooling | `notebooks/11–17` | see §3 |
