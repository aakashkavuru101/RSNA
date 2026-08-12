# Cross-Dimension Insights — RSNA Knee Abnormality Detection (2026)

## Insight 1 — This is fundamentally a weak-label learning competition, not an image competition
The decisive resource is not MRI modeling sophistication but the quality of silver labels extracted from ~4,349 unlabeled multilingual reports: only 58 expert-labeled studies exist for training. Measured deltas (LLM 0.878 vs regex 0.814 macro AUC vs gold; +0.056 for soft/graded targets; targeted not-addressed imputation +0.112 on synovitis) show label quality is the single largest controllable lever — larger than any architecture choice documented across all knee-MRI literature (architectural spread among strong methods ≈0.01–0.03 AUC).
Derived from: Dim 01, 02, 08, 10. Confidence: high.

## Insight 2 — Reports are a train-time teacher only; the winning frame is "text-supervised image model"
Because test.csv has no Report column, any test-time text dependency is dead weight. The correct mental model: distill multilingual report knowledge into an image-only student (teacher–student / cross-modal distillation), optionally with report-conditioned auxiliary training losses. This also explains why the competition is "multimodal" in data but unimodal (image) at inference — a design most public commentary initially missed.
Derived from: Dim 01, 02, 07, 10. Confidence: high (host-confirmed premise).

## Insight 3 — The competition's difficulty is deliberately engineered at the evaluation layer
Three converging design choices make leaderboard trust dangerous: (a) macro AUC weights the worst class equally — rare/hard classes (Fracture, MCL, Synovitis) dominate score variance; (b) public LB is only ~30% of ~1,300 test studies with organizers explicitly warning prevalence is NOT matched across splits; (c) the 58 gold studies are pathology-enriched (every study ≥1 positive, mean 4.14 findings), so even local gold validation is biased. Robust scanner-grouped + iterative-stratified CV and conservative ensembling beat LB-chasing; expect shake-up.
Derived from: Dim 01, 02, 08. Confidence: high.

## Insight 4 — Domain shift (16 sites, 265 scanner fingerprints) is the hidden ceiling on generalization
Metadata-only models retain ~0.05 AUC of scanner memorization that dies under grouped CV; literature shows 0.07–0.10 AUC drops on external validation for knee MRI models (MRNet 0.911→0.824 zero-shot; CoPAS 0.812→0.72). The private test set's site composition is unknown — so techniques that trade a little train accuracy for invariance (intensity normalization, aggressive intensity augmentation, site-stratified validation, per-volume z-score) are net-positive, while ComBat-style harmonization demonstrably does not help DL.
Derived from: Dim 02, 04, 07, 08. Confidence: high.

## Insight 5 — The proven architecture recipe converges from three independent directions
Past RSNA winners (2022–2025), the knee-MRI literature (MRNet→ELNet→MPFuseNet→CoPAS), and modeling theory (attention-MIL > max-pooling) all converge on: per-plane/series 2D or 2.5D pretrained CNN → slice-sequence aggregator (BiLSTM or gated attention-MIL) → multi-head binary classifier, with optional localize-then-classify ROI stage as the reliable booster (+0.01–0.03). End-to-end 3D, large ViTs, and joint multi-plane models consistently underperform. A newcomer should implement this skeleton first, not explore architectures.
Derived from: Dim 04, 06, 07. Confidence: high.

## Insight 6 — The efficiency track is a separate, winnable game favoring distillation
With $18k across 3 places, judged on runtime+AUC (not accuracy rank), the efficiency track rewards: ELNet-class tiny models (0.2M params beating MRNet's 183M), student distillation of big ensembles, FP16/AMP + torch.compile, no TTA, fast DICOM I/O. A mid-tier main-track team can plausibly win efficiency — a rational strategy for a solo newcomer.
Derived from: Dim 01, 04, 06, 07, 08. Confidence: medium-high.

## Insight 7 — CoPAS is a legal, open-code starting template nearly identical to the task
CoPAS (Nat Commun 2024) targets the same 12 abnormalities on multi-plane multi-sequence knee MRI from 5 centers with public code — its 0.812 AUC (arthroscopy-referenced) vs the competition's current 0.94 LB shows the competition's report-supervision + scale advantages, but its architecture (per-plane branches + cross-plane attention + plane-aware fusion) is directly transplantable. Combined with the unresolved external-data ruling, cloning architecture (not data) is the safe reuse path.
Derived from: Dim 01, 04, 07, 09. Confidence: high.

## Insight 8 — Small-but-clean beats big-but-noisy: invest in validation before modeling
CheXbert-style evidence (weak-label pretraining + fine-tune on a small expert set nearly matches radiologists) maps directly onto this competition's structure: the 58 gold studies are the only honest anchor; every modeling decision (label extractor, imputation, augmentation, architecture) should be validated against them plus a held-out silver consensus — accepting that 58 studies carry high variance (<±0.02 macro AUC is noise).
Derived from: Dim 05, 08, 10. Confidence: high.

## Insight 9 — The 12 labels are clinically coupled, enabling smart multi-task structure
Labels cluster by tissue/mechanism (ligaments: ACL/MCL; menisci: medial/lateral; OA compartments: medial/lateral/PF; fluid: effusion/synovitis/Baker's; bone: contusion/fracture) with strong clinical co-occurrence (ACL tear ⇒ contusion pattern, effusion; meniscus damage ⇒ OA progression). This justifies shared representations with per-cluster heads, label-correlation-aware losses, and anatomically consistent TTA — including the critical trap that horizontal flip swaps medial↔lateral labels (a trick worth ~0.01 AUC in RSNA 2025).
Derived from: Dim 03, 04, 06, 08. Confidence: medium-high.

## Insight 10 — First-mover window is short; LB is already compressing
Public LB reached 0.9+ within one day and 0.942 within five days of launch, with shared LLM-label datasets and strong public baselines already commoditizing the first 0.89. Differentiation now requires second-order assets: better label ensembles (multi-LLM voting), localization stages, and CV discipline — not replicating public baselines.
Derived from: Dim 01, 02, 10. Confidence: high.
