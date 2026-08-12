# Cross-Verification — RSNA Knee Abnormality Detection Research (2026-08-10)

## HIGH CONFIDENCE (≥2 independent agents, consistent sources)
- **Metric**: unweighted macro-averaged ROC AUC over 12 binary exam-level labels: Final Score = (1/12) Σ AUC_i. Confirmed from live evaluation page (dim08) + submission format (dim01). Column names: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture (dim01, dim02, dim10).
- **Supervision design**: only 58 of 4,407 train studies carry expert image-derived labels (~1.3%); all 4,407 have reports; reports are TRAIN-ONLY (test.csv has no Report column, host-confirmed). Report text → silver labels only (~82.5% agreement with gold). (dim01, dim02, dim08, dim10)
- **Annotation protocol**: 2 subspecialty MSK radiologists + 3rd adjudicator; borderline = negative (specificity-favoring); positivity thresholds published (ACL >50% fibers; meniscal signal reaching surface on ≥2 images; OA ≥1 cm high-grade cartilage loss). (dim01, dim02, dim10)
- **Rules**: notebook-only code competition, ≤9h runtime, internet off at submission, 5 subs/day, team ≤5; external data/pretrained models allowed if publicly accessible at minimal cost; commercial LLM APIs permitted for label extraction (host ruling, thread 733965). (dim01, dim10)
- **Prizes**: $77,000 = main LB 10 places + efficiency track 3 places ($7k/$6k/$5k); first RSNA efficiency award; CC-BY-NC 4.0 winner license. (dim01, press release)
- **Timeline**: start Jul 30/Aug 5 2026; entry/team deadline Oct 15; final submission Oct 22; winners' materials Nov 5; RSNA 2026 Nov 29–Dec 3 Chicago. (dim01, dim10)
- **Data structure**: DICOM, one slice per file, train_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm; 819,640 files / 569.76 GB; ~24,371 series, 20–45 slices typical; 86-tag allowlist; train_series.csv flags Fluid_Sensitive / Fat_Suppression / Anatomical_Plane. (dim02)
- **Best prior art**: CoPAS (Qiu et al., Nature Communications 2024) — same 12 knee abnormalities, 5 centers, avg AUC 0.812 internal / 0.72 external, code public (github.com/zqiuak/CoPAS). Found independently by dim01, dim04, dim07.
- **MRNet anchor**: Bien et al. PLoS Med 2018, AUC 0.937/0.965/0.847 (abnormal/ACL/meniscus); 2D CNN + slice max-pool + per-plane logistic regression. (dim03, dim04, dim06, dim07)
- **Winning paradigm transfer**: past RSNA winners = localize-then-classify + 2.5D backbones + BiLSTM/attention-MIL slice aggregation; 3D end-to-end and large ViTs consistently fail. (dim06, supported by dim07 literature)
- **Label extraction**: LLM-derived report labels beat regex (0.8780 vs 0.8136 macro AUC vs the 58 gold); 25.4% of report cells "not addressed" — silence ≠ negative. (dim01, dim02, dim08, dim10)
- **Public LB state (as of Aug 10)**: top ≈0.942; strong baselines (pilkwang v1 0.891 public, T4×2, 6.5 min); DICOM-metadata shortcut ruled out (0.65 random / 0.60 scanner-grouped). (dim01, dim02, dim08)

## MEDIUM CONFIDENCE (single authoritative source or indirect)
- Efficiency formula orientation: Efficiency ≈ (Benchmark − AUC)/(Benchmark − max AUC) + RuntimeSeconds/32400 — KaTeX partially lost in extraction; direction (minimize, runtime-weighted) confirmed by dim01 + dim08 but exact normalization flagged medium.
- Languages attested ≈10 (EN/ES/NL dominant; FR/DE/PT/IT/TR tail; BG/GR observed) — resolves RSNA "nine" vs "a dozen" inconsistency (dim02, dim10).
- Kaggle hardware menu (P100 / T4×2; no L4x evidence) — verify at competition time (dim09).
- OrthoFoundation (1.25M knee X-ray/MRI SSL foundation model, weights on GitHub) — single-agent find, medium-high (dim05).
- Graded/soft targets +0.056 macro over binary; targeted effusion→synovitis imputation 0.678→0.790 — community-reported, plausible, single source each (dim10).

## CONFLICT ZONES (resolved / flagged)
1. **"Nine" vs "a dozen" languages** — RESOLVED as temporal/rounding inconsistency: challenge page said nine (volunteer-call era), press release says a dozen; ~10 attested by community EDA. (dim02, dim10)
2. **Competition start date**: Kaggle lists Jul 30, 2026 vs RSNA public launch Aug 5, 2026 — RESOLVED: soft open Jul 30, public announcement Aug 5. (dim01)
3. **External data eligibility (MRNet/OAI/fastMRI+)**: gated click-through datasets NOT yet officially ruled on — UNRESOLVED; treat as pending; monitor Kaggle rules thread. (dim01, dim09)
4. **TensorRT on Kaggle**: INT8 can be ~147× slower than PyTorch FP16 on T4 if TensorRT libs missing — flag for efficiency track; safe path = FP16/AMP + torch.compile. (dim07, single source, arXiv:2607.08241)
5. **ComBat harmonization**: does NOT help DL models under scanner shift (Sci Rep 2023) vs traditional radiomics practice — RESOLVED: use intensity normalization + augmentation + site-aware CV instead. (dim07)

## LOW CONFIDENCE / GAPS
- Exact per-class prevalence in the hidden test set (only the 58 pathology-enriched gold known; organizers warn prevalence differs across splits).
- Full distribution of DICOM Rows×Columns and complete language list (notebook bodies not fully renderable).
- "KneeXNet" 2025 near-perfect claims — flagged low confidence by dim04.
- No published knee MRI+report fusion paper exists (dim04 negative finding) — open research opportunity, not a gap in evidence.
