# Research Dim 3 — RSNA Knee Abnormality Detection Competition Context

## Premise corrections (high confidence)
1. **Competition is LIVE (as of Aug 2026)**, not concluded. Started July 30, 2026; entry/merger deadline Oct 15, 2026; final submission Oct 22, 2026; winners announced Nov 2026 (RSNA 2026, Chicago). ~1,764 teams; public LB ~0.93. NO winning-solution writeups exist yet — only early baselines, EDA notebooks, discussion threads.
2. **Data is NOT MRNet/Stanford-derived.** Brand-new RSNA AI Challenge dataset: >5,000 knee MRI exams from 16–19 sites across five continents, paired with original radiology reports in ~9–12 languages — first RSNA image+text multimodal challenge. MRNet only discussed as external data.
Sources: kaggle.com/competitions/rsna-knee-abnormality-detection/overview | rsna.org/media/press/2026/2669

## Data (official data page)
- Files: train.csv (StudyInstanceUID, free-text Report, 12 binary labels), train_series.csv (StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive [1 = T2/PD/STIR-like], Fat_Suppression, Anatomical_Plane [Sagittal/Coronal/Axial]), DICOMs train_series/<Study>/<Series>/<SOP>.dcm, one .dcm per slice. ~819,640 files, 569.76 GB.
- Metadata does NOT give named sequences (no "sagittal T1 / coronal PD FS") — only plane + fluid-sensitivity + fat-suppression flags. Host: fluid-sensitive (PD/T2 FS) sequences are where most abnormalities are detected. Community EDA counts ~9,864 Sagittal / 8,609 Coronal / 5,898 Axial series.
- Slices per series: typically 20–45 (median 30), long tail to a few hundred.
- Image dimensions explicitly VARIABLE: "Intensities, orientations, and resolutions vary across series and studies."
- DICOM: mixed transfer syntaxes (Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE); 86-tag allowlist.
- Labels: only 58 of 4,407 train studies have per-condition labels; the other 4,349 have report text only (core challenge = mine labels from multilingual reports). Test ~1,300 studies, no Report at test time.
- Metric: macro-averaged ROC AUC over 12 targets. Code competition, internet off, ≤9h, external data + pretrained models allowed.

## Label taxonomy (matches the user's 12-finding list exactly)
Official columns: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
**Host-posted annotation criteria (discussion/733343):**
- ACL = high-grade partial (>50% fibers) or full tear
- MCL = high-grade acute tear
- Meniscus = signal touching surface on ≥2 images OR morphologic deformity
- OA = ≥1 cm of high-grade (>50% thickness) cartilage loss per compartment
- Effusion, Baker's = moderate or large
- Borderline findings graded NEGATIVE
- Reference labels double-read by MSK radiologists with third-reader adjudication.
- No "abnormal" umbrella label; no PCL/LCL labels.

## Forum / early-baseline techniques (community practice, NOT validated winners)
- Physical-scale central crop (discussion 734105, DINOv2 replication): "We crop to a fixed 130 mm field and resize to 336 px, ~0.387 mm/px… check that the requested crop actually fits; otherwise it can silently become a no-op"; rationale: resizing a larger FOV without physical cropping can erase 1–3 mm pathology. Notebook fleongg/rsna-knee-dinov2-base-physical-scale-soup (LB 0.768) names 130 mm physical crop as key change; amanatar used 160 mm; romanrozen EDA/baseline (LB 0.809–0.894) has "Sampling at a fixed physical scale" section. → 130–160 mm physical crops are live community practice, not official prescription.
- Slice handling: sort by ImagePositionPatient · (row×col from ImageOrientationPatient), NOT filename; no Laterality column — derive laterality from image-center geometry (flip coronal/axial horizontally, reverse sagittal stack order) because 5 of 12 targets are medial/lateral pairs. Slice sampling: 2.5D ResNet-34 baseline maps 24 depth slices as channels on 160×160; a 3D baseline samples 16 uniform slices/plane at 192×192; MIL/attention pooling over slices proposed. Warn against averaging pretrained conv1 weights for channel-stacking.
- Planes per finding (host overview): cruciate ligaments & menisci best on sagittal + coronal; patellofemoral cartilage on axial; fluid-sensitive FS sequences for most findings. Community: per-plane encoders + cross-plane fusion into 12 sigmoid heads.
- Pretrained: DINOv2 (small/base) dominant backbone; ImageNet ResNets in baselines. MRNet, fastMRI+, OAI, SKM-TEA raised as external data (discussion 733652); host clarifications (733965): commercial hosted LLM APIs may NOT be sent report text; local open-weights LLMs fine.
- Metadata-only probe: 0.65 macro AUC random-fold / 0.598 scanner-grouped → no meaningful metadata shortcut.

## MRNet / Bien et al. 2018 (confirmed)
- PLoS Medicine 15(11):e1002699 (Nov 27, 2018) — NOT Nature Medicine.
- 1,370 knee MRI exams, Stanford 2001–2012, GE (56.6% 3T). Protocol: coronal T1, coronal T2 FS, sagittal PD, sagittal T2 FS, axial PD FS; model used sagittal T2, coronal T1, axial PD (one series per plane).
- Slices per series 17–61 (mean 31.5, SD 8.0); images scaled 256×256; intensity-standardized.
- Labels: abnormal 80.6%, ACL tear 23.3%, meniscal tear 37.1%.
- Model: AlexNet per slice → global avg pool → max-pool across slices → FC; 9 task×plane models; logistic-regression fusion across planes.
- Plane importance (LR coefficients): axial PD most beneficial for abnormality and meniscal tear; coronal T1 for ACL tear. AUCs 0.937 / 0.965 / 0.847. External validation zero-shot 0.824 → 0.911 after retraining.

## Knee MRI FOV vs 130 mm crop
- Confirmed protocol values: FOV most commonly 14–16 cm (16 cm with 240×320–384×320 matrices, 3–4 mm slices; ranges 12–20 cm occur). In-plane resolution ~0.4–0.6 mm/px.
- Inference (anatomy-based, no single source): 130 mm central crop is smaller than typical 140–160 mm FOV → trims ~5–15 mm periphery per side. For average adult knee (joint-line width ~9–12 cm) bones/menisci/cruciates stay inside, BUT: MCL is subcutaneous at medial skin margin; Baker's cyst sits in posteromedial popliteal soft tissues; both near FOV edges → tight 130 mm crop can clip them in large knees, off-center acquisitions, small-FOV sagittal series; can also trim suprapatellar pouch (effusion) on sagittals. Matches forum warning to verify crop fits. No published source quantifies diagnostic loss — treat as plausible risk.
- Bottom line: official facts = 12-label taxonomy (matches user list), new multi-site DICOM dataset (not MRNet), 20–45 slices/series, variable resolution, only 58 labeled train studies. Everything about 130 mm crops, slice sampling, plane weighting, DINOv2/MRNet pretraining = early-competition community practice, not validated winning knowledge.
