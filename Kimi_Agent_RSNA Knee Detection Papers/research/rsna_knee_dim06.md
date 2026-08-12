# Dimension 06 — Winning Solutions Archaeology: Past RSNA Kaggle Competitions
**Target:** RSNA Knee Abnormality Detection (2026, knee MRI + reports, 12 abnormalities, ends 2026-10-22)
**Date compiled:** 2026-08-10
**Scope:** 1st-place + gold-medal solutions from RSNA 2022 (Cervical Spine Fracture, CT), RSNA 2023 (Abdominal Trauma, CT), RSNA 2024 (Lumbar Spine Degenerative, MRI — closest analog), RSNA 2025 (Intracranial Aneurysm, CT/MR multimodal), plus the Stanford MRNet knee-MRI literature as direct domain precedent.
**Search effort:** ~17 distinct web queries + 6 deep page opens (Kaggle write-ups via browser, GitHub READMEs, arXiv, CSDN/Zhihu/Zenn summaries). Kaggle discussion pages require JS; write-up pages were read directly.

---

## 1. RSNA 2022 Cervical Spine Fracture Detection (CT, volumetric, weighted multi-label log loss)

### 1a. 1st Place — Qishen Ha (@haqishen): 3D segmentation → 2.5D + LSTM

**Claim:** Two-stage pipeline: Stage 1 = 3D semantic segmentation of C1–C7 vertebrae (trained on only 87 masks); Stage 2 = 2.5D CNN + LSTM classification per vertebra and per patient.
- **Source:** Kaggle write-up "1st Place Solution" (Qishen Ha)
- **URL:** https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution
- **Date:** 2022-10-29
- **Excerpt:** "I designed a 2-stage pipeline to deal with this problem. stage1: 3D semantic segmentation -> stage2: 2.5D w/ LSTM classification... For 3D semantic segmentation, we only have 87 samples w/ 3d mask in the dataset, but it's sufficient to train 3D semantic segmentation models with good performance. I use 128x128x128 input, to train resnet18d or efficientnet v2s + unet model, for segmenting C[1-7] vertebraes (7ch output)."
- **Confidence:** High (primary source)

**Claim:** Slice handling: crop each vertebra via the predicted 3D mask; extract 15 evenly spaced slices along z; each slice stacked with ±2 adjacent slices → 5-channel "2.5D" image; the predicted vertebra mask is appended as a 6th channel.
- **Source:** same write-up
- **Excerpt:** "for each vertebrae sample, I extracted 15 slices evenly by z-dimension, and for each slice, I further extracted +-2 adjacent slices to form an image with 5 channels... I added the predicted mask of corresponding vertebrae as the 6th channel to each image, as a way to exclude the effect of having multiple vertebraes in a single sample."
- **Confidence:** High

**Claim:** 3D CNN classification failed; 2.5D + LSTM won. Two model types: type1 = vertebra-level (15 slices → 2D CNN → LSTM), type2 = patient-level (7×15 images at once, learns `patient_overall`). Type2 needed small backbones due to GPU memory.
- **Source:** same write-up
- **Excerpt:** "Theoretically the easiest way to deal with this data is to train 3D CNN on it. But unfortunately this method does not work. Training a 3D CNN on this data did not give me satisfactory results. So I backed off and chose the 2.5D approach." / "However, the disadvantage of this model is that it takes up too much GPU memory and therefore can only use small backbones."
- **Confidence:** High

**Claim:** Final ensemble: 2× 5-fold 3D seg models + 6 classification models (effv2s 512², convnext tiny 384², convnext nano/pico 512², nfnet l0 384²), mixup used; total submission runtime 7.5 h on Kaggle; ensemble gave ~+0.02 CV; type2 added ~+0.02 on patient_overall.
- **Source:** same write-up + author comments
- **Excerpt:** "The submission time is 7.5 hours." / "Ensemble make CV score around ~0.02 better." / "Type2 make the CV of patient_overall 0.02 better, so for whole CV 0.01 or so." / "To avoid overfitting, mixup is one of the choses."
- **Confidence:** High

**Claim:** Training loop details (stage 2): AdamW, AMP/GradScaler, CosineAnnealing(WarmRestarts), mixup applied with probability p_mixup.
- **Source:** Zenn.dev summary of 1st-place code
- **URL:** https://zenn.dev/kabupen/articles/da4a2d180e9e62
- **Date:** 2023-09-03
- **Excerpt:** "AdamW, GradScaler / torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=eta_min)" + mixup training-loop code.
- **Confidence:** Medium-High (secondary but code-verbatim)

### 1b. Other 2022 gold patterns
- **5th place (Team Speedrun, pascal-pfeiffer et al.)** — 3-stage: 2D vertebra classification/segmentation → slice-level fracture pseudo-labels (vertebra class probability × fracture label) → study-level aggregation model. GitHub: https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place. Excerpt: "Train a 2D classification/ segmentation model for vertebrate using the provided segmentation labels... multiply the result with the overall fracture label to get 2D image level labels... Collect all image level labels per study and train an aggregation model to predict the given study level labels." Confidence: High.
- **2nd place (RAWE)** — 2.5D CNN + UNet segmentation, then CNN (tf_efficientnetv2_s, resnest50d) + BiGRU + Attention classifier; 24 slices per vertebra. Source: CSDN summary of Kaggle discussion 365115 + https://github.com/ryanyuerong/RSNA2022RAWE. Confidence: Medium (secondary summary).
- **6th place (i-pan)** — 3D CNN (X3D) vertebra-level classifiers on cropped 3D chunks + TD-CNN slice models with pseudo-labels. GitHub: https://github.com/i-pan/kaggle-rsna-cspine. Confidence: High (repo).
- **Outcome note:** top-8 models averaged AUC 0.96 on hidden test (PMC10831508, "Performance of the Winning Algorithms of the RSNA 2022 Cervical Spine Fracture Detection Challenge"). Confidence: High.

---

## 2. RSNA 2023 Abdominal Trauma Detection (CT, volumetric, weighted log loss)

### 2a. 1st Place — Team Oxygen (Harshit Sheoran, Nischay Dhankhar, Qishen Ha)

**Claim:** Three parts: (1) 3D organ segmentation → masks/crops; (2) 2.5D 2D-CNN + GRU/RNN for kidney/liver/spleen/bowel; (3) same for bowel + extravasation. CV = 4-fold GroupKFold at patient level.
- **Source:** Kaggle write-up "1st Place Solution: Team Oxygen"
- **URL:** https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen
- **Date:** 2023-10-22
- **Excerpt:** "Split used: 4 Fold GroupKFold ( Patient Level)... Part 1: 3D segmentation for generating masks / crops [Stage 1]. Part 2: 2D CNN + RNN based approach for Kidney, Liver, Spleen & Bowel [Stage 2]. Part 3: 2D CNN + RNN based approach for Bowel + Extravasation [Stage 2]"
- **Confidence:** High (primary)

**Claim:** Volume handling: study-level crop from organ segmentation bounding boxes; 96 equidistant slices per volume reshaped to (32, 3, H, W) with adjacent slices as 3 channels (2.5D); soft per-slice targets = patient label × normalized organ-visibility curve; slice-level predictions max-aggregated to study level.
- **Source:** same write-up + GitHub https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- **Excerpt:** "each volume extracted with equi-distant 96 slices for a study which is then reshaped to (32, 3, image_size, image_size) in a 2.5D manner... 3 channels are formed by using the adjacent slices... Then we multiply targets * patient-level target for each middle slice of the sequence and that is our label... simple maximum aggregation is applied on sigmoid predictions to fetch study level prediction used in submissions."
- **Confidence:** High

**Claim:** Auxiliary segmentation loss from shared encoder (Unet decoder or plain conv head on last/2nd-last blocks, Dice loss) gave +0.01 to +0.03; main loss BCE; AdamW, LR 1e-4–4e-4, cosine + warmup; image size 384²; augs = Perspective/HFlip/VFlip/Rotate(±25).
- **Source:** same write-up
- **Excerpt:** "One of the key things which made our training much more stable and helped in improving scores was using auxiliary losses based on segmentation... This trick gave us around +0.01 to +0.03 boost in our models." / "Loss: BCE Loss for Classification, Dice Loss for segmentation... loss = loss1 + (loss2 * 0.125)"
- **Confidence:** High

**Claim:** Class imbalance handled by balanced sampling, not weighted loss: "we are sampling injured and non-injured patients equally"; weighted BCE "did not help model converge."
- **Source:** same write-up, comment by Harshit Sheoran
- **Excerpt:** "Yes, we did try weighted BCE loss, in our experiments, it did not help model converge, one reason for this could be our sampling, in our GitHub code, you can see that we are sampling injured and non-injured patients equally"
- **Confidence:** High

**Claim:** Final ensemble = multiple CoAt-Lite-Medium / CoAt-Lite-Small / EfficientNetV2-S + GRU models across folds and full-data seeds, two preprocessing variants (own soft-tissue windowing vs TheoViel's); best single 4-fold OOF CV 0.326 (CoaT lite medium), ensemble 0.31x. Inference ran on Kaggle GPU notebooks; extra pip packages shipped as Kaggle datasets (no internet).
- **Source:** same write-up + GitHub README
- **Excerpt:** "Architectures used in Final ensemble: Coat Lite Medium w/ GRU... Coat Lite Small w/ GRU! Efficientnet v2s w/ GRU [Timm]" / "All packages were installed via uploaded kaggle dataset." / "No major postprocessing was applied except for tuning scaling factors based on CV scores."
- **Confidence:** High

### 2b. Other 2023 gold patterns
- **2nd place (TheoViel, "On Strike")** — segmentation-assisted cropping + CNN with GRU head; two-stage training ("RNN only sees probabilities precomputed by the CNN, so training is done in 2 stages"); no explicit imbalance handling. Code: https://github.com/TheoViel/kaggle_rsna_abdominal_trauma. Confidence: High.
- **3rd place ("sheep")** — organ crops from enlarged segmentation masks, two crop scales per organ, custom batch sampler grouping same-organ boxes per batch ("masking for liver model / custom sampler for all class models" were biggest accuracy contributors). Source: Kaggle discussion 447464 via kaggle.curtischong.me mirror. Confidence: Medium-High.

---

## 3. RSNA 2024 Lumbar Spine Degenerative Classification (MRI, multi-series, 25 condition-level labels × 3 severities; metric = sample-weighted log loss, weights 1/2/4 + any_severe_spinal) — CLOSEST ANALOG TO KNEE TASK

### 3a. 1st Place — NANACHI (team "Avengers", @wadakoki): localization cascade + bi-LSTM + Attention-MIL

**Claim:** 2-stage pipeline; stage 1 split into (a) instance_number (slice) prediction per disc level using a 3D ConvNeXt (volumes 0–1 normalized, sorted by DICOM metadata, depth-padded to 32), trained jointly with classification (CE over 32 positions) and regression (L1 on normalized xyz) heads; (b) 2D coordinate (xy) prediction with ConvNeXt-base / EfficientNetV2-L on a 3-channel center-slice image, pretrained on @brendanartley's public coordinates dataset.
- **Source:** Kaggle write-up "1st place solution" (NANACHI)
- **URL:** https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution
- **Date:** 2024-10-28
- **Excerpt:** "In this part, I used simple 3D ConvNeXt to predict instance_number for each level. Data that is fed into models is just normalized from 0 to 1, sorted by dicom's metadata and padded 32 to depth direction to align shape... I trained models 2 tasks, regression and classification, and I used L1 Loss and Cross Entropy Loss respectively." / "I used ConvNeXt-base and Efficientnet-v2-l for this task. Before I train these models, I trained these models using @brendanartley's dataset. These pretrained models were slightly better than pretrained models that were trained using imagenet."
- **Confidence:** High (primary)

**Claim:** Stage 2 = severity prediction from cropped 5-slice stacks around predicted coordinates; core architecture = ConvNeXt-small / EfficientNetV2-s encoder + bi-LSTM + Attention-based MIL with auxiliary attention loss. Attention MIL improved public LB 0.37→0.35; bi-LSTM + aux losses + ensembling → 0.33. Aux loss alone: validation 0.2624→0.2522 (5-fold mean).
- **Source:** same write-up + author comment
- **Excerpt:** "I used ConvNeXt-small and Efficientnet-v2-s as the encoder. After implementing Attention-based MIL, my public LB score was improved from 0.37 -> 0.35. Then, adding bi-LSTM, aux losses and ensembling improve my score from 0.35 to 0.33." / Comment: "public LB improved 0.3729 -> 0.3588 (diff is 0.0141) and private LB improved 0.4259 -> 0.4062 (diff is 0.0197)." Code: `self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)` + separate `attention` and `aux_attention` (Tanh→Linear→1) heads.
- **Confidence:** High

**Claim:** Crucial augmentation = robustness to stage-1 errors: random shift of predicted coordinate (±10 px) and random shift of instance_number (±2, with shift probability matched to each stage-1 model's measured error distribution).
- **Source:** same write-up
- **Excerpt:** "random shift of coordinate x and y (-10~+10 pix) / random shift of instance_number (-2~+2. shifting probability was decided error probability of each instance_number prediction models)... Especially, random shift of instance_number was crucial for robustness of error of 1st stage." (Post-crop: RandomBrightnessContrast p=0.25, ShiftScaleRotate p=0.5.)
- **Confidence:** High

**Claim:** What didn't work (2024 1st): MAMBA/self-attention instead of bi-LSTM; weight sharing between aux and main attention; cross-modality inputs for mismatched conditions; long epochs (7 for convnext-s, 14 for effv2-s best); large models (convnext-large < base < small); ViTs worse than conv nets.
- **Source:** same write-up
- **Excerpt:** "what didn't work: MAMBA and Self-Attention instead of bi-LSTM / sharing weight between aux_attention layer and attention layer... / long epochs... / large models (convnext-large < convnext-base < convnext-small in my experiments) / vision transformers (I think this was my problem. but convolution models were better than vits in my experiments)"
- **Confidence:** High

### 3b. 6th Place — NVSpine (TheoViel, darraghdog et al.): localization + RNN classifiers + metric-optimizing MLP stacker

**Claim:** Ensemble of 3 pipelines: (1) CoAtNet keypoint model → crops → CoAtNet + RNN classifiers (13 frames for sagittal, 3 RNN heads for left/right/center) trained with CE "tweaked to maximize the AUC"; (2) effnet-v2-t xy/z localizers + 2.5D effnet-v2-t + biRNN(512) crop classifiers with weighted CE; (3) 2.5D effv2-s with 3D-patched convs on concatenated multi-view 288×288×9 cubes with view-dropout augmentation. All aggregated by an MLP stacker that directly accounts for the competition metric; per-condition independent MLPs worked best.
- **Source:** Kaggle write-up "6th Place Solution" (NVSpine)
- **URL:** https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/nvspine-6th-place-solution
- **Date:** 2024-10
- **Excerpt:** "Final results are aggregated using an MLP model that directly optimizes the competition metric... Surprisingly, what worked best here is to consider each target independently." / "A 2.5d model of efficientnetv2_rw_t with a bidirectional single layer RNN head (512 dim) and weighted CE loss is used." / "Dropping one of the views with a chance of 10% is used as additional augmentation."
- **Confidence:** High (primary)

**Claim:** NVSpine "what did not help": SpineNet/medical-specific models for condition-level modelling; pseudo-labelling on external data; denoising; aux localization encoder-decoder; bi-encoder axial+sagittal joint models.
- **Source:** same write-up
- **Excerpt:** "What did not help - SpineNet and other medical specific models for condition level modelling. - Pseudo labelling on external data - Denoising techniques - Encoder-decoder architectures to add an auxiliary injury localization task - Bi-encoder architectures to jointly learn on axial and sagittal images"
- **Confidence:** High

**Claim:** Trusted CV over public LB: "Our experience from previous RSNA competitions led us to expect a small yet reasonable shake-up, and we trusted our CV more than public LB." (4-fold CV; final CV 0.382 / public 0.355 / private 0.401.)
- **Source:** same write-up. **Confidence:** High

### 3c. Other 2024 gold/silver patterns
- **2nd place (IanPan-Kevin-Yuji-Bartley)** — per-member diversity: YOLOX disc detection → ConvNeXt-S classification; CNN-transformer slice selection + keypoints + MIL/attention-pooling models (maxvit/coatnet/NFNet/CSN-ResNet101), 27-pattern slice/spatial-offset crops, 3- and 5-channel slices; center-24-frames + encoder+LSTM+attention pooling with sequence-reversal, left/right swap, manifold mixup, 9× rotation TTA, 50:50 real:pseudo labels; noise filtering by dropping high-loss samples; post-hoc ×1.25 scaling of "severe" probabilities boosted score. Source: Zenn summary (https://zenn.dev/mkj/articles/ba630a8837ee72, 2024-12-20). Confidence: Medium-High (secondary).
- **3rd place (SonySpine/tkmn/Moyashii)** — CenterNet-style detectors (EffNetB6/B4+FPN) → 2D encoder + attention center/side classifiers; ignored level/left-right distinctions to multiply data; manual label correction + pseudo-labels; AdamW + OneCycleLR. Source: same Zenn summary. Confidence: Medium-High.
- **5th place (siwooyong)** — two-step metric-aware training: pretrain for AUC without weighted loss, then fine-tune only the head with weighted + "any" loss. GitHub: https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification. Excerpt: "this led to overfitting on the weighted labels, resulting in poor auc score... 2nd-step(finetuning) We employed weighted loss and any loss, freezing the model's backbone and training only the head parameters to optimize for the competition metric." Confidence: High (repo verbatim).
- **7th place (hengck23 + lhwcv)** — one-stage pvt_v2_b4 with 3D decoder heads + two-stage crop models; shape-alignment preprocessing; 5-fold. GitHub: https://github.com/hengck23/solution-rsna-2024-lumbar-spine. Confidence: High.
- **9th place (adamnarai)** — DeepLabV3Plus Gaussian-heatmap keypoint detection per series type → level-wise ROI crops → 2.5D ensemble with GRU heads (ResNet18/Swin-Tiny/ConvNeXt-Nano). GitHub: https://github.com/adamnarai/kaggle-rsna-2024. Confidence: High.
- **Silver (DaoyuanLi2816)** — YOLOv8 per condition; 75 bbox classes encoding level×side×severity; max aggregation; probability normalization to sum 1. GitHub: https://github.com/DaoyuanLi2816/rsna-2024-lumbar-spine. Confidence: High.
- **Metric (verbatim):** "Sample Weights: Normal/Mild: 1, Moderate: 2, Severe: 4" + any_severe_spinal log-loss term (from DaoyuanLi2816 README and Zenn metric description). Confidence: High.

---

## 4. RSNA 2025 Intracranial Aneurysm Detection (CTA/MRA/MRI multimodal, 14 labels; metric = weighted multi-label AUC, "Aneurysm Present" weighted 13/14... [per competition])

### 4a. 1st Place — tomoon33: vessel-segmentation-guided ROI classification (coarse-to-fine)

**Claim:** Pipeline: DICOM→NIfTI (dcm2niix, gdcmconv fallback), RAS reorientation, per-volume z-score; coarse-to-fine nnU-Net vessel segmentation (model 1 at 1mm³ iso for ROI candidate via DBSCAN on coarse mask → 140³ mm crop; models 2&3 at (0.80,0.45,0.44)mm with SkeletonRecall/Tversky losses for thin-vessel recall); then a 3D ROI classifier whose backbone is an nnU-Net pretrained on vessel segmentation, with vessel-region-masked pooling per location + Location-Aware Transformer + MLP heads, plus auxiliary sphere (r=5px) detection task at annotated aneurysm sites.
- **Source:** Kaggle write-up "1st Place Solution" (tomoon33)
- **URL:** https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution
- **Date:** 2025-10-15
- **Excerpt:** "The core of my approach is a robust, coarse-to-fine pipeline that uses vessel segmentation to guide a region-of-interest (ROI) based classifier, producing location-aware predictions." / "Backbone: The core of the model is an nnU-Net pre-trained for the vessel segmentation task. This approach was more accurate and faster to train than standard 2.5D or 3D timm backbones." / "I treated each of the 14 labels as an independent binary classification problem. This design helps with the severe class imbalance, as positive cases for any single location are very rare."
- **Confidence:** High (primary)

**Claim:** Loss weighting prioritized auxiliary localization: weights 0.1 (13 location BCE) / 0.05 (Aneurysm-Present BCE) / 1.0 (aux sphere segmentation, Balanced-BCE + Focal-Tversky++); "higher weights on the classification losses led to overfitting." Ablation: backbone pretraining on segmentation was the single biggest factor (0.794 vs 0.902 without); aux seg loss +0.026; Location-Aware Transformer ~+0.02 macro AUC.
- **Source:** same write-up
- **Excerpt:** "I set the weights to 0.1 for the 13 location losses, 0.05 for the Aneurysm Present loss, and 1.0 for the auxiliary sphere segmentation loss... I found that higher weights on the classification losses led to overfitting, so this balance was important." / Ablation table: "Without backbone pretraining 96×192×192 0.777/0.811/0.794" vs "Final model 96×192×192 0.907/0.898/0.902."
- **Confidence:** High

**Claim:** Training/inference: AdamW lr 1e-4, effective batch 8 via grad accumulation, cosine + warmup, EMA weights, multilabel-stratified 5-fold CV, ensemble = mean of 4 folds, TTA = original + left-right flip average. Kaggle 2×T4 inference ≈ 4.0 s preprocessing + 10.7 s segmentation + 3.3 s classification per series. ~60 series excluded for data quality. Fail-safe fallback to OOF-mean probabilities on pipeline anomaly.
- **Source:** same write-up
- **Excerpt:** "The final predictions are an average of the models from 4 of the 5 cross-validation folds... I averaged the predictions from the original volume and a left-right flipped version of it." / "EMA Weights: I used the Exponential Moving Average (EMA) of the model weights for inference." / "If an anomaly occurred... it fell back to a set of pre-determined probabilities... the mean of the out-of-fold predictions from my cross-validation set."
- **Confidence:** High

### 4b. 7th Place — MIC-DKFZ (Isensee team): single 3D nnU-Net blob regression

**Claim:** Task formulated as multichannel EDT-"blob" (heatmap) regression with a single nnU-Net ResEnc U-Net (6 stages, [32,64,128,256,320,320]), trained from scratch; each aneurysm = EDT-transformed sphere (best radius 65 voxels) in one of 14 channels (14th = pixelwise max of 13); loss = BCE on worst 20% voxels (TopK); prediction = max per channel over patches. Single model, no TTA, public/private 0.83/0.83.
- **Source:** Kaggle write-up "7th place solution - 3D nnU-Net + blob regression (again)" + GitHub https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution
- **URL:** https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution
- **Date:** 2025-10-16
- **Excerpt:** "We formulate the task as multichannel blob regression, optimized using a TopK (20%) BCE loss and then taking the maximum per channel as probability prediction." / "Our final model was trained with a batch size of 32 and a patch size of 96,x160x128 voxels. Initial learning rate is 0.01... We train with SGD for 3000 epochs... The loss function utilized is binary cross-entropy, computed only on the 20% worst voxels."
- **Confidence:** High (primary)

**Claim:** Practical details: 200×160×160 mm center-superior ROI crop; resample to median spacing (0.70, 0.47, 0.47) with a fast PyTorch resampler; global z-score normalization; 5-fold CV stratified by modality (not vessel class); hyperparameters tuned on fold 0 only; no left/right mirroring (labels encode side); training 4.5 days on 4×A100-40GB; inference ~8 h on 2×T4 with patches split across GPUs.
- **Source:** same write-up + README
- **Excerpt:** "We split the provided challenge data into five cross-validation folds, stratifying for modalities across folds... most of the hyperparameter tuning happened exclusively on the first fold." / "we did not apply mirroring augmentations in the left/right axis, since several of the labels contained a left/right codification."
- **Confidence:** High

**Claim:** What did not work / constraints: nnDetection (Retina U-Net) better on public but worse private & internal; isotropic 1mm resampling "substantially worsened" results; co-training on external aneurysm datasets "did not really help"; larger patch sizes did not improve; Gaussian patch weighting & TTA helped but couldn't be used due to Kaggle platform timeouts; whole-image processing was dropped for an ROI crop which was both faster and better.
- **Source:** same write-up
- **Excerpt:** "Isometric space resampling with [1.0, 1.0, 1.0] mm... substantially worsened our results, so it was discontinued early on." / "In the end, co-training did not really help, so we resorted back to training only on the challenge cases from scratch." / "We first started with processing the image as a whole but time limitations forced us to crop around the ROI which also resulted in better performance."
- **Confidence:** High

### 4c. 2025 winners list / context
- Winners: 1 tomoon33, 2 BraveCoWCoW, 3 BTYND, 4 Harshit Sheoran, 5 more CV challenge pls, 6 Ian/Theo & Bartley, 7 MIC-DKFZ, 8 Konni, 9 Vibes and Genius Trade-Off; Educational Merit: Ian/Theo/Bartley + MIC-DKFZ. Source: https://www.rsna.org/news/2025/november/2025-ai-challenge-winners (2025-11-25). Confidence: High. (Note: 2025 used 13 anatomical locations + "Aneurysm Present"; weighted-AUC metric — per RSNA/ASNR challenge page.)

---

## 5. Direct knee-MRI precedent: Stanford MRNet and successors

**Claim:** MRNet (Bien et al., Stanford, PLOS Medicine): per-slice AlexNet (ImageNet-pretrained) features → element-wise max-pool across slices → FC; one model per task×plane (9 models); plane predictions combined by logistic regression; AUCs 0.937 abnormal / 0.965 ACL / 0.847 meniscus.
- **Source:** PMC6258509 "Deep-learning-assisted diagnosis for knee magnetic resonance imaging" (MRNet)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/
- **Excerpt:** "AlexNet features from each slice of the MRI series are combined using a max pooling (element-wise maximum) operation. The resulting vector is fed through a fully connected layer to produce a single output probability. We trained a different MRNet for each task... and series type (sagittal, coronal, axial), resulting in 9 different MRNets... combined predictions from 3 series per exam using logistic regression."
- **Confidence:** High

**Claim:** MRNet training: per-example loss scaled inversely to class prevalence; augmentation = random rotation ±25°, shift ±25 px, hflip 0.5; transfer learning from ImageNet essential given ~1,130 exams.
- **Source:** same + Azcona et al. arXiv:2010.01947
- **Excerpt:** "The loss of a sample was scaled inversely proportionate to the prevalence of that sample's class in the dataset" / "we initialized the weights of the AlexNet portion of the MRNet to values optimized on the ImageNet database... then fine-tuned these weights."
- **Confidence:** High

**Claim:** Follow-ups: ResNet18-per-plane + logistic-regression stacking reached 0.934 avg AUC, beating AlexNet MRNet; per-plane/per-task augmentation-probability grid search was the biggest single gain; combining planes in one network or multi-task loss hurt ("negative transfer among tasks"). Interpolating to a fixed 15 slices beat center-slice selection (per Tsai/ELNet-era comparisons).
- **Source:** Azcona thesis/paper (doras.dcu.ie PDF + arXiv:2010.01947) and MDPI review https://www.mdpi.com/2504-4990/3/4/50
- **Excerpt:** "The approaches that use a separate network for each plane and task... give better performance than those trained on a combination of planes... or using a multi-objective loss. This is somewhat surprising and indicates some negative transfer among tasks." / "the factor that we believe most contributed outperforming the published baseline was the grid search on the percentage of images being augmented."
- **Confidence:** Medium-High

---

## 6. Cross-competition synthesis: recurring winning patterns (evidence above)

1. **Two-stage "localize-then-classify" dominates RSNA volumetric comps.** 2022 1st (vertebra seg→crops→2.5D+LSTM), 2023 1st (organ seg→crops→2.5D+GRU), 2024 1st (instance+coordinate models→crops→biLSTM+MIL), 2025 1st (vessel seg→ROI→classifier). Small annotation sets suffice for stage 1 ("only 87 samples w/ 3d mask... sufficient").
2. **2.5D >> 3D for classification under Kaggle constraints.** "Training a 3D CNN on this data did not give me satisfactory results" (2022 1st); 2024 1st found conv nets > ViTs, small > large. The arXiv AnyMC3D paper confirms: "winning solutions in recent 3D classification challenges also adopt this strategy [pretrained 2D + slice fusion]" and characterizes the canonical RSNA recipe as "finetunes 2D pretrained backbones with 2.5D input and uses bidirectional-LSTM for slice fusion" (arxiv.org/html/2512.12887v2).
3. **Slice/sequence fusion recipe:** adjacent-slice channel stacking (3–6 channels) + per-slice features → BiLSTM/GRU/attention-pooling → max or attention aggregation to study level. Attention-based MIL was the 2024 winner's key differentiator (+0.014 public / +0.020 private).
4. **Auxiliary localization losses are the most reliable booster:** 2023 1st +0.01–0.03 (aux dice on encoder features); 2024 1st aux attention/depth heads (0.2624→0.2522); 2025 1st aux sphere-detection loss weighted 10–20× higher than classification losses (ablation +0.026); MIC-DKFZ's entire model is a detection-style blob regression.
5. **Stage-1-error-robustness augmentation:** jitter predicted coordinates/instances during stage-2 training with probabilities matched to measured stage-1 error (2024 1st "crucial"); NVSpine "augmenting the target coord position was very effective."
6. **Class imbalance: prefer balanced sampling over loss re-weighting.** Team Oxygen: weighted BCE hurt convergence, equal sampling won; 2024 5th: weighted loss overfit → two-step (AUC pretrain → frozen-backbone weighted fine-tune). For knee's 12 abnormalities with a weighted metric, replicate the metric in the loss only at the fine-tune/head stage.
7. **CV discipline:** patient-level GroupKFold (4–5 folds), stratify by modality where relevant; tune on fold 0 only when compute-limited; trust CV over public LB ("we trusted our CV more than public LB" — NVSpine); train final models on full data / more seeds.
8. **Ensembles:** 4–20+ models: multiple folds × seeds × 2–3 backbones (CoAtNet, ConvNeXt-small/base, EfficientNetV2-s, NFNet) × 2 preprocessing variants; simple mean/max aggregation; post-hoc probability scaling tuned on CV (2023 1st "tuning scaling factors"; 2024 2nd ×1.25 on severe class). TTA when budget allows (hflip / 9 rotations / LR-flip), but 2025 showed single-model no-TTA can still medal when time-constrained.
9. **Preprocessing standards:** modality-appropriate intensity handling (CT: windowing e.g. soft-tissue; MRI: per-volume min-max or z-score; DICOM window tags), consistent orientation (RAS), sort slices by DICOM metadata, equidistant resampling of slice count (15–96 slices), robust DICOM readers (dicomsdl, dcm2niix+gdcmconv fallback, pydicom+SimpleITK).
10. **Kaggle inference-constraint playbook:** ship packages as Kaggle datasets (no internet); keep pipeline per-series fast (2025 1st ≈ 18 s/series on 2×T4; MIC-DKFZ split patches across 2 GPUs); wrap inference in try/except with fallback predictions; precompute/cached preprocessing; budget total runtime (2022 1st: 7.5 h submission); BETA-test submission notebooks early because platform timeouts killed MIC-DKFZ's later (better) checkpoints.

### Transfer to "RSNA Knee Abnormality Detection" (12 abnormalities, MRI + reports)
- Expect multi-plane multi-sequence MRI per study (sag/cor/axial × T1/T2/PD-fat-sat) — mirror the 2024 lumbar pattern: per-series-type slice/coordinate localization models → ROI crops → 2.5D CNN (ConvNeXt-small / EfficientNetV2-s / CoAtNet) + BiLSTM or attention-MIL per abnormality group; max/attention pool to study level.
- 12 abnormality heads: treat as independent binary tasks (tomoon33: "I treated each of the 14 labels as an independent binary classification problem. This design helps with the severe class imbalance"); consider an "any abnormality" head = max/union (2025 channel-max trick; MRNet per-task models + logistic stacking as fallback).
- If report text is provided: no past RSNA winner used text, but late-fuse report embeddings with image logits via a small MLP stacker that optimizes the competition metric (NVSpine MLP precedent) rather than early fusion.
- Weighted metric: pretrain unweighted for AUC, fine-tune head with class/sample weights matching the metric (siwooyong two-step); tune per-class probability scaling on OOF.
- Bone/cartilage/meniscus ROI cropping from a small segmentation or keypoint model is likely the highest-value stage-1 investment (consistent across all 4 competitions); MRNet literature confirms ImageNet-pretrained 2D CNNs + slice max-pooling already reach radiologist-level AUCs on knee MRI, so a strong 2.5D baseline is cheap to build.
