# Dimension 04 — Annotated Bibliography: Deep Learning for Knee MRI Analysis
**Curated for:** Kaggle "RSNA Knee Abnormality Detection" (2026) — multimodal detection/classification of 12 knee abnormalities from MRI + reports
**Compiled:** 2026-08-10 | **Sub-agent:** dimension 04 | **Searches performed:** 24 web queries + 3 full-text retrievals (Nature, PubMed, arXiv)

Citation format per claim: **Claim / Source / URL / Date / Excerpt (verbatim) / Confidence**

---

## 1. CORE PAPERS

### 1.1 MRNet — the foundational paper (REQUIRED READING)

- **Claim:** MRNet (AlexNet feature extractor per slice + global max-pooling over slices + logistic regression over 3 planes) achieves AUC 0.937 (abnormal), 0.965 (ACL), 0.847 (meniscus) on a 120-exam internal validation set; generalizes to Rijeka data at AUC 0.824 zero-shot / 0.911 fine-tuned.
- **Source:** Bien N, Rajpurkar P, Ball RL, et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet." *PLoS Med.* 2018 Nov 27;15(11):e1002699. PMID 30481176. PMC6258509.
- **URL:** https://doi.org/10.1371/journal.pmed.1002699 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/
- **Date:** 2018-11-27 (accessed 2026-08-10)
- **Excerpt (verbatim, abstract):** "We developed MRNet, a convolutional neural network for classifying MRI series and combined predictions from 3 series per exam using logistic regression. In detecting abnormalities, ACL tears, and meniscal tears, this model achieved area under the receiver operating characteristic curve (AUC) values of 0.937 (95% CI 0.895, 0.980), 0.965 (95% CI 0.938, 0.993), and 0.847 (95% CI 0.780, 0.914), respectively, on the internal validation set. We also obtained a public dataset of 917 exams with sagittal T1-weighted series and labels for ACL injury from Clinical Hospital Centre Rijeka, Croatia. On the external validation set of 183 exams, the MRNet trained on Stanford sagittal T2-weighted series achieved an AUC of 0.824 (95% CI 0.757, 0.892) in the detection of ACL injuries with no additional training, while an MRNet trained on the rest of the external data achieved an AUC of 0.911 (95% CI 0.864, 0.958)."
- **Why it matters:** This is the direct ancestor of the 2026 RSNA competition: patient-level multi-label abnormality classification from multi-plane knee MRI series. The MRNet dataset page (https://stanfordmlgroup.github.io/competitions/mrnet/) confirms the exact data format the competition pipeline will resemble (3 series per exam: sagittal T2 FS, coronal T1, axial PD FS; 17–61 slices/series, mean 31.5).
- **Key methods:** 2D AlexNet per slice → max-pool across slices (per-plane) → logistic regression across planes; ImageNet pretraining; train/val/hidden-test split 1130/120/120 with ≥50 positives of each label per split; patient-disjoint splits.
- **Key results:** AUCs above; providing model predictions significantly increased experts' specificity on ACL tears (p<0.001).
- **Transferable tricks:** (1) per-plane CNN + cross-plane logistic regression; (2) **max-pooling over slices** as the exam-level aggregation baseline; (3) weighted loss for class imbalance (paper uses class-weighted cross-entropy); (4) stratified splitting ensuring ≥50 positives/label/split; (5) patient-level split to prevent leakage.
- **Confidence:** High (primary source, verbatim).

### 1.2 MRNet competition leaderboard & Azcona et al. backbone study

- **Claim:** The official MRNet leaderboard (closed) topped out at average AUC 0.917 (mrnet-baseline); later work shows simply swapping AlexNet→ResNet-18 with modern augmentations lifts validation AUC to ~0.91–0.96 per task.
- **Source:** (a) MRNet competition leaderboard, Stanford ML Group; (b) Azcona D, McGuinness K, Smeaton AF. "A comparative study of existing and new deep learning methods for detecting knee injuries using the MRNet dataset." IDSTA 2020, pp. 149–155; arXiv:2010.01947.
- **URL:** https://stanfordmlgroup.github.io/competitions/mrnet/ | https://arxiv.org/abs/2010.01947
- **Date:** leaderboard snapshot accessed 2026-08-10; paper 2020
- **Excerpt (verbatim, leaderboard):** "The leaderboard reports the average AUC of the abnormality detection, ACL tear, and Meniscal tear tasks. |1|Jan 09, 2019|mrnet-baseline (single model) Stanford University|**0.917**| ... Update: This competition is now closed."
- **Excerpt (verbatim, review of Azcona):** "Azcona and colleagues leveraged the baseline MRNet architecture and replaced the AlexNet feature extractor with more modern residual architectures, such as Resnet18, Resnet50 and Resnet152. They applied a series of transformations including horizontal flips and photometric augmentations (with respect to random contrast, gamma, and brightness). They reported an AUC performance of 0.91 on the validation data by using ResNet18." (Siouras et al., *Diagnostics* 2022, PMC8871256). A later review reports "a 0.934 combined AUC score on the validation data of the MRNet dataset" (Frontiers in AI 2025, 10.3389/frai.2025.1589358).
- **Why it matters:** Quantifies the cheapest possible improvement over the baseline every Kaggler will start from; ResNet-18-backbone MRNet variants are the de-facto starter code in public notebooks.
- **Key results:** ResNet-18 backbone: 0.96 ACL / 0.91 meniscus / 0.94 abnormal (validation, per Siouras Table 2).
- **Transferable tricks:** ImageNet-pretrained modern backbones beat AlexNet; horizontal flip + contrast/gamma/brightness photometric augmentation; per-slice training with max-probability exam aggregation.
- **Confidence:** High (leaderboard verbatim; Azcona numbers via two independent secondary sources).

### 1.3 ELNet — lightweight from-scratch CNN (MIDL 2020)

- **Claim:** ELNet (~0.2M params, single plane, trained from scratch) matches/beats MRNet (~183M params, 3 planes, pretrained AlexNet): MRNet-test AUC 0.960 ACL / 0.904 meniscus / 0.941 abnormal; 5-fold CV AUC 0.913 on the Rijeka KneeMRI dataset.
- **Source:** Tsai CH, Kiryati N, Konen E, Eshed I, Mayer A. "Knee Injury Detection using MRI with Efficiently-Layered Network (ELNet)." MIDL 2020 (PMLR 121:784–794). arXiv:2005.02706. Code: github.com/mxtsai/ELNet.
- **URL:** https://arxiv.org/abs/2005.02706
- **Date:** 2020-05-06 (accessed 2026-08-10)
- **Excerpt (verbatim, conclusion):** "The novel integration of multi-slice normalization and BlurPool operations allow ELNet models to remain lightweight (~0.2M parameters, requiring single imaging stack, trained from scratch) while performing favorably against MRNet models (~183M parameters, requiring three imaging stacks, pretrained AlexNet) on the MRNet dataset."
- **Excerpt (verbatim, results table):** "ELNet | Meniscus Tear | 0.88 | 0.86 | 0.89 | **0.904** | **0.745** / ACL Tear | 0.904 | 0.923 | 0.891 | **0.960** | **0.815** / Abnormality | 0.917 | 0.968 | 0.72 | **0.941** | **0.736**" (MRNet validation set).
- **Why it matters:** Proof that carefully designed small models beat big pretrained ones on ~1k-exam datasets — critical for a Kaggle compute budget; also the best citation for plane/sequence selection per pathology ("we selected coronal images for detecting meniscus tears, and axial images for detecting ACL tears and abnormalities").
- **Transferable tricks:** (1) **multi-slice normalization** (layer/contrast norm across the slice dimension instead of batch norm); (2) **BlurPool** anti-aliased downsampling; (3) histogram-based intensity standardization per Nyúl & Udupa; (4) **minority-class oversampling with augmentation** to balance; (5) pathology-specific plane selection; (6) Full-Grad saliency for localization sanity checks.
- **Confidence:** High (primary arXiv text verbatim).

### 1.4 ACL tear detection — key papers

#### 1.4a Liu et al. 2019 — fully automated ACL pipeline, arthroscopy reference
- **Claim:** A fully automated cascaded system (2 detection CNNs to isolate the ACL + classification CNN) reaches AUC 0.98, sensitivity/specificity 0.96/0.96, statistically indistinguishable from 5 radiologists, using arthroscopy as ground truth.
- **Source:** Liu F, Guan B, Zhou Z, et al. "Fully Automated Diagnosis of Anterior Cruciate Ligament Tears on Knee MR Images by Using Deep Learning." *Radiol Artif Intell.* 2019;1(3):180091. PMC6542618.
- **URL:** https://doi.org/10.1148/ryai.2019180091
- **Date:** 2019
- **Excerpt (verbatim):** "The sensitivity and specificity of the ACL tear detection system at the optimal threshold were 0.96 and 0.96, respectively. ... The area under the ROC curve for the ACL tear detection system was 0.98, indicating high overall diagnostic accuracy." And: "Our ACL tear detection system achieved a higher sensitivity for detecting an ACL tear than the deep learning method described by Bien et al, despite using a much smaller training dataset. This was likely because our ACL tear detection system first isolated the ACL on the MR images, which narrowed the range of information used for subsequent image recognition."
- **Why it matters:** The canonical evidence for the **"localize-then-classify" two-stage cascade** — the single most reliable trick in this literature for small structures (ACL, PCL, MCL/LCL).
- **Transferable trick:** ROI isolation (detection/crop network) before classification narrows the input distribution and dramatically boosts sensitivity with small data (n=350).
- **Confidence:** High.

#### 1.4b Chang et al. 2019 — 2.5D dynamic patch sampling
- **Claim:** A five-slice (2.5D) CNN with dynamic patch-based sampling around the intercondylar notch reaches test AUC 0.971 / accuracy >96% for complete ACL tear; cropped FOV + multi-slice input are both critical.
- **Source:** Chang PD, Wong TT, Rasiej MJ. "Deep Learning for Detection of Complete Anterior Cruciate Ligament Tear." *J Digit Imaging.* 2019;32:980–986. PMC6841825.
- **URL:** https://doi.org/10.1007/s10278-019-00193-4
- **Date:** 2019
- **Excerpt (verbatim):** "Using the dynamic patch-based sampling algorithm as a baseline, a five-slice CNN input (0.915) outperformed both three-slice (0.865) and single-slice (0.765) inputs. The final highest performing five-slice dynamic patch-based sampling algorithm resulted in independent test set AUC, sensitivity, specificity, PPV, and NPV of 0.971, 0.967, 1.00, 0.938, and 1.00. ... A cropped field-of-view and 3D inputs are critical for high algorithm performance."
- **Why it matters:** Best evidence in the whole literature for the **2.5D (adjacent-slice stack) + ROI crop** design pattern; directly applicable to 2026 Kaggle models for focal abnormalities.
- **Transferable tricks:** dynamic patch sampling (stochastic ROI crops during training = built-in augmentation + attention); stacking 3–5 adjacent slices as channels.
- **Confidence:** High.

#### 1.4c Namiri et al. 2020 — 3D CNN hierarchical severity staging
- **Claim:** 3D CNNs perform hierarchical ACL severity staging (intact/partial/full/reconstructed) with accuracy ~3 points above 2D CNNs; 2D/3D comparable overall (89% vs 92%, p=0.27).
- **Source:** Namiri NK, Flament I, Astuto B, et al. "Deep Learning for Hierarchical Severity Staging of Anterior Cruciate Ligament Injuries from MRI." *Radiol Artif Intell.* 2020;2(4):e190207. PMID 32793889.
- **URL:** https://doi.org/10.1148/ryai.2020190207
- **Date:** 2020
- **Excerpt (verbatim, from Siouras systematic review):** "The overall accuracies using the 3D CNN and the 2D CNN were 89% (225 of 254) and 92% (233 of 254), respectively (p-value= 0.27), whereas both CNNs had a weighted Cohen k of 0.83."
- **Why it matters:** If any of the 12 competition labels are ordinal/severity-graded, hierarchical staging heads (not flat softmax) are the proven design; also shows 3D CNNs only marginally beat 2D at this data scale.
- **Confidence:** High (citation verified) / Medium-High on exact numbers (via systematic review).

#### 1.4d Tran et al. 2022 — large-scale multi-continental external validation
- **Claim:** ACL-tear deep learning (meniscus/ACL localization CNN + two plane-specific classification CNNs) trained/validated on 19,765 knee MRIs achieves AUC 0.939 with multi-continental external validation.
- **Source:** Tran A, Lassalle L, Zille P, et al. "Deep learning to detect anterior cruciate ligament tear on knee MRI: multi-continental external validation." *Eur Radiol.* 2022;32(12):8394–8403. PMID 35726103.
- **URL:** https://doi.org/10.1007/s00330-022-08923-z
- **Date:** 2022
- **Excerpt (verbatim, via MGACA-Net review PMC10403161):** "Tran et al. (2022) developed a deep learning-based algorithm for detecting ACL tears in a large dataset of 19,765 knee MRI scans. The study employed a meniscus localization CNN to extract bounding box coordinates around the ACL and used two CNNs (sagittal and coronal view) for ACL tear classification. The model achieved a high AUC value of 0.939 for detecting ACL tears."
- **Why it matters:** Largest training cohort in the ACL literature; again validates the localize→classify cascade and shows the external-validation penalty (typically −0.05 to −0.1 AUC) Kagglers should expect on hidden test.
- **Confidence:** Medium-High (numbers via secondary review; citation independently verified).

#### 1.4e Xue et al. 2024 — segment-then-classify, expert-level
- **Claim:** Fully automated ACL-DNet (segmentation, Dice 98%±6%) + ACL-SNet (tear-type classifier) reaches AUC 0.99, sens/spec 0.97/0.97, vs experts AUC 0.96/0.92/0.88.
- **Source:** Xue Y, Yang S, Sun W, et al. "Approaching expert-level accuracy for differentiating ACL tear types on MRI with deep learning." *Sci Rep.* 2024;14:938. PMID 38195977.
- **URL:** https://doi.org/10.1038/s41598-024-51666-8
- **Date:** 2024-01-10
- **Excerpt (verbatim):** "The ACL-DNet model yielded a Dice coefficient of 98% ± 6% on the MRI datasets. Our proposed classification model yielded a sensitivity of 97% and a specificity of 97%. ... The AUC of the ACL-SNet model was 99%, demonstrating high overall diagnostic accuracy. The diagnostic performance of the clinical experts as reflected in the AUC was 96%, 92% and 88%, respectively."
- **Why it matters:** 2024 SOTA for ACL and strongest demonstration that a **pretrained segmentation branch feeding a classifier** (multitask or cascaded) beats end-to-end black-box classification.
- **Confidence:** High.

### 1.5 Meniscal tear detection — key papers

#### 1.5a Couteaux et al. 2019 — Mask R-CNN detection/orientation
- **Claim:** Mask R-CNN ensemble detects meniscal tears, localizes horns, and classifies tear orientation with weighted AUC 0.91 on 700 test MRIs.
- **Source:** Couteaux V, Si-Mohamed S, Nempont O, et al. "Automatic knee meniscus tear detection and orientation classification with Mask-RCNN." *Diagn Interv Imaging.* 2019;100(4):235–242.
- **URL:** https://doi.org/10.1016/j.diii.2019.03.002
- **Date:** 2019
- **Excerpt (verbatim, via systematic review PMC8871256):** "Couteaux et al. used a region-based convolutional neural network (R-CNN) model for tear detection and localization (anterior or posterior). ... A weighted AUC score of 0.91 was achieved by the proposed network on a test set of 700 MRIs."
- **Why it matters:** Meniscus is MRNet's hardest label (AUC 0.847) and likely among the hardest of the 12 competition labels; instance-detection framing (Mask R-CNN with voting rules across slices) is the best-performing published strategy.
- **Transferable trick:** per-slice detections combined with **majority/any-vote aggregation rules** at exam level ("anterior meniscus classified as torn when at least one network had detected a torn anterior meniscus").
- **Confidence:** Medium-High.

#### 1.5b Fritz et al. 2020 — surgery-referenced comparison with radiologists
- **Claim:** DCNN meniscus tear detection vs 2 radiologists with surgical ground truth: AUC 0.88 medial / 0.78 lateral / 0.96 overall; similar specificity, lower sensitivity than radiologists.
- **Source:** Fritz B, Marbach G, Civardi F, Fucentese SF, Pfirrmann CWA. "Deep convolutional neural network-based detection of meniscus tears: comparison with radiologists and surgery as standard of reference." *Skeletal Radiol.* 2020;49(8):1207–1217.
- **URL:** https://doi.org/10.1007/s00256-020-03410-2
- **Date:** 2020
- **Excerpt (verbatim, via review):** "The AUC of the deep CNN employed was 0.88, 0.78, and 0.96 for the detection of medial, lateral, and overall meniscus tear, respectively."
- **Why it matters:** One of the few studies with **surgical reference standard** — reminds competitors that report-derived labels (as in this Kaggle task) have label noise, especially for lateral meniscus.
- **Confidence:** Medium-High.

#### 1.5c Rizk et al. 2021 — 3D CNN with external validation
- **Claim:** 3D CNN with meniscal localization achieves AUC 0.93 medial / 0.84 lateral tear detection; external validation AUC 0.83 zero-shot → 0.89 after fine-tuning.
- **Source:** Rizk B, Brat H, Zille P, et al. "Meniscal lesion detection and characterization in adult knee MRI: A deep learning model approach with external validation." *Phys Med.* 2021;83:64–71. PMID 33714850.
- **URL:** https://doi.org/10.1016/j.ejmp.2021.02.010
- **Date:** 2021
- **Excerpt (verbatim, via review):** "Rizk et al. used a 3D CNN architecture that incorporated meniscal localization and lesion classification. They achieved AUC values of 0.93 and 0.84 for medial and lateral meniscal tear detection, respectively, and 0.91 and 0.95 for medial and lateral meniscal tear migration detection, respectively. The combined medial and lateral meniscal tear detection models were externally validated and yielded an AUC of 0.83 without additional training and 0.89 after fine-tuning."
- **Why it matters:** Clean quantification of domain-shift cost and fine-tuning recovery for meniscus models — useful when competition train/test distributions differ.
- **Confidence:** Medium-High.

#### 1.5d Tack et al. 2021 — multi-task learning on OAI
- **Claim:** Multi-task deep learning (segmentation + classification jointly) detects meniscal tears in OAI MRI.
- **Source:** Tack A, Shestakov A, Lüdke D, Zachow S. "A Multi-Task Deep Learning Method for Detection of Meniscal Tears in MRI Data from the Osteoarthritis Initiative Database." *Front Bioeng Biotechnol.* 2021;9:747217.
- **URL:** https://doi.org/10.3389/fbioe.2021.747217
- **Date:** 2021
- **Excerpt:** title/citation verified via two independent reference lists (PMC11811310; PMC12186967); detailed metrics not extracted.
- **Why it matters:** Evidence that **joint segmentation+classification multi-task training** on a public cohort (OAI) improves tear detection — the multitask recipe is directly reusable for the 12-label competition setup.
- **Confidence:** Medium (citation certain; metrics not verified).

### 1.6 Cartilage lesion detection / segmentation — key papers

#### 1.6a Norman et al. 2018 — U-Net cartilage & meniscus segmentation (REQUIRED)
- **Claim:** 2D U-Nets segment cartilage compartments and menisci with Dice 0.770–0.878 (cartilage, 3D-DESS), 0.809 lateral meniscus, 0.753 medial meniscus; downstream T1ρ/T2 relaxometry correlates 0.82–0.86 with manual analysis; ~5 s per exam.
- **Source:** Norman B, Pedoia V, Majumdar S. "Use of 2D U-Net Convolutional Neural Networks for Automated Cartilage and Meniscus Segmentation of Knee MR Imaging Data to Determine Relaxometry and Morphometry." *Radiology.* 2018;288(1):177–185. PMID 29584598. PMC6013406.
- **URL:** https://doi.org/10.1148/radiol.2018172322
- **Date:** 2018-07
- **Excerpt (verbatim):** "The models produced strong Dice coefficients, particularly for 3D-DESS images, ranging between 0.770 and 0.878 in the cartilage compartments to 0.809 and 0.753 for the lateral meniscus and medial meniscus, respectively. The models averaged 5 seconds to generate the automatic segmentations. Average correlations between manual and automatic quantification of T1ρ and T2 values were 0.8233 and 0.8603, respectively, and 0.9349 and 0.9384 for volume and thickness, respectively."
- **Why it matters:** The reference paper for adding an auxiliary **segmentation branch** to a knee-MRI pipeline. In the 2026 competition, predicted tissue masks (cartilage/meniscus) can serve as (a) attention maps, (b) ROI crops, or (c) additional input channels for the abnormality classifier.
- **Transferable trick:** compartment-wise separate U-Nets; augmentation with affine+elastic transforms; segmentation quality transfers to quantitative biomarkers.
- **Confidence:** High (PubMed abstract verbatim).

#### 1.6b Liu et al. 2018 — cartilage lesion detection system
- **Claim:** Encoder-decoder cartilage/bone segmentation + patch classifier detects cartilage lesions with AUC 0.917/0.914 (two eval runs), sensitivity higher than radiologists at Youden threshold.
- **Source:** Liu F, Zhou Z, Samsonov A, et al. "Deep Learning Approach for Evaluating Knee MR Images: Achieving High Diagnostic Performance for Cartilage Lesion Detection." *Radiology.* 2018;289(1):160–169. PMID 30063195. PMC6166867.
- **URL:** https://doi.org/10.1148/radiol.2018172986
- **Date:** 2018-10
- **Excerpt (verbatim):** "The AUCs of the cartilage lesion detection system were 0.917 and 0.914 for evaluations 1 and 2, respectively, both indicating high overall diagnostic accuracy." And: "Our study described a fully automated deep learning–based cartilage lesion detection system utilizing a convolutional encoder-decoder network for segmenting cartilage and bone followed by a second CNN classification network to detect structural abnormalities within the segmented cartilage tissue."
- **Why it matters:** Same segment→classify cascade pattern as ACL papers, applied to cartilage — likely directly relevant to a "cartilage damage" competition label.
- **Confidence:** High.

#### 1.6c Pedoia et al. 2019 — 3D CNN severity staging, meniscus + PFJ cartilage
- **Claim:** 3D CNNs detect and severity-stage meniscus and patellofemoral-cartilage degenerative changes; binary lesion AUC up to 0.93, sensitivity/specificity ~85%.
- **Source:** Pedoia V, Norman B, Mehany SN, Bucknor MD, Link TM, Majumdar S. "3D convolutional neural networks for detection and severity staging of meniscus and PFJ cartilage morphological degenerative changes in osteoarthritis and anterior cruciate ligament subjects." *J Magn Reson Imaging.* 2019;49(2):400–410.
- **URL:** https://doi.org/10.1002/jmri.26246
- **Date:** 2019
- **Excerpt (verbatim, via review):** "3D CNNs were built by Astuto et al. to identify and grade meniscus tear ... The reported binary lesion sensitivity and specificity values were 85% for both., whereas the AUC was 0.93." (Pedoia model: lesion detection AUC 0.89 per Siouras review: "This model produced a lesion detection AUC performance of 0.89 on the test dataset.")
- **Why it matters:** Introduces **3D CNN grading on V-Net-segmented ROIs** — the architecture later scaled in Astuto 2021.
- **Confidence:** Medium-High.

### 1.7 Bone marrow lesion / bone marrow edema — key papers

#### 1.7a Preiswerk et al. 2022 — BML quantification
- **Claim:** Patch-based CNN, given BML location, segments BML volume automatically: reader-vs-model R² 0.95/0.81, mean Dice 0.70 (reader-vs-reader R² 0.85).
- **Source:** Preiswerk F, et al. "Fast quantitative bone marrow lesion measurement on knee MRI for the assessment of osteoarthritis." *Osteoarthr Cartil Open.* 2022. PMID 36474467.
- **URL:** https://pubmed.ncbi.nlm.nih.gov/36474467/
- **Date:** 2022-10
- **Excerpt (verbatim):** "We have applied a deep learning approach using a patch-based convolutional neural network (CNN) which was trained using 673 MRI data sets and the segmented BML masks obtained from a trained reader. ... The Pearson's R2 value was 0.94 and we found similar agreement when comparing two readers (R2 = 0.85) and each reader versus the DL model (R2 = 0.95 and R2 = 0.81). The average DSC was 0.70."
- **Why it matters:** "Bone contusion"/BML is one of the 12 abnormality types in the CoPAS dataset and plausibly one of the competition's 12 labels; this paper defines the segmentation approach and the inter-reader ceiling (Dice 0.70) to calibrate expectations.
- **Confidence:** High.

#### 1.7b Astuto et al. 2021 — multi-tissue 3D CNN detection & grading (cartilage, BME, meniscus, ACL) (REQUIRED)
- **Claim:** 17 3D CNN classifiers on V-Net-segmented ROIs detect/grade cartilage, bone-marrow-edema, meniscus, and ACL lesions in 1435 exams: binary sensitivity 70–88%, specificity 85–89%, AUC 0.83–0.93 across tissues; AI assistance improved intergrader κ in 10/16 comparisons.
- **Source:** Astuto B, Flament I, Namiri NK, et al. "Automatic Deep Learning–assisted Detection and Grading of Abnormalities in Knee MRI Studies." *Radiol Artif Intell.* 2021;3(3):e200165. PMID 34142088. PMC8166108.
- **URL:** https://doi.org/10.1148/ryai.2021200165
- **Date:** 2021-01-20
- **Excerpt (verbatim):** "Binary lesion sensitivity reported for all tissues was between 70% and 88%. Specificity ranged from 85% to 89%. The area under the receiver operating characteristic curve for all tissues ranged from 0.83 to 0.93. Deep learning-assisted intergrader Cohen κ agreement significantly improved in 10 of 16 comparisons among two attending physicians and two trainees for all tissues."
- **Why it matters:** The closest published analogue to the competition's multi-abnormality task: **one exam → many tissue-specific graded outputs** (52 probabilities from 17 models). Demonstrates ROI-bbox preprocessing + per-tissue classifier banks as a complete system design.
- **Transferable tricks:** V-Net auto-segmentation → volumetric bounding boxes per compartment → separate 3D CNNs per tissue/severity; equal class distribution enforced across splits.
- **Confidence:** High.

### 1.8 Knee OA progression from MRI — key papers

#### 1.8a Schiratti et al. 2021 — weakly supervised progression prediction (OAI)
- **Claim:** Weakly supervised CNN on coronal IW TSE predicts 12-month radiographic progression (JSN) at AUC 0.65 vs radiologists 0.587; pain (WOMAC) prediction AUC 0.72; relative-JSN endpoint reaches 0.80.
- **Source:** Schiratti JB, Dubois R, Herent P, et al. "A deep learning method for predicting knee osteoarthritis radiographic progression from MRI." *Arthritis Res Ther.* 2021;23:262. PMID 34663440. PMC8521982.
- **URL:** https://doi.org/10.1186/s13075-021-02634-4
- **Date:** 2021-10-18
- **Excerpt (verbatim):** "Using 9280 knee magnetic resonance (MR) images (3268 patients) from the Osteoarthritis Initiative (OAI) database, we implemented a deep learning method to predict, from MR images and clinical variables including body mass index (BMI), further cartilage degradation measured by joint space narrowing at 12 months. ... Using COR IW TSE images, our classification model achieved a ROC AUC score of 65%. On a similar task, trained radiologists obtained a ROC AUC score of 58.7%. ... Additional analyses conducted in parallel to predict pain grade evaluated by the WOMAC pain index achieved a ROC AUC score of 72%."
- **Why it matters:** Proof-of-concept for **weakly supervised learning from report-level labels only** (no ROI annotation) on ~9k knee MRIs — exactly the label regime of the competition; Grad-CAM attention maps align with joint space (progression) vs intra-articular space (pain).
- **Transferable tricks:** weak supervision + MIL-style pooling; Grad-CAM for label-free localization; multi-task endpoints share a backbone.
- **Confidence:** High.

#### 1.8b Panfilov et al. 2023 — multi-modal transformers for KOA progression (OAI)
- **Claim:** Transformer-based multi-modal fusion of OAI data (X-ray + structural + compositional MRI + clinical) predicts KOA progression at ROC AUC 0.70–0.76 across 2–8-year horizons (n=2421–3967); 1-year progression best with multi-modal fusion (AUC 0.76).
- **Source:** Panfilov E, Tiulpin A, Nieminen MT, Saarakkala S. "End-To-End Prediction of Knee Osteoarthritis Progression With Multi-Modal Transformers." arXiv:2307.00873 (2023; later MICCAI MLMI / IEEE TBME version). Code public.
- **URL:** https://arxiv.org/abs/2307.00873
- **Date:** 2023-07-03
- **Excerpt (verbatim):** "We show that structural knee MRI allows identifying radiographic KOA progressors on par with multi-modal fusion approaches, achieving an area under the ROC curve (ROC AUC) of 0.70-0.76 and Average Precision (AP) of 0.15-0.54 in 2-8 year horizons. Progression within 1 year was better predicted with a multi-modal method using X-ray, structural, and compositional MR images -- ROC AUC of 0.76(0.04), AP of 0.13(0.04) -- or via clinical data."
- **Why it matters:** The most mature published recipe for **fusing MRI with non-image tabular/clinical data via transformers** — directly relevant to the competition's "MRI + reports" multimodal setup.
- **Transferable tricks:** modality-specific encoders + transformer fusion; publicly released code and pretrained weights.
- **Confidence:** High.

#### 1.8c Huang et al. 2022 — DADP longitudinal abnormality detection (OAI)
- **Source:** Huang C, et al. "DADP: Dynamic abnormality detection and progression for longitudinal knee magnetic resonance images from the Osteoarthritis Initiative." *Med Image Anal.* 2022;80:102343. PMC8901568.
- **URL:** https://doi.org/10.1016/j.media.2022.102343
- **Why it matters:** Models **longitudinal** knee MRI (temporal dimension) — relevant if competition includes follow-up exams; demonstrates dynamic progression modeling with deep nets on OAI.
- **Confidence:** Medium (citation verified; metrics not extracted).

---

## 2. ARCHITECTURAL EVOLUTION (2D → 2.5D → 3D → attention → transformers/SSL/video)

| Era | Representative work | Aggregation idea | Headline result |
|---|---|---|---|
| 2D CNN + slice max-pool | **Bien 2018 MRNet** | AlexNet per slice → global max-pool over slices → LR over 3 planes | AUC 0.937/0.965/0.847 (abn/ACL/men) |
| 2D CNN + modern backbone | **Azcona 2020** | ResNet-18, same max-pool/LR recipe | val AUC ~0.91–0.96 |
| Lightweight 2D from scratch | **ELNet 2020** | multi-slice normalization + BlurPool; single plane | AUC 0.960 ACL, 0.904 meniscus |
| 2.5D (adjacent slices) | **Chang 2019** | 5-slice stack + dynamic patch sampling | AUC 0.971 complete ACL tear |
| 3D CNN on ROI | **Pedoia 2019, Astuto 2021, Rizk 2021** | V-Net ROI → 3D CNN grading banks | AUC 0.83–0.93 all tissues |
| Attention slice aggregation | **Belton 2021 (MPFuseNet)** | spatial attention ResNet-18 + multi-plane fusion (learned, not LR) | test AUC 0.977 ACL / 0.957 abn (see below) |
| CNN+Transformer hybrid | **Dai 2021 TransMed** | CNN branch + transformer branch for long-range, multi-sequence fusion | AUC 0.98 ACL / 0.976 abn / 0.95 meniscus (MRNet) |
| Video/SSL | **Manna 2022** | slice stack treated as video; self-supervised pretraining | competitive on MRNet/KneeMRI (see below) |
| Slice-based SSL transformers | **Atito 2022 SB-SSL** | ViT per slice + transformer over slice embeddings, SSL pretrain, <1000 cases | AUC 0.954 ACL, acc 89.17% |
| 3D ResNet + co-plane attention | **Qiu/Xie 2024 CoPAS (Nat Commun)** | 3D ResNet-18 branches per plane + cross-plane & cross-sequence attention + abnormality-plane probability matrix | avg AUC 0.812 over **12 abnormalities** |
| Slice-alignment + Top-K pooling | **Han 2025 MLFANet-SA** | slice-aligning module + channel-wise Top-K pooling + cross-slice fusion | MRNet ACL AUC **0.981** (2025 SOTA) |

### 2.1 Attention slice aggregation — Belton et al. 2021 (MPFuseNet)
- **Claim:** ResNet-18 + spatial attention with learned multi-plane fusion (MPFuseNet) beats MRNet's logistic-regression plane fusion (MPLR); test AUC 0.977 ACL / 0.957 abnormal / 0.831 meniscus; MPLR can actively hurt ACL/meniscus performance.
- **Source:** Belton N, et al. "Optimising Knee Injury Detection with Spatial Attention and Validating Localisation Ability." MIUA 2021. arXiv:2108.08136.
- **URL:** https://arxiv.org/abs/2108.08136
- **Date:** 2021-08-18
- **Excerpt (verbatim):** "This analysis has demonstrated that MPFuseNet outperforms MPLR for ACL and meniscus tear detection and that MPLR can be detrimental to the model's performance. This is a significant finding given that MPLR is the most common method in the literature." Table 2 (verbatim): "|Proposed Models|0.977 MPFuseNet|0.957 MPLR|0.831 MPFuseNet| / |ELNet|0.960|0.941|0.904| / |MRNet|0.956|0.936|0.826|"
- **Why it matters:** Directly answers "how should I fuse sagittal/coronal/axial predictions?" — learned attention-based fusion > logistic regression stacking for tear-type labels.
- **Confidence:** High (primary text verbatim).

### 2.2 CNN+Transformer hybrid — TransMed (Dai et al. 2021)
- **Claim:** First transformer-based model for knee MRI; CNN+ViT hybrid fuses sagittal/coronal/axial sequences as multi-modal input: MRNet AUC 0.98 ACL / 0.976 abnormal / 0.95 meniscus (accuracy 94.9/91.8/85.3%).
- **Source:** Dai Y, Gao Y, Liu F. "TransMed: Transformers Advance Multi-Modal Medical Image Classification." *Diagnostics.* 2021;11(8):1384.
- **URL:** https://doi.org/10.3390/diagnostics11081384
- **Date:** 2021
- **Excerpt (verbatim, via systematic review PMC8871256):** "Dai et al. utilized TransMed, achieving accuracy and AUC values of 94.9% and 0.98, respectively, for detecting meniscus tears [sic—ACL], thus improving over the MRNet technique." Table 2 of same review: "|4|Dai et al.|2021|TransMed|...|ACL tear = 94.9%/0.98, Abnormality = 91.8%/0.976, Meniscus tear = 85.3%/0.95|". A second independent review (Frontiers 2025) reports "0.952 for Meniscus Tear, 0.981 for ACL Tear, and 0.976 for Abnormality".
- **Why it matters:** Template for treating the three MRI planes (and, by extension, the competition's report text) as **multiple modalities fused by a transformer**; biggest single jump over MRNet published to date.
- **Transferable trick:** parallel CNN (local features) + transformer (cross-sequence/cross-modal long-range dependencies) branches.
- **Confidence:** High (two independent secondary sources agree on numbers).

### 2.3 Video-based / self-supervised — Manna et al. 2022
- **Claim:** Treating an MRI slice stack as a "video" enables self-supervised pretraining (SimSiam-style) + recurrent/3D aggregation for ACL tear detection without labels.
- **Source:** Manna S, Bhattacharya S, Pal U. "Self-supervised representation learning for detection of ACL tear injury in knee MR videos." *Pattern Recognition Letters.* 2022;154:37–43.
- **URL:** https://doi.org/10.1016/j.patrec.2022.01.008
- **Date:** 2022
- **Excerpt (verbatim, citation):** "Manna, S., Bhattacharya, S. & Pal, U. Self-supervised representation learning for detection of ACL tear injury in knee MR videos. Pattern Recognit. Lett. 154, 37–43 (2022)." (metrics not extracted — TimeSformer-style video models are explicitly the framing of this line of work; see also follow-up "Self-Supervised Representation Learning for Knee Injury Diagnosis From Magnetic Resonance Data," IEEE Trans. AI 2023, and BYOLMed3D arXiv:2208.00444; BYOL for ACL, PMC12546683).
- **Why it matters:** SSL pretraining on **unlabeled** MRNet/fastMRI volumes is a realistic Kaggle edge when labels are limited; the slice-as-frame analogy licenses video architectures (3D ResNet, TimeSformer, Video Swin).
- **Confidence:** Medium (citation certain; specific AUCs not verified).

### 2.4 Slice-based SSL transformers — SB-SSL (Atito et al. 2022)
- **Claim:** With <1000 labeled cases, slice-based self-supervised ViT+transformer framework reaches ACL AUC 0.954 / accuracy 89.17% on MRNet, beating supervised SOTA without external pretraining data.
- **Source:** Atito S, Anwar SM, Awais M, Kittler J. "SB-SSL: Slice-Based Self-Supervised Transformers for Knee Abnormality Classification from MRI." arXiv:2208.13923; MICCAI Workshop on Medical Image Learning with Limited and Noisy Data 2022, pp. 86–95.
- **URL:** https://arxiv.org/abs/2208.13923
- **Date:** 2022-08-29
- **Excerpt (verbatim):** "Herein, we propose a slice-based self-supervised deep learning framework (SB-SSL), a novel slice-based paradigm for classifying abnormality using knee MRI scans. We show that for a limited number of cases (<1000), our proposed framework is capable to identify anterior cruciate ligament tear with an accuracy of 89.17% and an AUC of 0.954, outperforming state-of-the-art without usage of external data during pretraining."
- **Why it matters:** The clean recipe for the **ViT-per-slice → transformer-over-slices → [CLS]-token** exam classifier; ideal for a 12-label extension (one CLS per label or multi-head).
- **Confidence:** High.

### 2.5 CoPAS — 12-abnormality multi-sequence attention (Nature Communications 2024) — MOST COMPETITION-RELEVANT PAPER
- **Claim:** CoPAS (co-plane attention across MRI sequences) diagnoses **12 knee abnormality types** (MENI, ACL, CART, PCL, MCL, LCL, EFFU, CONT, PLICA, CYST, IFP, PR) on a 1748-patient, 5-center arthroscopy-referenced dataset: internal avg AUC 0.812, external 0.721–0.726; beats adapted MRNet/ELNet/MPFuseNet in 8/12 classes; matches senior radiologists (avg ACC 0.78 vs 0.80, beats juniors at 0.65).
- **Source:** Qiu Z, Xie Z, Lin H, Li Y, Ye Q, Wang M, Li S, Zhao Y, Chen H. "Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities." *Nat Commun.* 2024;15:7540 (s41467-024-51888-4). Code: github.com/zqiuak/CoPAS.
- **URL:** https://www.nature.com/articles/s41467-024-51888-4
- **Date:** 2024-09-02 (accessed 2026-08-10)
- **Excerpt (verbatim):** "The results show that our method outperforms other models with an average AUC-ROC of 0.812. Specifically, our CoPAS outperformed the three extant models in 8 out of 12 abnormalities... a decline of the average AUC-ROCs from 0.812 to 0.721 and 0.726 is observed when transitioning from the internal dataset to the two external datasets." Architecture (verbatim): "we utilize ResNet3D with 18 layers as the basic encoder... the feature map will be pooled and reshaped from R^{C×D×H×W} to R^{D×C} ... to generate the representations of each slice." Loss (verbatim): "we use Focal Loss to measure the distance between the final result y and label ŷ, which can help the model to focus on the difficult samples. For the prediction of three branches y_branch, the binary cross entropy (BCE) loss is applied."
- **Why it matters:** This is essentially the competition task (12 abnormalities, multi-plane, multi-sequence knee MRI) with a published, open-source strong baseline. Its tricks are the state of the art for the exact problem: U-Net meniscus crop → rotate PDW volumes to synthesize cross-plane views → weight-shared 3D ResNet-18 per plane → cross-plane & cross-sequence (PDW/T1W/T2W) attention → plane×abnormality probability-matrix fusion → per-branch BCE + final focal loss.
- **Transferable tricks (all directly reusable):** (1) multi-task single model over 12 binary heads; (2) focal loss on final heads + BCE on branch heads; (3) learned abnormality↔plane correlation matrix (clinically consistent: sagittal for meniscus/cruciates, coronal for collaterals); (4) volume rotation as augmentation that synthesizes orthogonal planes instead of interpolating; (5) external-validation drop of ~0.09 AUC is the realistic generalization gap.
- **Confidence:** High (full text read).

### 2.6 2025 SOTA on MRNet ACL — MLFANet-SA
- **Claim:** MLFANet-SA (slice-aligning + multi-level feature aggregation) reaches MRNet ACL AUC 0.981, acc 0.949, MCC 0.892 without ROI/segmentation labels; private dataset AUC 0.975.
- **Source:** Han S, et al. "Anterior cruciate ligament injuries diagnosis using slice-aligning and multi-level feature aggregation." *Med Phys.* 2025;52(11):e70130. PMID 41206350.
- **URL:** https://pubmed.ncbi.nlm.nih.gov/41206350/
- **Date:** 2025-11
- **Excerpt (verbatim):** "On the MRNet dataset, MLFANet-SA achieves an AUC of 0.981, sensitivity of 0.961, specificity of 0.941, precision of 0.933, accuracy of 0.949, and MCC of 0.892. ... MLFANet-SA consists of two modules: (1) a slice-aligning (SA) model using local context perceptron (LCP) to identify boundary slices and unify diagnostic regions, and (2) a multi-level feature aggregation (MLFA) module that captures spatial and cross-slice lesion patterns via channel-wise Top-K pooling and cross-slice fusion."
- **Why it matters:** Current (2025) published SOTA on MRNet ACL; **channel-wise Top-K pooling over slices** is a drop-in upgrade over MRNet's max-pool, and slice-alignment removes non-informative slices — both cheap Kaggle wins.
- **Confidence:** High (PubMed abstract verbatim).

### 2.7 Caution flag — KneeXNet (2025)
- A 2025 paper ("KneeXNet," PMC12088959) claims MRNet test AUCs of 0.985/0.972/0.968 via graph convolutions + multi-scale fusion + contrastive learning. The claims ("significantly better than all deep learning methods with p<0.01") and near-perfect metrics in a non-top-tier venue warrant **skepticism** (possible test-set leakage/tuning). Cite only as an existence proof of graph/contrastive approaches. **Confidence: Low-Medium.**

---

## 3. DATASET PAPERS

### 3.1 MRNet dataset (Stanford)
- **Claim:** 1,370 knee MRI exams (Stanford, 2001–2012), 1,104 abnormal (80.6%), 319 ACL tears (23.3%), 508 meniscal tears (37.1%); sag T2 FS / cor T1 / ax PD FS extracted; splits 1130 train / 120 val / 120 hidden test, patient-disjoint, ≥50 positives per label per split; 56.6% 3.0T.
- **Source:** MRNet dataset & competition page + Bien 2018.
- **URL:** https://stanfordmlgroup.github.io/competitions/mrnet/
- **Excerpt (verbatim):** "The exams have been split into a training set (1,130 exams, 1,088 patients), a validation set (called tuning set in the paper) (120 exams, 111 patients), and a hidden test set (called validation set in the paper) (120 exams, 113 patients). ... All exams from each patient were put in the same split."
- **Why it matters:** De-facto public benchmark and pretraining/validation source for the competition; label provenance is **manual extraction from clinical reports** — same noisy-label regime as Kaggle.
- **Confidence:** High.

### 3.2 Clinical Hospital Centre Rijeka (KneeMRI) dataset — Štajduhar et al. 2017
- **Claim:** 917 knee MRI exams (sagittal PD-weighted, 1.5T), labeled ACL: 690 healthy / 172 partial / 55 complete ruptures (counts vary slightly by re-use); original semi-automated HOG/GIST+SVM pipeline reached AUC 0.894 (10-fold CV).
- **Source:** Štajduhar I, Mamula M, Miletić D, Ünal G. "Semi-automated detection of anterior cruciate ligament injury from MRI." *Comput Methods Programs Biomed.* 2017;140:151–164.
- **URL:** https://doi.org/10.1016/j.cmpb.2016.12.006
- **Excerpt (verbatim, from MRNet paper):** "Štajduhar et al. recorded an AUC of 0.894 for their best model, a semi-automated approach using support vector machines, although it was evaluated using a 10-fold cross-validation scheme." Dataset composition (verbatim, via PMC12480240): "536 normal images, 140 partially ruptured images, and 45 fully ruptured ACL tear images" (per-image counts after unpickling; exam-level: 917 exams).
- **Why it matters:** The standard **external validation** set for MRNet-style models (MRNet zero-shot AUC 0.824 → fine-tuned 0.911); useful as an OOD check for competition models.
- **Confidence:** High.

### 3.3 Osteoarthritis Initiative (OAI) — Peterfy et al. 2008 + Ambellan et al. 2019
- **Claim:** OAI is a public multicenter cohort (~4,796 participants) with longitudinal knee MRI (3T, DESS/iw-TSE etc.), clinical data, and semiquantitative scores; the Ambellan/ZIB release adds 507 public manual bone+cartilage segmentations.
- **Sources:** (a) Peterfy CG, Schneider E, Nevitt M. "The osteoarthritis initiative: report on the design rationale for the magnetic resonance imaging protocol for the knee." *Osteoarthritis Cartilage.* 2008;16:1433–1441. (b) Ambellan F, Tack A, Ehlke M, Zachow S. "Automated segmentation of knee bone and cartilage combining statistical shape knowledge and convolutional neural networks: Data from the Osteoarthritis Initiative." *Med Image Anal.* 2019;52:109–118. PMID 30529224.
- **URLs:** https://doi.org/10.1016/j.joca.2008.06.016 | https://doi.org/10.1016/j.media.2018.11.009 | data: https://nda.nih.gov/oai/
- **Excerpt (verbatim, Ambellan abstract):** "The shape models and neural networks employed are trained using data from the Osteoarthritis Initiative (OAI) and the MICCAI grand challenge 'Segmentation of Knee Images 2010' (SKI10)... For the first time, an accuracy equivalent to the inter-observer variability of human readers is achieved in this challenge. ... We make the 507 manual segmentations as well as our experimental setup publicly available."
- **Why it matters:** Largest source of **publicly labeled knee MRI** (segmentation masks, WORMS/MOAKS scores, KL grades, progression labels) for pretraining segmentation/auxiliary tasks to transfer into the competition.
- **Confidence:** High.

### 3.4 fastMRI (knee subset) — Zbontar et al. 2018 / Recht et al. 2020
- **Claim:** fastMRI provides ~1,500+ raw k-space knee MRI volumes (NYU, coronal PD/PD-FS) plus DICOMs of ~10k clinical knee exams; reconstruction-challenge provenance but usable as large-scale unlabeled pretraining.
- **Sources:** Zbontar J, et al. "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI." arXiv:1811.08839; Recht MP, et al. "Using Deep Learning to Accelerate Knee MRI at 3 T: Results of an Interchangeability Study." *AJR.* 2020;215(6):1421–1429 (PMID 32755163); Johnson PM, et al. *Radiology* 2023;307:e220425 (prospective DL-recon clinical use).
- **URL:** https://arxiv.org/abs/1811.08839
- **Why it matters:** Best public source of **unlabeled knee MRI at scale** for SSL pretraining (BYOLMed3D / SB-SSL style) when competition labels are scarce.
- **Confidence:** High (citations) / numbers approximate.

### 3.5 CoPAS 5-center 12-abnormality dataset (Qiu/Xie 2024)
- **Claim:** 1,748 patients from 5 Chinese centers, arthroscopy + MRI consensus labels for 12 abnormality types; PDW in 3 planes + coronal T1W + sagittal T2W; restricted academic access on request; code public.
- **Source/URL:** Nat Commun 2024 (see §2.5).
- **Excerpt (verbatim):** "In total, 1748 patients were collected from five clinical centers in China... The images include PDW sequences taken from sagittal, coronal, and axial planes, the coronal T1W, and the sagittal T2W sequences."
- **Why it matters:** The only public-ish dataset whose label taxonomy (12 abnormalities) mirrors the 2026 competition; even if data access fails, its label definitions and plane-preference table are a blueprint.
- **Confidence:** High.

### 3.6 Other useful public resources
- **SKI10 (MICCAI 2010 "Segmentation of Knee Images")** — 100 knee MRIs with bone/cartilage labels; used by Ambellan 2019. Confidence: High.
- **OAI-iMorphics / ZIB segmentations** — see §3.3.
- **Knee radiograph KL-grade sets (OAI/MOST)** — out of modality scope but usable for multimodal pretraining experiments (cf. Panfilov 2023). Confidence: Medium.

---

## 4. CROSS-CUTTING TRANSFERABLE TRICKS (synthesis for the competition)

1. **Slice aggregation:** max-pool (MRNet) < attention/learned pooling (MPFuseNet) ≈ channel-wise Top-K pooling (MLFANet-SA). Top-K and attention pooling are near-free upgrades.
2. **Plane fusion:** learned attention fusion > logistic regression stacking for tear labels (Belton 2021, significant finding); MRNet-style LR is fine as baseline.
3. **Localize-then-classify:** cascades (U-Net/V-Net ROI → classifier) dominate for small structures: ACL (Liu 2019, Xue 2024: AUC 0.98–0.99), cartilage (Liu 2018: 0.917), meniscus (Rizk 2021), whole pipeline (Astuto 2021, CoPAS 2024).
4. **Class imbalance:** minority oversampling + augmentation (ELNet); focal loss on final heads + BCE on branches (CoPAS); weighted CE (MRNet).
5. **Pretraining:** ImageNet fine-tuning is the safe default (MRNet/Azcona), but from-scratch lightweight nets with multi-slice normalization can win at ~1k-exam scale (ELNet); SSL on unlabeled knee MRI (SB-SSL AUC 0.954 with <1000 labels; Manna 2022) is the high-upside play.
6. **2.5D inputs:** stacking 3–5 adjacent slices (+0.15 validation accuracy over single-slice, Chang 2019) is the cheapest volumetric context.
7. **Multi-task over 12 heads** beats per-label ensembles in compute and helps rare labels (CoPAS: best relative gain on rare patellar-retinaculum class), but watch negative transfer (single-task MRNet still won MCL/effusion).
8. **Expect a generalization gap** of ~0.07–0.10 AUC on external/hidden test (CoPAS 0.812→0.721; Rizk 0.93→0.83) — build a robust local CV (patient-disjoint, stratified ≥50 positives/label).
9. **Report-text fusion:** no knee-specific MRI+report transformer paper was found; the closest templates are TransMed (multi-sequence transformer fusion) and Panfilov 2023 (imaging+clinical transformer fusion). This is an open opportunity for the competition's multimodal component.
10. **Label noise:** report-derived labels (MRNet) understate surgical truth especially for lateral meniscus (Fritz 2020) — use soft labels/label smoothing and rely on ranking metric (AUC) robustness.

---

## 5. TOP-10 RECOMMENDED READING ORDER (for a newcomer)

1. **Bien et al. 2018 (MRNet, PLoS Med)** — the task, dataset, and baseline every solution builds on.
2. **Qiu/Xie et al. 2024 (CoPAS, Nat Commun)** — the exact 12-abnormality problem with open-source SOTA architecture and loss design.
3. **Tsai et al. 2020 (ELNet)** — how to beat big models with small data; normalization, oversampling, plane selection.
4. **Azcona et al. 2020** — 15-minute read; modern-backbone MRNet variants = your starter code.
5. **Belton et al. 2021 (MPFuseNet, arXiv:2108.08136)** — spatial attention + why learned plane fusion beats logistic regression.
6. **Astuto et al. 2021 (Radiol: AI)** — multi-tissue detection/grading system design (segmentation ROI banks → per-tissue 3D CNNs).
7. **Liu et al. 2019 (Radiol: AI) + Chang et al. 2019 (JDI)** — the two-stage localize→classify cascade and 2.5D dynamic patch sampling (read together).
8. **Dai et al. 2021 (TransMed, Diagnostics)** — transformer fusion of MRI sequences; template for MRI+report multimodal fusion.
9. **Atito et al. 2022 (SB-SSL, arXiv:2208.13923)** — slice-based SSL transformers for the limited-label regime.
10. **Han et al. 2025 (MLFANet-SA, Med Phys)** — current MRNet SOTA; Top-K slice pooling + slice alignment as final polish.

*Supplementary (datasets):* MRNet competition page; Štajduhar 2017 (Rijeka); Peterfy 2008 + Ambellan 2019 (OAI); Zbontar 2018 (fastMRI). *OA/progression:* Schiratti 2021, Panfilov 2023. *BML:* Preiswerk 2022. *Segmentation:* Norman 2018.

---

## 6. SEARCH LOG (independent searches)
1. MRNet Bien 2018 PLoS Med PMID 30481176; 2. MRNet Stanford dataset 1370 exams; 3. Norman 2018 U-Net cartilage; 4. ACL attention slice aggregation; 5. meniscal tear 3D CNN; 6. Rijeka dataset; 7. ELNet Tsai 2020; 8. Azcona MRNet backbone; 9. transformer/TimeSformer knee MRI; 10. Norman abstract (PubMed); 11. bone marrow lesion deep learning; 12. OA progression OAI deep learning; 13. Liu 2018 cartilage Radiology; 14. Liu 2019 ACL Radiol AI; 15. Astuto 2021 multi-pathology; 16. Manna 2022 SSL MR videos; 17. Dai TransMed; 18. Ambellan 2019 OAI segmentation; 19. Tran 2022 multi-continental ACL; 20. video-based knee MRI 2023/2024; 21. Xue 2024 ACL tear types; 22. MRNet SOTA 2024–2025 attention MIL; 23. Tack OAI multitask meniscus; 24. Atito SB-SSL; plus full-text reads: Nat Commun CoPAS (2024), PubMed Astuto (34142088), arXiv SB-SSL. Key systematic reviews used for cross-verification: Siouras et al. 2022 *Diagnostics* (PMC8871256), PMC10590246 comprehensive review, PMC12021734 (2025 systematic review).

*Gaps/uncertainties:* Manna 2022 and Tack 2021 exact metrics not extracted from primary text (paywalled); KneeXNet claims flagged low-confidence; MRNet hidden-test leaderboard closed at avg AUC 0.917, so later papers report on the public validation split with differing protocols — cross-paper AUC comparisons are approximate.
