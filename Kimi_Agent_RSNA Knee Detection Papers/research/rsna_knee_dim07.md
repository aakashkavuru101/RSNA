# Dimension 07 — Technical Modeling Methods for Volumetric MRI Classification
## RSNA Knee Abnormality Detection (2026): 12-label knee MRI exam classification (multi-plane, multi-sequence, variable slices) + report text
Research date: 2026-08-10 | Sub-agent: dim07 | Searches performed: 24 independent queries (arXiv/PMC/GitHub/Nature priority)

---

## 1. Volumetric classification approaches

### 1a. 2D CNN per-slice + pooling (MRNet style) — the reference baseline

**Claim:** MRNet (Stanford, Bien et al. 2018) runs each 2D slice through an AlexNet feature extractor, global-average-pools spatially, then applies **max pooling across slices** followed by a FC+sigmoid head; one model per (task × plane) = 9 models, combined per exam with logistic regression. Internal AUCs: 0.937 abnormality / 0.965 ACL / 0.847 meniscus. External (Croatia, different scanner/sequence) dropped to 0.824 without retraining, recovering to 0.911 after retraining.
**Source:** Bien et al., "Deep-learning-assisted diagnosis for knee MRI: Development and retrospective validation of MRNet", PLoS Medicine
**URL:** https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699
**Date:** 2018-11-27
**Excerpt (verbatim):** "The input to MRNet has dimensions s × 3 × 256 × 256, where s is the number of images in the MRI series (3 is the number of color channels). First, each 2-dimensional MRI image slice was passed through a feature extractor based on AlexNet to obtain a s × 256 × 7 × 7 tensor containing features for each slice. A global average pooling layer was then applied to reduce these features to s × 256. We then applied max pooling across slices to obtain a 256-dimensional vector, which was passed to a fully connected layer and sigmoid activation function to obtain a prediction in the 0 to 1 range." And: "We trained a different MRNet for each task (abnormality, anterior cruciate ligament [ACL] tear, meniscal tear) and series type (sagittal, coronal, axial), resulting in 9 different MRNets." And on external validation: "an MRNet trained on the rest of the external data achieved an AUC of 0.911 (95% CI 0.864, 0.958)" vs 0.824 with no retraining.
**Confidence:** High (primary paper).

**Claim:** On MRNet data, CNN transfer learning (ResNet50) beats ViT and InceptionV3 in the small-data regime; per-slice CNN + aggregation remains a very strong baseline.
**Source:** Yiu et al., "A Systematic Study of Deep Learning Models and xAI Methods for ROI Detection in MRI Scans", arXiv:2508.14151
**URL:** https://arxiv.org/html/2508.14151v1
**Date:** 2025-08-19
**Excerpt (verbatim):** "ResNet50 achieved the strongest performance, with an AUC of 0.8184 and an accuracy of 0.74, outperforming both InceptionV3 (AUC = 0.72, Accuracy = 0.66) and ViT (AUC = 0.74, Accuracy = 0.67). ResNet's residual connections and pretrained initialization allowed it to effectively capture deep hierarchical features while generalizing well to limited medical data."
**Confidence:** Medium-High (student systematic study; directionally consistent with literature).

**Pros (2D+pooling):** ImageNet-pretrained backbones usable; handles variable slice counts natively (pooling is cardinality-invariant); cheap; strong small-data regime; per-slice CAM interpretability. **Cons:** no explicit inter-slice context; max pooling can be noisy/unstable (see MIL critique below); one-model-per-plane is parameter/compute heavy at 12 labels × planes.

### 1b. ELNet: lightweight per-slice CNN with multi-slice normalization (key efficiency reference)

**Claim:** ELNet (~0.2M params, trained from scratch, single plane) matches/beats MRNet (~183M params, pretrained AlexNet, 3 planes) on MRNet data: meniscus AUC 0.904 vs 0.826, ACL 0.960 vs 0.956, abnormal 0.941 vs 0.936. Key ingredients: **multi-slice normalization** (slice-wise layer/contrast norm instead of batch norm — batch norm caused divergence) and **BlurPool** anti-aliased downsampling instead of max pooling.
**Source:** Tsai et al., "Knee Injury Detection using MRI with Efficiently-Layered Network (ELNet)", MIDL 2020 / arXiv:2005.02706; official code github.com/mxtsai/ELNet
**URL:** https://arxiv.org/pdf/2005.02706
**Date:** 2020-05-06
**Excerpt (verbatim):** "The novel integration of multi-slice normalization and BlurPool operations allow ELNet models to remain lightweight (~ 0.2M parameters, requiring single imaging stack, trained from scratch) while performing favorably against MRNet models (~ 183M parameters, requiring three imaging stacks, pretrained AlexNet) on the MRNet dataset." Ablation: "it is evident that the use of batch normalization aggravates ELNet performance. In practice, we observe network divergence during training after 10-15 epochs ... when BlurPool is paired with the intended multi-slice normalization, we observe an overall improvement in performance compared to max-pooling." (Multi-slice norm + BlurPool: 0.904/0.960/0.941 AUC vs Batch norm + MaxPool: 0.797/0.906/0.880.) Also: "we perform histogram-based intensity standardization according to the training set statistics, thus enabling similar-valued pixels to be associated with the relevant tissue type" (Nyúl standardization) and oversampling for class imbalance.
**Confidence:** High.

### 1c. 2.5D (adjacent slices as channels)

**Claim:** 2.5D stacks adjacent slices (or orthogonal planes) as input channels of a 2D CNN — retaining ImageNet-pretrainability while adding limited inter-slice context at 2D cost.
**Source:** Roth et al., "A New 2.5D Representation for Lymph Node Detection using Random Sets of Deep CNN Observations", PMC4295635; plus recent glioma work arXiv:2603.17219
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4295635/
**Date:** 2015 (PMC record); arXiv 2026
**Excerpt (verbatim):** "We map this set-up by assigning the axial, coronal and sagittal slices in a Volume-of-Interest (VOI) into to these three channels." And (arXiv:2603.17219): "2.5D tri-planar input: A SliceEncoder25D concatenating adjacent slices across four modalities (12-channel), preserving inter-slice gradients ∇² at 2D cost."
**Confidence:** High for technique; quantitative knee-specific evidence thin.
**Pros:** near-2D cost, transfer-learning compatible, captures local z-context. **Cons:** still needs exam-level aggregation over stacks; anisotropic spacing (thick slices) limits benefit.

### 1d. 3D CNN (ResNet3D / MedicalNet)

**Claim:** Tencent MedicalNet (Med3D, Chen et al. 2019) provides 3D-ResNet (10–200) weights pretrained on an aggregated corpus ("23 datasets" version released 2019-07-30), with large transfer gains on lung segmentation and nodule classification; MIT license; weights downloadable (Google Drive/Weiyun; HF mirrors exist: TencentMedicalNet/MedicalNet-Resnet*, Zenodo record 15234379) → usable in a Kaggle no-internet notebook if uploaded as a Kaggle dataset.
**Source:** github.com/Tencent/MedicalNet; Med3D arXiv:1904.00625; Zenodo mirror
**URL:** https://github.com/Tencent/MedicalNet
**Date:** repo 2019-07-17 (updated 2019-07-30); Zenodo 2025-04-17
**Excerpt (verbatim):** "The MedicalNet project aggregated the dataset with diverse modalities, target organs, and pathologies to build relatively large datasets. Based on this dataset, a series of 3D-ResNet pre-trained models and corresponding transfer-learning training code are provided." Transfer results: "3D-ResNet50 | Train from scratch | 52.94% || MedicalNet | 89.25%" (LungSeg Dice); NoduleCls accuracy 84.85%→89.90% (ResNet50). Zenodo: "This is a 3D ResNet-50 model pre-trained on 23 medical datasets".
**Confidence:** High.

**Claim:** CoPAS (the most on-point prior work — see §2) uses ResNet3D-18 as the volumetric encoder for 12-class knee abnormality classification, exploiting the slice dimension explicitly: "The 3-dimensional convolution in ResNet3D enables the learning of patterns from the third dimension, which, in our case, corresponds to the spatial information along slices."
**Source:** Qiu et al., Nature Communications 2024 (details in §2).
**Confidence:** High.

**Pros (3D CNN):** native volumetric context; MedicalNet init accelerates convergence and accuracy. **Cons:** memory heavy; requires fixed-size volumes (resample/crop); MedicalNet pretraining is CT/MRI-segmentation-centric, not knee; risk of overfitting on ~10^3–10^4 exams.

### 1e. Video models treating slices as frames (SlowFast, X3D, TimeSformer, ViViT)

**Claim:** SlowFast has been applied directly to HRCT slice sequences for acute exacerbation of idiopathic pulmonary fibrosis classification, using a uniform "divide into 32 equal parts, sample 1 slice per part" strategy to handle variable slice counts.
**Source:** "Novel 3D-based deep learning for classification of acute exacerbation of idiopathic pulmonary fibrosis using high-resolution CT", PMC10928777
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10928777/
**Date:** 2024 (PMC)
**Excerpt (verbatim):** "For each patient, 128 consecutive scans from the middle section were selected and divided into 32 equal parts. Then, 1 scan was randomly chosen from each part to form a learnable sample consisting of 32 slices. To train the video classification model SlowFast, a sample with 32 images is proper. Samples were randomly drawn 10 times from each patient series yielding a total of 3060 samples."
**Confidence:** High.

**Claim:** ViViT (video vision transformer) applied end-to-end to 3D brain MRI by treating slices as frames with tubelet embedding + spatio-temporal attention + CLS-token aggregation (ViTranZheimer, Alzheimer's classification).
**Source:** "Leveraging Video Vision Transformer for Alzheimer's Disease ...", arXiv:2501.15733
**URL:** https://arxiv.org/pdf/2501.15733
**Date:** 2025-01
**Excerpt (verbatim):** "By treating the 3D MRI slices as consecutive frames in a video, we leverage a complex video understanding deep-based model to handle the long sequences of slices/frames. In this study, the recently introduced ViViT, a transformer-based video encoder, is employed for AD classification tasks."
**Confidence:** Medium-High (single application paper, no knee data).

**Claim:** X3D provides highly efficient video recognition (X3D-XL: Kinetics-400 top-1 79.1 at only 48.4×30 views-mult and 11.0M params — strong efficiency/accuracy tradeoff reference for efficiency award).
**Source:** Feichtenhofer, "X3D: Expanding Architectures for Efficient Video Recognition", CVPR 2020
**URL:** https://openaccess.thecvf.com/content_CVPR_2020/papers/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.pdf
**Date:** 2020
**Excerpt (verbatim):** "X3D-XL | Kinetics-400 | ... 48.4x30 | 11.0M" (table entry; X3D models reach state of the art at a fraction of SlowFast+NL's 234x30 / 59.9M budget).
**Confidence:** High (canonical paper).
**Overall pros:** Kinetics-pretrained weights (torchvision/pytorchvideo, downloadable for offline use), native temporal/strided sampling. **Cons:** MRI slice axis is not "time" (no motion dynamics); depth ordering must be canonicalized; evidence in MSK MRI is sparse vs. MRNet-style and 3D CNN approaches.

### 1f. MIL (multiple instance learning) over slices

**Claim:** Framing volumes as bags of slice-instances with only exam-level labels is standard; plain max/mean MIL pooling suffers unstable training (gradient issues), and attention-based MIL (ABMIL, Ilse et al. 2018) with gated attention is the usual remedy; attention weights give interpretable slice-importance maps.
**Source:** "Robust Weakly Supervised Learning for COVID-19 Recognition Using Multi-Center CT Images", arXiv:2112.04984
**URL:** https://arxiv.org/pdf/2112.04984.pdf
**Date:** 2021
**Excerpt (verbatim):** "some of the MIL pooling strategies, such as max-pooling and mean-pooling, very often lead to insufficient and unstable training because of gradient vanishing. To fix this problem, Ilse et al. combined the gated attention mechanism with the MIL strategy to solve the medical image classification problem".
**Confidence:** High.

**Claim:** Recent MIL extensions add probabilistic attention (uncertainty over instance contributions) and cluster-level sparsity; attention-MIL remains the dominant paradigm for weakly labeled volumetric/pathology data.
**Source:** arXiv:2507.14932 (Probabilistic smooth attention for deep MIL); arXiv:2509.11034 (csMIL)
**URL:** https://arxiv.org/abs/2507.14932
**Date:** 2025-07-20
**Excerpt (verbatim):** "MIL methods cast medical images as bags of instances (e.g. patches in whole slide images, or slices in CT scans), and only bag labels are required for training. Deep MIL approaches have obtained promising results by aggregating instance-level representations via an attention mechanism to compute the bag-level prediction."
**Confidence:** High.
**Pros:** matches exam-level supervision exactly; interpretable; cardinality-free. **Cons:** slice independence assumption; attention can dilute over many uninformative slices.

### 1g. Hybrid CNN–transformer slice aggregators

**Claim:** CNN-per-slice encoder + sequence model (BiLSTM or transformer) aggregation is a recognized family; pure ViT on small knee data underperforms CNNs, but hybrids that keep CNN slice features and learn inter-slice dependencies perform well.
**Source:** arXiv:2501.15733 (ViTranZheimer related-work); arXiv:2508.14151 (ViT underperformance on MRNet)
**URL:** https://arxiv.org/pdf/2501.15733
**Date:** 2025-01
**Excerpt (verbatim):** "CNN/ViT-BiLSTM models typically work by first using a CNN/ViT network to extract spatial features from each slice of MRI. The extracted features are then fed into an LSTM network to learn the temporal dependencies between the frames." And (arXiv:2508.14151): "InceptionV3 and ViT underperformed, likely due to their higher capacity and greater reliance on larger datasets or domain-specific pretraining to realize their full potential."
**Confidence:** Medium-High.
Also relevant: SB-SSL (Atito et al. 2022, "Slice-based self-supervised transformers for knee abnormality classification from MRI", cited in CoPAS refs) — self-supervised slice transformers on MRNet.

---

## 2. Multi-plane / multi-sequence fusion strategies

### 2a. MRNet-style: separate model per plane + logistic regression (late/decision fusion)

**Claim:** Per-plane models + logistic regression stacking was validated as best among MRNet variants by Azcona et al. (2020) comparative study.
**Source:** "Comparative Analysis of Backbone Networks for Deep Knee MRI Classification Models", MDPI J. Imaging 2022 (citing Azcona et al. 2020, IDSTA)
**URL:** https://www.mdpi.com/2504-2289/6/3/69
**Date:** 2022-06-21
**Excerpt (verbatim):** "In Azcona et al. (2020), the authors have shown that the three-model MRNet architecture with logistic regression has the best validation performance, compared to different variations of the same architecture, including self-trained AlexNets as a feature extraction layer, and single models, which operate on a concatenation of slices throughout all three MRI planes."
**Confidence:** High.

### 2b. Shared/per-branch encoders + attention fusion: CoPAS — the single most relevant prior work (12 knee abnormalities!)

**Claim:** CoPAS (Qiu/Xie/Chen et al., Nature Communications 2024) tackles *exactly* 12 knee abnormality classes from multi-plane (sag/cor/ax PDW + coronal T1W + sagittal T2W) multi-center (5 centers, 1748 patients) knee MRI. Design: (i) crop around U-Net-meniscus-segmented ROI; (ii) generate **synthetic cross-plane volumes by rotating PDW volumes** to counter anisotropic slice thickness; (iii) three plane branches sharing a ResNet3D-18 encoder, with **cross-plane cross-attention** (main-plane volume as query, rotated volumes as key/value); (iv) **co-plane cross-sequence attention** where T1W/T2W act as channel-wise attention filters on PDW features; (v) plane-aware 12×12×12 probability matrix + SE-style correlation mining for final prediction; (vi) Focal loss on final output + BCE on branch outputs. Result: average AUC-ROC 0.812 internal, beating MRNet/MPFuseNet/ELNet in 8/12 classes; external-center drop to 0.721–0.726. Competitive with senior radiologists; boosts junior radiologists.
**Source:** Qiu et al., "Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities", Nature Communications 15, 2024; code: https://github.com/zqiuak/CoPAS
**URL:** https://www.nature.com/articles/s41467-024-51888-4
**Date:** 2024-09-02
**Excerpts (verbatim):**
- "The results show that our method outperforms other models with an average AUC-ROC of 0.812. Specifically, our CoPAS outperformed the three extant models in 8 out of 12 abnormalities."
- "a decline of the average AUC-ROCs from 0.812 to 0.721 and 0.726 is observed when transitioning from the internal dataset to the two external datasets. The results indicate a strong reliance on T1W and T2W sequences for diagnosing bone contusion (CONT)... the performance drop is due to fewer sequences available in the external dataset, as well as the data distribution shifting between different centers caused by multiple factors including scanning parameters and patient demographics."
- On fusion design vs naive concat: "Belton et al. explored different strategies for fusing multi-planar MRI. The paper uses simple concatenation for late fusion. This equal-contribution structure ignores the implicit correlations in low-level features and thus makes it vulnerable to noise when the number of sequences increases."
- On multi-task caveat: "single-task methods, such as the MRNet, may outperform our approach in some tasks. The underlying reason is that certain characteristics of these abnormalities may be misguided by other tasks in our multi-task model... For instance, the intensity enhancement of effusion resembles that of a cyst."
- Loss: "we use Focal Loss to measure the distance between the final result y and label... For the prediction of three branches y_branch, the binary cross entropy (BCE) loss is applied."
**Confidence:** High (peer-reviewed, directly on-task, code available).

### 2c. MPFuseNet: spatial attention + multi-plane fusion study

**Claim:** Belton et al. systematically compared single-plane vs multi-plane and fusion methods on MRNet data; their MPFuseNet (ResNet18 + spatial attention block, multi-view) reached AUC 0.977 ACL / 0.957 abnormal — SOTA on MRNet validation; multiple planes help, and how you fuse matters.
**Source:** Belton et al., "Optimising Knee Injury Detection with Spatial Attention and Validating Localisation Ability", MIUA 2021 / arXiv:2108.08136
**URL:** https://arxiv.org/abs/2108.08136
**Date:** 2021-08-18
**Excerpt (verbatim):** "As MRI data is acquired from three planes, we compare our technique using data from a single-plane and multiple planes (multi-plane). For multi-plane, we investigate various methods of fusing the planes in the network. This analysis resulted in the novel 'MPFuseNet' network and state-of-the-art Area Under the Curve (AUC) scores for detecting Anterior Cruciate Ligament (ACL) tears and Abnormal MRIs, achieving AUC scores of 0.977 and 0.957 respectively."
**Confidence:** High.

### 2d. General fusion evidence (early/joint/late) for image+non-image modalities

**Claim:** Systematic review of imaging+EHR fusion: early fusion most common and consistently ≥ single-modality; joint (intermediate) fusion can outperform late fusion (Yoo et al.: joint 0.746 vs late 0.724 AUROC); late fusion is simplest, robust to missing modalities, and needs less data. Guidance: build single-modality baselines first, then compare fusion strategies.
**Source:** Huang et al., "Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines", PMC7567861 (npj Digital Medicine)
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7567861/
**Date:** 2020
**Excerpts (verbatim):** "Based on the reviewed papers, early fusion consistently improved performance over single modality models, and is supported by this review as an initial strategy to fuse multimodal data." "Yoo et al. further compared their joint fusion model to a late fusion model and achieved a 0.02 increase in Area Under Receiver Operating Characteristic Curve (AUROC)." Table: late fusion = "Able to make predictions when not all modalities are present ✓; Does not necessarily require a large amount of training data ✓".
**Confidence:** High (systematic review).

**Claim:** Intermediate fusion timing is task-dependent; a sequential forward search over fusion points (SFSA with MMTM modules) beat both late fusion and unimodal baselines on multimodal MRI.
**Source:** "Timing Is Everything: Finding the Optimal Fusion Points in Multimodal Medical Imaging", arXiv:2505.02467
**URL:** https://arxiv.org/html/2505.02467v1
**Date:** 2025-05-05
**Excerpt (verbatim):** "late fusion methods often miss critical cross-modal interactions that emerge at intermediate representation levels. Intermediate fusion, bonded with deep networks, offers a promising middle ground... Our results consistently demonstrate improved classification metrics and reduced training time."
**Confidence:** Medium-High.

**Evidence synthesis for this competition:** per-plane models + stacking (MRNet) is proven and simple; attention-based cross-plane/cross-sequence fusion (CoPAS) is the current best-published design for 12-class knee MRI; naive concatenation fusion underperforms. With 16 institutions and heterogeneous protocols, keep modality/plane branches modular so missing sequences degrade gracefully (late/joint fusion handles missing modalities better than early fusion).

---

## 3. Pretraining

### 3a. ImageNet transfer value — Raghu et al. "Transfusion" (NeurIPS 2019)

**Claim:** ImageNet transfer to medical images gives only modest final-performance gains but large convergence-speed gains; most of the benefit comes from **weight scaling of the lowest layers**, not high-level feature reuse. Practical implications: (i) Mean-Var Init (Gaussian init matching pretrained per-layer mean/var) recovers most convergence benefit without weights; (ii) transferring only the lowest ~2 layers captures most benefit; (iii) smaller/slimmer architectures are competitive on small medical datasets.
**Source:** Raghu, Zhang, Kleinberg, Bengio, "Transfusion: Understanding Transfer Learning for Medical Imaging", arXiv:1902.07208 / NeurIPS 2019 (summary via AlphaXiv + AI Summer)
**URL:** https://arxiv.org/abs/1902.07208 (summary: https://www.alphaxiv.org/abs/1902.07208; https://theaisummer.com/medical-imaging-transfer-learning/)
**Date:** 2019
**Excerpts (verbatim, from summaries quoting the paper's findings):** "While transfer learning provided minimal performance improvements, it consistently accelerated convergence. The authors discovered that this acceleration stems largely from better weight scaling rather than meaningful feature reuse." "The largest marginal benefit comes from reusing only the first convolutional layer (conv1); Benefits plateau after reusing the first two blocks/stages." "Transfer the scale (range) of the weights instead of the weights themselves. This offers feature-independent benefits that facilitate convergence."
**Confidence:** High (canonical paper; findings widely replicated).
**Implication:** For 12-label knee MRI with limited data, ImageNet init is still the easiest strong default (as MRNet itself did), but do not expect large gains vs. good init; ELNet's from-scratch 0.2M-param result corroborates this.

### 3b. RadImageNet

**Claim:** RadImageNet (1.35M CT/MRI/US images, 11 anatomical regions; 4 backbones: InceptionResNetV2, ResNet50, DenseNet121, InceptionV3) gives +0.9%–9.4% AUROC over ImageNet pretraining across 8 downstream medical applications. Weights on GitHub (BMEII-AI/RadImageNet) — **but officially TensorFlow/Keras (.h5)**; PyTorch availability is unofficial/partial (DenseNet121), which matters for a PyTorch Kaggle pipeline.
**Source:** Mei et al., Radiology: AI 2022; github.com/BMEII-AI/RadImageNet; framework caveat via EmergentMind topic summary
**URL:** https://github.com/BMEII-AI/RadImageNet ; https://www.emergentmind.com/topics/radimagenet
**Date:** 2022 (paper); repo 2021-06-02
**Excerpts (verbatim):** "RadImageNet pretrained models show superior performance in the classification of eight independent medical applications as compared with ImageNet pretrained models, showing improvements from 0.9% to 9.4% for AuROC curve." Caveat: "original RadImageNet pretraining used TensorFlow for InceptionV3 and ResNet-50 and published PyTorch weights for DenseNet-121, and it explicitly raises the possibility that differences between TensorFlow and PyTorch weight dumps may degrade downstream performance in PyTorch (Frees et al., 25 Aug 2025)."
**Confidence:** High on dataset/claims; Medium on cross-framework weight fidelity.

### 3c. MedicalNet (3D) — see §1d. MIT license, HF mirrors exist (`TencentMedicalNet/MedicalNet-Resnet10..200`, unofficial HF classification wrapper `nwirandx/medicalnet-resnet3d50_23datasets`, Zenodo CPU-only ResNet-50) — all can be zipped into a Kaggle dataset for offline use.

### 3d. BiomedCLIP / medical VLMs

**Claim:** BiomedCLIP (ViT-B/16 image encoder + PubMedBERT text encoder, trained on PMC-15M: 15M figure-caption pairs from 4.4M PMC articles) sets SOTA on biomedical VLP, beats radiology-specific BioViL on RSNA Pneumonia; **Apache 2.0, publicly downloadable** (HF `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) — usable on Kaggle if mirrored as a dataset; English-only text encoder (relevant if reports are multilingual).
**Source:** Zhang et al., arXiv:2303.00915 / NEJM AI 2025; Azure AI catalog entry
**URL:** https://arxiv.org/abs/2303.00915 ; https://ai.azure.com/catalog/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
**Date:** 2023-03-02
**Excerpts (verbatim):** "BiomedCLIP is a biomedical vision-language foundation model that is pretrained on PMC-15M, a dataset of 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central, using contrastive learning. It uses PubMedBERT as the text encoder and Vision Transformer as the image encoder, with domain-specific adaptations." "The model is publicly available under the Apache 2.0 license." "This model was developed using English corpora, and thus can be considered English-only." "To date, BiomedCLIP outperforms the radiology-specific BioViL model on the RSNA pneumonia detection benchmark and achieves a mean accuracy of 75.5% across five zero-shot classification datasets, a 12-point improvement over general-domain CLIP."
**Confidence:** High.

**Claim:** Newer MRI-specific foundation models exist and are worth checking for Kaggle-usable weights: MRI-CORE (2D MRI foundation model, 6M+ slices, 110k+ volumes, 18 body locations) and MedImageInsight (multi-modality medical VLM).
**Source:** "Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations", arXiv:2509.21249
**URL:** https://arxiv.org/html/2509.21249v2
**Date:** 2025
**Excerpt (verbatim):** "MRI-CORE: A 2D MRI-specific foundation model pretrained on more than 6 million slices from over 110,000 MRI volumes across 18 body locations."
**Confidence:** Medium (availability of weights must be verified case by case).

### 3e. MONAI bundles
No direct search hits retrieved in this session (2 queries returned 0 results). Known facts (lower confidence, to verify): MONAI Model Zoo / bundles (e.g., spleen CT, whole-body segmentation, MedNIST classifiers) distribute self-contained weights+configs as versioned bundles loadable offline via `monai.bundle.load`; NGC-hosted downloads must be mirrored into a Kaggle dataset for no-internet runs. Confidence: Medium.

---

## 4. Variable slice counts & heterogeneous acquisition (16 institutions)

### 4a. Slice-count strategies
- **Cardinality-invariant aggregation** (max/attention/MIL pooling over slices) — MRNet/ELNet/CoPAS branch design; no resampling needed.
- **Uniform slice subsampling for fixed-length models**: SlowFast-on-CT recipe — "128 consecutive scans from the middle section were selected and divided into 32 equal parts. Then, 1 scan was randomly chosen from each part to form a learnable sample consisting of 32 slices" (PMC10928777, verbatim above). Random per-epoch resampling = free augmentation.
- **Resample/pad to fixed grid for 3D CNNs**: e.g., glioma pipeline: "resampling to a target volume size of (160,160,160), intensity clipping to the 1st and 99th percentiles, slice-wise CLAHE" (arXiv:2601.07035, verbatim). For knee: CoPAS crops ROI via U-Net meniscus segmentation and uses synthetic rotated volumes instead of interpolating thick slices: "Conventionally, using interpolation will reduce this imbalance, but the interpolation with one image itself does not provide any information gain and may introduce additional errors." (CoPAS, verbatim).
**Confidence:** High.

### 4b. Scanner/site domain shift — severity
**Claim:** Cross-manufacturer domain shift is severe: "Our experimental results reveal a substantial decline in classification performance when models trained on one type of scanner manufacturer are tested with data from different manufacturers. Moreover, despite applying ComBat-based harmonization, the harmonized images do not demonstrate any noticeable performance enhancement for disease classification tasks." (Kushol et al., Scientific Reports 2023, GE/Philips/Siemens multi-center 3D MRI).
**URL:** https://www.nature.com/articles/s41598-023-43715-5.pdf
**Confidence:** High. **Implication:** ComBat on images is NOT a reliable fix for DL classifiers; prefer in-model robustness (normalization, augmentation, domain-adversarial/BN strategies) and stratified CV by site.

**Claim (ComBat where it does work):** feature-level ComBat/NeuroCombat is effective for radiomics/derived features: "ComBat harmonization effectively corrected the heterogeneity for most BIMs, though it was less successful for certain BIMs" (PubMed 41380987); "ComBat successfully removes systematic biases associated with scanner across multiple sites in which acquisition protocols were not fully harmonized" (PMC9036665). ComBat drawbacks: "ComBat relies heavily on labelled data to perform efficient batch correction... if new data is to be harmonized then it must be added in the existing pool of data" (MDPI JPM 2021) — i.e., awkward for test-time-only data in a competition.
**URLs:** https://pubmed.ncbi.nlm.nih.gov/41380987/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC9036665/ ; https://www.mdpi.com/2075-4426/11/9/842
**Confidence:** High.

**Claim (intensity standardization):** Histogram/Nyúl intensity standardization per training-set statistics is a cheap, effective mitigation used by ELNet: "we perform histogram-based intensity standardization according to the training set statistics, thus enabling similar-valued pixels to be associated with the relevant tissue type" (arXiv:2005.02706, verbatim). Deep alternatives: DeepHarmony / HACA3 image translation beat neuroCombat for cross-protocol consistency (OUCI summary, Imaging Neuroscience 2025).
**Confidence:** High for Nyúl/z-score/histogram matching; Medium-High for GAN harmonizers.

**Practical recipe for 16 institutions:** per-exam percentile clipping + z-score or Nyúl standardization; heavy augmentation (intensity jitter, contrast, bias-field-like artifacts, rotation/translation as in MRNet: "rotated randomly between –25 and 25 degrees, shifted randomly between –25 and 25 pixels, and flipped horizontally with 50% probability"); site-stratified validation; optionally domain-adversarial heads or per-site BN statistics. MRNet external-validation drop (0.824 zero-shot → 0.911 retrained) quantifies the stakes.

---

## 5. Text-side modeling (radiology reports)

### 5a. Encoders & fine-tuning
**Claim:** Fine-tuned BERT-family models on radiology reports reach SOTA labeling with little data and few epochs; CheXbert F1 0.798 vs rule-based CheXpert 0.743 on chest reports; knee-MR report labeling specifically: neural net F1 0.867 > random forest 0.822.
**Source:** "Automated labelling of radiology reports using natural language processing" (PMC11080679); CheXbert (EMNLP 2020, aclanthology.org/2020.emnlp-main.117)
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11080679/
**Date:** 2024 (PMC record)
**Excerpts (verbatim):** "By adding a simple single‐hidden‐layer neural network classifier on top of BERT and fine‐tuning BERT, superior performance can be achieved, even with small datasets with few positive labels... fine‐tuning BERT on radiology reports thus only requires a small single‐digit number of epochs to achieve state‐of‐the‐art performance." Table 2: "Conventional machine learning (e.g., random forest ensemble on MR reports of the knee) ... F1 0.822 | Neural network (e.g., neural network on MR reports of the knee) ... F1 0.867". CheXbert: "CheXbert achieves a statistically significant improvement on F1 of 0.055 (0.039, 0.070). The board-certified radiologist achieves an F1 of 0.805... 0.007 F1 points higher than... CheXbert." CheXbert inference: "With a single TITAN-XP GPU, the CheXbert model's inference time reduces to ~18 minutes" for 190k reports.
**Confidence:** High.

### 5b. Multilingual encoders
**Claim:** XLM-RoBERTa (100 languages) is the standard multilingual fine-tuning backbone and consistently tops multilingual classification comparisons (e.g., 0.87 accuracy vs mBERT 0.80, deBERTa 0.81 on bilingual classification); for embedding-only use, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (118M params, 384-dim, 50+ languages, Apache-2.0, ONNX export supported) is the efficiency-friendly choice.
**Source:** arXiv:2406.07287 (EXIST 2024 bilingual classification); HF model card mirror
**URL:** https://arxiv.org/html/2406.07287 ; https://github.com/shinichiro-takahashi-sbr/paraphrase-multilingual-MiniLM-L12-v2
**Date:** 2024
**Excerpts (verbatim):** "XLM-RoBERTa: A multilingual variant of RoBERTa, trained on 100 languages, known for robust performance across various multilingual benchmarks" (table: XLM-R/param tuning 0.87 acc > mBERT 0.80). MiniLM card: "It maps sentences & paragraphs to a 384 dimensional dense vector space"; "118M parameters; License: Apache 2.0; Supported Languages: 50+ languages; Output Dimensions: 384 ... implements mean pooling".
**Confidence:** High.

### 5c. Fine-tune vs freeze; fusion with images; train-time-only text
- **Fine-tune vs freeze:** evidence above favors fine-tuning even with small data; freezing pretrained encoders + linear probe is viable when compute/data are tiny (BiomedCLIP linear-probe results: "moderate adaptation strategies (e.g. linear probing) will yield further performance gains at a reasonable computational cost" — arXiv:2506.14136).
- **Late vs cross-modal fusion:** general medical-fusion evidence (§2d) supports starting with late fusion (concatenate exam-embedding + report-embedding → MLP, or stack predictions); intermediate/cross-attention fusion gives small gains (≈+0.02 AUROC) at higher complexity and brittleness when reports may be missing at test time — a key competition design question. Late fusion degrades gracefully if test-time reports are absent.
- **Reports at train time only:** reports can be used to (i) derive/verify labels via a fine-tuned report classifier (CheXbert-style — directly relevant since this competition supplies reports), (ii) distill text knowledge into the image model (train-time cross-modal attention/teacher), keeping the test-time pipeline image-only. CheXbert shows report→label mapping is fast and near-radiologist accurate.
**Confidence:** Medium-High (fusion-partly extrapolated from adjacent domains).

---

## 6. Efficiency optimization (efficiency awards)

**Claim (PTQ → TensorRT INT8 on medical models):** MedPTQ pipeline (fake-quantized ONNX with Q/DQ nodes → real INT8 TensorRT engine) on 7 SOTA 3D medical segmentation models: "reduces model size by factors of 2.42× to 3.85×, and inference latency by 2.05× to 2.66× ... without sacrificing performance, as the mDSC remains nearly unchanged after quantization."
**Source:** "Post-Training Quantization for 3D Medical Image Segmentation: A Practical Study on Real Inference Engines", arXiv:2501.17343 (also PMC: MedPTQ)
**URL:** https://arxiv.org/html/2501.17343v1
**Confidence:** High.

**Claim (Kaggle T4 caveat — critical):** On Kaggle's T4 environment, TensorRT libraries may be absent; ONNX Runtime INT8 then falls back with per-op GPU↔CPU memcpy and can be ~147× SLOWER than PyTorch FP16: "TensorRT libraries are absent from the Kaggle T4 environment (libnvinfer.so.10 not found), preventing ONNX Runtime from dispatching quantized operators to the GPU's INT8 tensor cores. As a result, the quantized ONNX graph contains 192 GPU↔CPU memcpy nodes." Practical takeaway: on Kaggle, prefer **PyTorch FP16/AMP or torch.compile; treat INT8-ORT with caution; if using TensorRT, vendor the libs or build engines in-notebook**.
**Source:** "Guidance-Aware Quantization for Classifier-Free Diffusion", arXiv:2607.08241
**URL:** https://arxiv.org/html/2607.08241v1
**Date:** 2026-07-09
**Confidence:** High (measured on Kaggle T4).

**Claim (typical quantization impact):** FP16 ≈2× speedup, <0.5% accuracy drop; INT8 2–4× speedup, 1–3% drop (deployment guide consensus); BERT TensorRT case: FP16 4.09×, INT8 4.34× with 99.47% accuracy recovery (github.com/Yangjianxiao0203/bert-lora).
**Confidence:** Medium-High (community benchmarks).

**Claim (distillation):** Teacher→student KD with soft targets retains accuracy at small size on medical classification: "the distilled student model maintains high classification performance with significantly reduced parameters and inference time, making it an optimal choice in resource-constrained clinical environments" (arXiv:2508.15251, VGG19/ViT teachers → OFA-595 student on CXR). Subclass KD helps low-class-count tasks (+1.49% F1 over vanilla KD; x-mol summary of SKD).
**Confidence:** High for KD-as-strategy; Medium for exact deltas.

**Claim (small backbones that retain accuracy):** ELNet (~0.2M params) matches MRNet on knee MRI (§1b) — the single best efficiency evidence for this task. X3D (11M params) and MobileNet-class 2D encoders are proven efficient backbones; ensemble then distill into one small student for the efficiency track.
**Confidence:** High.

---

## 7. Bottom-line recommendations for the competition
1. **Baseline:** MRNet-style — per-slice ImageNet-pretrained ResNet/EfficientNet + attention-MIL (gated) pooling per plane/sequence; late-fuse with logistic regression/stacker. Proven, robust to variable slice counts and missing sequences.
2. **Upgrade path:** CoPAS architecture (github.com/zqiuak/CoPAS) — literally built for 12-class multi-plane multi-sequence knee MRI; add cross-plane attention via rotated volumes + cross-sequence channel attention; multi-task BCE/focal loss with per-branch supervision.
3. **3D option:** ResNet3D-18/50 with MedicalNet weights (mirrored to Kaggle dataset); needs fixed volume size — crop ROI + uniform slice sampling.
4. **Pretraining:** ImageNet is fine (Transfusion: mostly low-layer scaling benefits); RadImageNet adds up to +9.4% AUROC but is TF-native (PyTorch fidelity caveat); BiomedCLIP ViT-B/16 (Apache-2.0) is the best off-the-shelf medical 2D ViT; MRI-CORE/MedImageInsight if weights obtainable.
5. **Domain shift (16 institutions):** Nyúl/z-score intensity standardization + aggressive intensity augmentation; site-stratified CV; don't rely on image-level ComBat (shown ineffective for DL classifiers); feature-level ComBat only if radiomics-style features are used.
6. **Text:** fine-tuned XLM-R (multilingual) or MiniLM-L12-v2 embeddings (118M, Apache-2.0) on reports; late-fuse report embedding with image exam-embedding; alternatively use reports at train time only (label refinement via CheXbert-style classifier, or text→image distillation) if test-time reports are unavailable.
7. **Efficiency track:** ELNet-class tiny CNN is shockingly strong on knee MRI; distill ensembles into a small student; deploy FP16 PyTorch/ONNX-FP16; INT8-ORT on Kaggle T4 can backfire (~147× slowdown) unless TensorRT libs are available.
