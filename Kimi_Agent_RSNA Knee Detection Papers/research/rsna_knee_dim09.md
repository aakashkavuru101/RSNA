# Dimension 09 — Practical Engineering Seed Knowledge
## Kaggle "RSNA Knee Abnormality Detection" (2026) — knee MRI DICOMs + multilingual reports
**Research date:** 2026-08-10 | **Sub-agent:** dimension 09 (practical engineering)
**Scope:** DICOM handling, MRI preprocessing, Kaggle platform mechanics, baseline volumetric pipeline, external data, agentic-coding workflow.

Citation format: **Claim / Source / URL / Date / Excerpt (verbatim) / Confidence**.

---

## 1. DICOM HANDLING

### 1.1 Correct slice sorting (ImagePositionPatient along the slice normal, not InstanceNumber alone)
- **Claim:** Robust 3D volume construction sorts slices by projecting `ImagePositionPatient` (0020,0032) onto the slice normal computed as `cross(row, col)` from `ImageOrientationPatient` (0020,0037); InstanceNumber sorting is only a fallback. Slice spacing is best computed from actual positions of first/last slices rather than trusting the `SliceThickness` tag.
- **Source:** 3D Slicer community discourse (worked code example for a DL dataset)
- **URL:** https://discourse.slicer.org/t/dicom-to-voxel-python-landmark-placement-issue-for-dl-dataset/44366
- **Date:** 2025-09-05
- **Excerpt:** "if hasattr(first_slice, 'ImagePositionPatient'): dicom_datasets.sort(key=lambda s: float(get_value_tag(s, tag='ImagePositionPatient')[2])) else: dicom_datasets.sort(key=lambda s: int(s.InstanceNumber)) ... # --- Sort slices robustly along the normal --- dicom_datasets.sort(key=lambda s: np.dot(get_value_tag(s, tag='ImagePositionPatient'), normal)) ... # The most robust method is to use the difference between the positions of the first and last slices and divide by the number of steps. This handles cases where spacing might be slightly irregular."
- **Confidence:** High (standard practice; matches DICOM standard geometry).

- **Claim:** The voxel-to-patient mapping is fully defined by ImagePositionPatient + ImageOrientationPatient + PixelSpacing (DICOM Equation C.7.6.2.1-1): P_xyz = S_xyz + i·Δi·X_xyz + j·Δj·Y_xyz.
- **Source:** DICOM Standard Browser (Innolitics), Image Position (Patient) attribute
- **URL:** https://dicom.innolitics.com/ciods/rt-dose/image-plane/00200032
- **Date:** 2024-04-18
- **Excerpt:** "The Image Plane Attributes, in conjunction with the Pixel Spacing Attribute, describe the position and orientation of the image slices relative to the Patient-Based Coordinate System. In each image frame Image Position (Patient) (0020,0032) specifies the origin of the image with respect to the Patient-Based Coordinate System. RCS and Image Orientation (Patient) (0020,0037) values specify the orientation of the image frame rows and columns."
- **Confidence:** High (DICOM standard).

### 1.2 Tags that matter for knee MRI volumes
- **Claim:** The core tag set for stacking MRI series into arrays: SeriesInstanceUID (0020,000E) for grouping, SeriesDescription (0008,103E) for sequence identification (T1/T2/PD, fat-sat, plane), InstanceNumber (0020,0013), ImagePositionPatient (0020,0032), ImageOrientationPatient (0020,0037), PixelSpacing (0028,0030), SliceThickness (0018,0050), SliceLocation (0020,1041), PhotometricInterpretation (0028,0004), RescaleSlope/Intercept (0028,1053/1052), BitsStored/BitsAllocated, WindowCenter/Width (0028,1050/1051).
- **Source:** "DICOM images in Python: An overview" (full annotated DICOM header dump)
- **URL:** https://www.peco602.com/post/0090-python-dicom/
- **Date:** 2023-02-23
- **Excerpt:** "(0028, 0004) Photometric Interpretation CS: 'MONOCHROME2' ... (0028, 0030) Pixel Spacing DS: [0.453125, 0.453125] ... (0028, 1052) Rescale Intercept DS: '-1024.0' (0028, 1053) Rescale Slope DS: '1.0' ... You can access specific elements by their DICOM keyword or tag number."
- **Confidence:** High.

- **Claim:** Multi-frame / enhanced DICOM (functional groups) buries spacing/orientation/position inside `PerFrameFunctionalGroupsSequence` or `SharedFunctionalGroupsSequence`; naive `ds.PixelSpacing` access fails on such files.
- **Source:** Same 3D Slicer discourse code
- **URL:** https://discourse.slicer.org/t/dicom-to-voxel-python-landmark-placement-issue-for-dl-dataset/44366
- **Date:** 2025-09-05
- **Excerpt:** "if has_perframe(ds): return [float(ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[i]) for i in (0, 1)] if has_shared(ds): return [float(ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[i]) for i in (0, 1)] return [float(ds.PixelSpacing[i]) for i in (0, 1)]"
- **Confidence:** High.

### 1.3 Photometric interpretation & LUT pitfalls
- **Claim:** `ds.pixel_array` returns raw stored values; display/analysis-correct pixels require `apply_modality_lut` (rescale slope/intercept) BEFORE `apply_voi_lut` (windowing), and MONOCHROME1 images must be inverted (min value displays as white).
- **Source:** pydicom official API docs (apply_voi_lut, apply_modality_lut)
- **URL:** https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html ; https://pydicom.github.io/pydicom/2.1/reference/generated/pydicom.pixel_data_handlers.apply_modality_lut.html
- **Date:** accessed 2026-08-10 (pydicom 3.0.2 docs)
- **Excerpt (apply_voi_lut):** "When the dataset requires a modality LUT or rescale operation as part of the Modality LUT module then that must be applied before any windowing operation." Excerpt (apply_modality_lut): "If (0028,1052) Rescale Intercept and (0028,1053) Rescale Slope are present then returns an array of np.float64. If neither are present then arr will be returned unchanged."
- **Confidence:** High (official docs).

- **Claim:** Practical MONOCHROME1 handling: after modality+VOI LUT, invert via `arr = arr.max() - arr`; skipping windowing and min-max normalizing raw 16-bit data to 8-bit yields washed-out/inverted images.
- **Source:** K-Dense-AI scientific-agent-skills pydicom SKILL.md + Daniweb DICOM display thread
- **URL:** https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/pydicom/SKILL.md ; https://www.daniweb.com/programming/software-development/threads/333141/help-with-displaying-dicom-images-in-python-imaging-library
- **Date:** 2025-10-19 (skill file)
- **Excerpt:** "For grayscale display, apply transforms in this order: `modality_values = apply_modality_lut(frame, ds); display_values = apply_voi_lut(modality_values, ds, index=0)` ... MONOCHROME1 may require presentation inversion. Palette Color requires `apply_color_lut()` ... Never use per-frame min/max normalization for quantitative analysis."
- **Confidence:** High for order/inversion; note MRI is usually MONOCHROME2 — check per-series.

- **Claim:** pydicom 3.x `pydicom.pixels.pixel_array` supports path-based frame-specific decoding; shape semantics: grayscale single frame `(rows, columns)`, grayscale multi-frame `(frames, rows, columns)`.
- **Source:** K-Dense-AI pydicom SKILL.md
- **URL:** https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/pydicom/SKILL.md
- **Date:** 2025-10-19
- **Excerpt:** "The stable `pydicom.pixels` API supports path-based, frame-specific decoding ... `raw=False` converts YCbCr pixel data to RGB when possible ... Use `iter_pixels(path, indices=[...])` for bounded multi-frame iteration."
- **Confidence:** High.

### 1.4 dcm2niix / nibabel / dicom2nifti for NIfTI
- **Claim:** dcm2niix is the de-facto standard DICOM→NIfTI converter; it emits a BIDS JSON sidecar with metadata. Simple usage: `dcm2niix -o ~/outdir ~/dicomdir`; batch via `dcm2niibatch batch_config.yml`.
- **Source:** Neuroimaging Cookbook recipe + dcm2niix repo (rordenlab)
- **URL:** https://neuroimaging-cookbook.github.io/recipes/dcm2nii_recipe/ ; https://github.com/rordenlab/dcm2niix/blob/master/BIDS/README.md
- **Date:** 2021-12-12 (recipe); repo current
- **Excerpt:** "The only required argument for dcm2niix is the location of the folder with the DICOM files to convert, which is always the final argument provided. `dcm2niix -o ~/outdir ~/dicomdir`" and "dcm2niix is designed to convert complicated DICOM images in to simple NIfTI images ... dcm2niix can create Brain Imaging Data Structure (BIDS) files that retain useful information. These are simple human-readable text files in the JSON format."
- **Confidence:** High.

- **Claim:** dcm2niix handles many transfer syntaxes (raw, RLE, JPEG lossless built-in; JPEG-LS via CharLS; JPEG2000 via OpenJPEG/Jasper) — relevant because competition DICOMs may be compressed.
- **Source:** dcm2niix README (mirror)
- **URL:** https://gitee.com/QQ975150313/dcm2niix
- **Date:** repo current (accessed 2026-08-10)
- **Excerpt:** "The base code includes support for raw, run-length encoded, and classic JPEG lossless decoding ... JPEG-LS lossless support is optional, and can be provided by using CharLS. JPEG2000 lossy and lossless support is optional, and can be provided using OpenJPEG or Jasper."
- **Confidence:** High.

- **Claim:** Caveat: metadata can be missing after anonymization — dcm2niix cannot emit fields the DICOM header lacks (relevant because RSNA data is anonymized; don't rely on rich protocol metadata).
- **Source:** dcm2niix BIDS README
- **URL:** https://github.com/rordenlab/dcm2niix/blob/master/BIDS/README.md
- **Date:** accessed 2026-08-10
- **Excerpt:** "Note that dcm2niix cannot provide information that is not in the DICOM header. Common reasons for absent fields are: ... It was removed from the DICOM during anonymization, possibly by accident or overzealousness"
- **Confidence:** High.

### 1.5 SimpleITK / MONAI series reading
- **Claim:** SimpleITK `ImageSeriesReader.GetGDCMSeriesFileNames(dir, seriesID)` returns correctly GDCM-sorted slice filenames; `GetGDCMSeriesIDs` enumerates series in a directory; `useSeriesDetails=True` separates multiple 3D volumes sharing one SeriesUID (perfusion/DTI case).
- **Source:** SimpleITK official Doxygen, ImageSeriesReader
- **URL:** https://simpleitk.org/doxygen/v2_3/html/classitk_1_1simple_1_1ImageSeriesReader.html
- **Date:** 2023-09-12 (v2.3 docs; v2_5 exists)
- **Excerpt:** "useSeriesDetails: Use additional series information such as ProtocolName and SeriesName to identify when a single SeriesUID contains multiple 3D volumes - as can occur with perfusion and DTI imaging. ... loadSequences: Parse any sequences in the DICOM data set. Loading DICOM files is faster when sequences are not needed."
- **Confidence:** High.

- **Claim:** Minimal SimpleITK series read pattern, plus conversion to numpy: `reader = sitk.ImageSeriesReader(); dicom_names = reader.GetGDCMSeriesFileNames(path); reader.SetFileNames(dicom_names); image = reader.Execute(); arr = sitk.GetArrayFromImage(image)` (array axis order is z,y,x). Isotropic resample via `sitk.ResampleImageFilter` with `SetOutputSpacing([1,1,1])`.
- **Source:** CSDN SimpleITK tutorial (mirrors common usage)
- **URL:** https://blog.csdn.net/q610098308/article/details/132740598
- **Date:** 2023-08-30
- **Excerpt:** "reader = sitk.ImageSeriesReader(); dicom_names = reader.GetGDCMSeriesFileNames('D:/dicom'); reader.SetFileNames(dicom_names); image = reader.Execute(); img_array = sitk.GetArrayFromImage(image) ... resample.SetOutputSpacing(newspacing); newimage = resample.Execute(image)"
- **Confidence:** High (matches SimpleITK official examples).

- **Claim:** Version-sensitivity warning: SimpleITK 2.4.0 changed DICOM series direction handling (Z component sign flip vs 2.3.1) — pin your SimpleITK version in the submission environment.
- **Source:** SimpleITK GitHub issue #2214
- **URL:** https://github.com/SimpleITK/SimpleITK/issues/2214
- **Date:** 2025-01-08
- **Excerpt:** "Reading a DICOM image series in version 2.4.0 results in an unexpected direction matrix, not seen in version 2.3.1. The Z component becomes negative (while the image origin remains unchanged)."
- **Confidence:** Medium-High (single bug report, but from official repo).

---

## 2. MRI PREPROCESSING — WHAT MATTERS FOR CNN CLASSIFICATION

### 2.1 Intensity normalization
- **Claim:** Z-score (per-volume/ROI), WhiteStripe, and Nyul histogram standardization are the three representative MRI normalization families; for *radiomics/handcrafted features* Nyul gave the most robust first-order features, but for *classification accuracy* all three performed similarly (and Nyul sometimes slightly worse than no normalization) — i.e., simple per-volume z-score is usually enough for CNNs.
- **Source:** Nature Scientific Reports: "Standardization of brain MR images across machines and protocols"
- **URL:** https://www.nature.com/articles/s41598-020-69298-z
- **Date:** 2020-07-23
- **Excerpt:** "Nyul's method provided the highest number of robust first-order features ... Images without any normalization did not generate any robust feature" BUT "No normalization and the WhiteStripe or Z-Score methods led to the same classification performances ... Nyul's method resulted in 5% lower performances on average than no normalization when considering the T1w-gd sequence" and (with absolute binning) "the mean balanced accuracy ... was equal to 0.68 ... without normalization ... reached 0.76 ..., 0.76 ..., and 0.78 ... when the Nyul, WhiteStripe and Z-Score methods were applied."
- **Confidence:** High (peer-reviewed, brain tumor context; knee MRI may differ slightly but conclusion is conservative).

- **Claim:** Practical pipeline order used in competitive/serious MRI DL: [denoise (NLM)] → N4 bias field → resample → Nyul standardization → per-sample z-score. Note Nyul requires a training-set-learned histogram template (must respect CV folds to avoid leakage).
- **Source:** GitHub ZHAN-GAN/MRI-Preprocess
- **URL:** https://github.com/ZHAN-GAN/MRI-Preprocess
- **Date:** 2022-06-07
- **Excerpt:** "1. Denoise, we use the Non-Local Means algorithm ... 2. Bias field correction, we use N4 algorithm ... 3. Resampling ... 4. Standardization, we use nyul histogram matching algorithm ... 5. Normalization, we use z-score algorithm to do normalization sample by sample ... It might take 3 to 5 minutes to process one sample, in which denoise stage and bias field correction take most of time."
- **Confidence:** Medium-High (practitioner repo; key takeaway = N4+NLM are SLOW — minutes per volume, bad for on-the-fly Kaggle pipelines).

- **Claim:** The `intensity-normalization` Python package provides ready CLIs for zscore, WhiteStripe, Nyul, FCM, KDE, RAVEL; only z-score and Nyul are not brain-specific (important for knee!).
- **Source:** intensity-normalization official docs
- **URL:** https://intensity-normalization.readthedocs.io/en/latest/readme.html
- **Date:** accessed 2026-08-10 (v2.2.4)
- **Excerpt:** "All algorithms except Z-score (zscore-normalize) and the Piecewise Linear Histogram Matching (nyul-normalize) are specific to images of the brain."
- **Confidence:** High.

### 2.2 Bias field correction (N4) — cost/benefit
- **Claim:** N4 (Tustison 2010, SimpleITK/ANTs) is the de-facto bias field standard, but DIPY's regression methods correlate 0.90–0.97 with N4 at 10–50× speed; DeepN4 reproduces N4 output at a fraction of inference time. For CNN classification with per-volume z-scoring and aggressive augmentation, N4 is frequently skipped; it matters most when pooling multi-field-strength scanners (1.5T + 3T) — which MRNet-like knee data includes.
- **Source:** DIPY official docs (bias correction guide) + DeepN4 paper (PMC)
- **URL:** https://docs.dipy.org/dev/examples_built/preprocessing/bias_correction_dwi.html ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10680935/
- **Date:** DIPY docs accessed 2026-08-10
- **Excerpt:** "Compared with classical N4: DIPY regression fields correlate at 0.90–0.97 with N4 on DWI data, achieve comparable CoV reductions, and are 10–50× faster. Use classical N4 or DeepN4 when correcting T1/T2 structural images, or when dealing with extreme inhomogeneity from surface-array coils at 7T." And (CSIC paper): "Bias Field Correction is necessary due to the inhomogeneities ... The principal reason for applying this is that our dataset combines different magnetic field strengths (3 T and 1.5T)."
- **Confidence:** Medium-High. Engineering verdict: N4 is plausible but likely overkill for a first baseline; percentile-clip + z-score first, ablate N4 offline (precompute & cache, never on-the-fly).

### 2.3 Resampling to isotropic spacing
- **Claim:** Isotropic resampling can yield large gains for 3D models (+7 Dice points in an nnU-Net MRI study with forced [1,1,1]mm + [192³] patches), but benefit is modality/task-dependent (mixed effect on T2 task). For 2.5D slice-based classification (the typical MRNet-style knee approach), uniform in-plane resize + slice-count normalization (interpolation to fixed N slices) is the cheaper standard.
- **Source:** arXiv 2508.21775 (pancreatic MRI multi-stage fine-tuning)
- **URL:** https://arxiv.org/html/2508.21775v1
- **Date:** 2025 (v1)
- **Excerpt:** "forcing the model to resample all images to a [1, 1, 1] mm spacing and use a large, isotropic [192, 192, 192] patch size yielded a dramatic improvement in performance for Task 1. The mean Tumor Dice score increased by over 7 percentage points ... Interestingly, for the more challenging Task 2, this change had a mixed effect."
- **Confidence:** Medium-High (segmentation not classification; directionally relevant).

- **Claim:** MRNet-style knee pipelines standardize slice count by interpolation (e.g., to 24 slices) or by taking middle slices, rather than full isotropic resampling.
- **Source:** GitHub Elzawawy/MRNet (pattern-recognition course implementation)
- **URL:** https://github.com/Elzawawy/MRNet
- **Date:** 2019-05-18
- **Excerpt:** "Each exam has multiple scans but not all of them has the same number, so we used data interpolation in order to make all exams have 24 scans(slices). Another approach we took was to only take the 3 middle scans(slices) from each exam."
- **Confidence:** Medium (course repo, but reflects common MRNet practice).

### 2.4 Recommended minimal preprocessing for this competition
Synthesis (engineering judgment, Medium-High confidence):
1. Group DICOMs by SeriesInstanceUID; sort by IPP·normal; record plane & sequence from SeriesDescription.
2. Apply modality LUT; check PhotometricInterpretation; cast to float32.
3. Percentile clip (e.g., 0.5–99.5%) per volume → z-score per volume (foreground only if easy) → resize/pad in-plane; normalize slice count by interpolation.
4. Cache resulting arrays (float16 npy / webdataset shards). Defer N4, NLM denoising, and isotropic resampling to ablations — they cost minutes/volume and are usually marginal for CNN classification.

---

## 3. KAGGLE PLATFORM MECHANICS (official docs, accessed 2026-08-10)

### 3.1 Hardware, quotas, session limits
- **Claim:** Sessions: 12h execution for CPU/GPU notebooks, 9h for TPU notebooks; 20 GB auto-saved output disk (/kaggle/working) + extra non-persistent scratchpad; specs: CPU 4 cores/30GB RAM; P100 = 1×Tesla P100, 4 cores, 29GB; T4x2 = 2×Tesla T4, 4 cores, 29GB; TPU v3-8 VM = 96 cores/330GB.
- **Source:** Kaggle official Notebooks documentation
- **URL:** https://www.kaggle.com/docs/notebooks
- **Date:** accessed 2026-08-10
- **Excerpt:** "12 hours execution time for CPU and GPU notebook sessions and 9 hours for TPU notebook sessions / 20 Gigabytes of auto-saved disk space (/kaggle/working) / Additional scratchpad disk space (outside /kaggle/working) that will not be saved outside of the current session / CPU Specifications: 4 CPU cores, 30 Gigabytes of RAM / P100 GPU Specifications: 1 Nvidia Tesla P100 GPU ... / T4 x2 GPU Specifications: 2 Nvidia Tesla T4 GPUs ... / TPU 1VM Specifications: 96 CPU cores, 330 Gigabytes of RAM"
- **Confidence:** High (official).

- **Claim:** Interactive editing sessions idle out after ~20 min; long runs must use "Save & Run All" (top-to-bottom clean run), which itself must finish within 12h (9h TPU).
- **Source:** Kaggle official Notebooks documentation
- **URL:** https://www.kaggle.com/docs/notebooks
- **Date:** accessed 2026-08-10
- **Excerpt:** "you are provided with 20 minutes of idle time for your interactive session ... Save & Run All creates a new session with a completely clean state and runs your notebook from top to bottom ... the entire Notebook must execute within 12 hours (9 hours for TPU notebooks)."
- **Confidence:** High.

- **Claim:** GPU weekly quota is 30 hours/week (P100 or T4x2), TPU ~20h/week; max 2 concurrent batch GPU sessions, 1 TPU session. Kaggle staff confirm 12h is a hard cap regardless of run mode.
- **Source:** "The Kaggle Book" (Bojan Tunguz) PDF + Kaggle Q&A thread #306441 (staff: Dustin)
- **URL:** https://www.alvinang.sg/s/The-Kaggle-Book-Bojan.pdf ; https://www.kaggle.com/questions-and-answers/306441
- **Date:** book ~2024; Q&A 2025-04-12
- **Excerpt (book):** "quotas are set to 30 hours a week for GPUs and 20 hours for TPUs. Further limitations are that you cannot run more than two batch GPU sessions and a single batch TPU session." Excerpt (Kaggle staff): "12h is the maximum runtime ... Yes, 12hrs is the maximum runtime no matter what/how you run it."
- **Confidence:** High for 30h GPU quota (corroborated by multiple 2026 third-party sources); note: no official evidence found of an "L4x" option — docs list only P100 / T4 x2 / TPU v3-8. Verify in the Accelerator dropdown at competition time.

- **Claim:** Colab Pro/Pro+ linking adds 15/30 extra Kaggle GPU hours/week (experimental promotion) on the same Kaggle hardware.
- **Source:** Kaggle official Notebooks documentation
- **URL:** https://www.kaggle.com/docs/notebooks
- **Date:** accessed 2026-08-10
- **Excerpt:** "Colab Pro and Pro+ users will get 15 and 30 hours of extra GPU hours per week respectively on Kaggle ... These extra GPU hours don't consume your Colab compute units. You'll be using the same Kaggle hardware you're accustomed to (CPUs, T4s, P100s, TPUv3-8)."
- **Confidence:** High (official; promo terms may change).

### 3.2 Persisting models / chaining notebooks / datasets
- **Claim:** Up to 20 GB of notebook output auto-saves and can be attached to future notebooks via Input → "Notebook Output Files"; this is the standard way to stage trained model weights. Output can also be converted into a Kaggle Dataset ("New dataset" from the Output tab), and datasets can be created programmatically from inside a notebook with the Kaggle API.
- **Source:** Kaggle official docs + Kaggle forum (staff Dustin, Paul Mooney) + StackOverflow
- **URL:** https://www.kaggle.com/docs/notebooks ; https://www.kaggle.com/product-feedback/162754 ; https://stackoverflow.com/questions/78734994/creating-a-dataset-directly-from-a-kaggle-notebook
- **Date:** 2024-07-11 (SO); accessed 2026-08-10
- **Excerpt (docs):** "Up to 20 GBs of output from a Notebook may be saved to disk in /kaggle/working. This data is saved automatically and you can then reuse that data in any future Notebook ... By chaining Notebooks as data sources in this way, it's possible to build pipelines." Excerpt (staff): "there is an intentional 1000 file limit on user datasets ... you can 'add' those output files to another notebook the same way you do a dataset using the 'Add Data' button." Excerpt (SO): "`!kaggle datasets init -p '{SAVE_FOLDER}' ... !kaggle datasets create -u -p '{SAVE_FOLDER}'`" (with KAGGLE_USERNAME/KAGGLE_KEY from secrets).
- **Confidence:** High.

### 3.3 Offline installs for code competitions
- **Claim:** Two sanctioned mechanisms: (a) Dependency Manager in the notebook editor — enter pip install commands; Kaggle builds a "Dependency Installation Notebook" with wheels attached to your notebook, allowing internet-off submissions; (b) manual wheelhouse: `pip download pkg==X -d ./wheels` (or `pip wheel`), upload as Kaggle Dataset, install offline with `pip install --no-index --find-links=...` (+`--no-deps` where needed).
- **Source:** Kaggle official docs + Kaggle Q&A #567059 and #559498
- **URL:** https://www.kaggle.com/docs/notebooks ; https://www.kaggle.com/questions-and-answers/567059 ; https://www.kaggle.com/questions-and-answers/559498
- **Date:** docs accessed 2026-08-10; Q&A 2025-03-07 / 2024-12-24
- **Excerpt (docs):** "Configure your notebooks to perform offline pip installs using the Dependency Manager editor. Configured notebooks are then able to be submitted to internet disabled competitions ... a Dependency Installation Notebook will be shared with you and automatically attached to your notebook. This notebook contains python wheels and an installation script that is executed before the start of your notebook." Excerpt (Q&A): "!pip download bitsandbytes==0.45.1 accelerate==1.3.0 transformers==4.47.0 peft==0.14.0 huggingface_hub==0.27.1 -d ./wheels ... pip install --no-index --find-links=/kaggle/input/wheels-files/wheels ..."
- **Confidence:** High.

### 3.4 Secrets
- **Claim:** Add-ons → Secrets stores key-value pairs per account; retrieve via `from kaggle_secrets import UserSecretsClient; UserSecretsClient().get_secret("LABEL")`. Only works inside Kaggle notebook environment (not local); secrets don't transfer on fork; service can rate-limit (HTTP 429).
- **Source:** Kaggle CLI docs (official) + Feature Launch post + bug report
- **URL:** https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md ; https://www.kaggle.com/discussions/product-feedback/114053 ; https://www.kaggle.com/discussions/product-feedback/467871
- **Date:** CLI docs current; launch post 2024-10-25
- **Excerpt:** "Use the `UserSecretsClient` from the `kaggle_secrets` package to retrieve your secrets at runtime ... The `kaggle_secrets` package is pre-installed and only functional within the Kaggle notebook execution environment. It will not work when running scripts locally."
- **Confidence:** High.

### 3.5 Docker image pinning & reproducibility
- **Claim:** Notebook versions are tied to a specific Docker image (github.com/Kaggle/docker-python, updated ~every 2 weeks); you can pin "original environment" vs "latest" in Session options — pin for reproducible submissions.
- **Source:** Kaggle official Notebooks documentation
- **URL:** https://www.kaggle.com/docs/notebooks
- **Date:** accessed 2026-08-10
- **Excerpt:** "Every Notebook version you create is associated with a specific Docker image version ... We update the images about every two weeks ... Under the 'Session options' section, look for the selection for 'Environment'. This will enable you to select the original environment that the Notebook was created with or the latest environment."
- **Confidence:** High.

### 3.6 W&B-less logging on Kaggle
- **Claim:** Submission notebooks require internet OFF, so wandb can't sync live; use `WANDB_MODE=offline` (set before `import wandb`), sync later from downloaded output — or just log to CSV/JSON + TensorBoard files in /kaggle/working.
- **Source:** kaggle-wandb-sync article (Zenn)
- **URL:** https://zenn.dev/shogaku/articles/kaggle-wandb-sync-offline-sync?locale=en
- **Date:** 2026-02-22
- **Excerpt:** "Kaggle competition notebooks require internet access to be disabled to be eligible for submission. This means that you cannot send metrics to W&B ... Calling `wandb.log()` will do nothing. ... Make sure to set `WANDB_MODE=offline` before `import wandb`."
- **Confidence:** High.

---

## 4. TYPICAL BASELINE PIPELINE FOR VOLUMETRIC CLASSIFICATION

### 4.1 MONAI transform pipeline
- **Claim:** Canonical MONAI volumetric pipeline: LoadImaged → EnsureChannelFirstd → Spacingd(pixdim) → Orientationd("RAS") → ScaleIntensityRanged(clip) → CropForegroundd → (Rand* augmentations) → ToTensord, wrapped in Compose.
- **Source:** Multiple MONAI pipeline write-ups (consistent with official MONAI tutorials)
- **URL:** https://blog.csdn.net/weixin_34598113/article/details/154299648 (and identical patterns in MONAI tutorials repo)
- **Date:** 2025-11-01
- **Excerpt:** "train_transforms = Compose([ LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')), Orientationd(keys=['image', 'label'], axcodes='RAS'), ScaleIntensityRanged(keys=['image'], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), ToTensord(keys=['image', 'label']) ])"
- **Confidence:** High (pattern matches official MONAI tutorial code). For MRI (no HU), replace fixed a_min/a_max with percentile-based clipping or `NormalizeIntensityd`.

### 4.2 Caching datasets
- **Claim:** MONAI CacheDataset caches pre-random-transform outputs in RAM for up to ~10× training speedup; PersistentDataset persists the same cache to disk/LMDB across runs (ideal for Kaggle: cache once in a prep notebook, save to /kaggle/working or a dataset); SmartCacheDataset replaces a fraction of the cache each epoch when RAM can't hold everything.
- **Source:** MONAI official docs (Modules Overview / fast_model_training_guide)
- **URL:** https://monai.readthedocs.io/en/0.9.0/highlights.html ; https://github.com/Project-MONAI/tutorials/blob/master/acceleration/fast_model_training_guide.md
- **Date:** 2022-06-13 (0.9.0 docs)
- **Excerpt:** "MONAI provides a multi-thread CacheDataset and LMDBDataset to accelerate these transformation steps during training by storing the intermediate outcomes before the first randomized transform in the transform chain. Enabling this feature could potentially give 10x training speedups ... The PersistentDataset is similar to the CacheDataset, where the intermediate cache values are persisted to disk storage or LMDB for rapid retrieval between experimental runs ... or when the entire data set size exceeds available memory."
- **Confidence:** High.

### 4.3 WebDataset / npy caching alternative
- **Claim:** WebDataset stores samples in numbered tar shards (`something-{000000..012345}.tar`) with purely sequential I/O (3–10× faster than random file access), streams from disk/HTTP/pipes, and plugs into PyTorch DataLoader; writing uses ShardWriter with maxcount/maxsize rotation; recommended shard sizes ~100MB (local) to 500MB–1GB (cloud).
- **Source:** webdataset official README (GitHub) + claru.ai guide
- **URL:** https://github.com/webdataset/webdataset/blob/main/README.md ; https://claru.ai/formats/webdataset
- **Date:** README current; guide 2026-08-05
- **Excerpt:** "WebDataset format files are tar files, with two conventions: within each tar file, files that belong together and make up a training sample share the same basename ... the shards of a tar file are numbered like something-000000.tar to something-012345.tar ... The WebDataset representation allows writing purely sequential I/O pipelines for large scale deep learning. This is important for achieving high I/O rates from local storage (3x-10x for local drives compared to random access)."
- **Confidence:** High. On Kaggle, plain per-case `.npy`/`.npz` caches under 20GB output are usually sufficient; WebDataset helps when DICOM counts are huge (random-access I/O bound) — also note the 1000-file limit on user datasets favors tars/zips.

### 4.4 AMP training loop
- **Claim:** Standard PyTorch AMP loop: `scaler = GradScaler()`; forward+loss inside `autocast`; `scaler.scale(loss).backward()`; optional `scaler.unscale_` + grad clip; `scaler.step(optimizer)`; `scaler.update()`. Prefer bf16 (`torch.cuda.is_bf16_supported()`) with no scaler when available; modern API is `torch.amp.GradScaler('cuda')` (torch.cuda.amp.GradScaler deprecated). NOTE: T4/P100 have no bf16 → use fp16+GradScaler on Kaggle GPUs.
- **Source:** mljourney AMP guide + PyTorch deprecation notice (CSDN)
- **URL:** https://mljourney.com/mixed-precision-training-with-pytorch-amp-fp16-bf16-and-gradscaler/ ; https://wenku.csdn.net/answer/28h7tucqzz
- **Date:** 2026-05-17 / 2025-05-28
- **Excerpt:** "with autocast(device_type='cuda', dtype=dtype): outputs = model(inputs); loss = criterion(outputs, targets); scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); scaler.step(optimizer); scaler.update()" and "`torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead."
- **Confidence:** High.

### 4.5 Architecture pattern for knee MRI (MRNet lineage)
- **Claim:** The canonical MRNet approach: per-plane 2D CNN (originally AlexNet features, now ResNet/EfficientNet) over each slice → max-pool/aggregate over slices → per-task sigmoid heads (abnormal/ACL/meniscus), trained per plane (axial/coronal/sagittal) and ensembled; typical hyperparams lr≈1e-5 Adam, ~50 epochs; MRNet baseline AUCs 0.937/0.965/0.847.
- **Source:** GitHub ahmedbesbes/mrnet + MRNet paper summary (thirdeyedata)
- **URL:** https://github.com/ahmedbesbes/mrnet ; https://thirdeyedata.ai/mrnet-deep-learning-assisted-diagnosis-for-knee-magnetic-resonance-imaging/
- **Date:** 2019-04-17 / 2019-01-22
- **Excerpt:** "parser.add_argument('-t', '--task', ... choices=['abnormal', 'acl', 'meniscus']); parser.add_argument('-p', '--plane', ... choices=['sagittal', 'coronal', 'axial']) ... --epochs, default=50 ... --lr, default=1e-5" and "the model achieved AUCs of 0.937 ..., 0.965 ..., and 0.847 ... for abnormality detection, ACL tear detection, and meniscal tear detection respectively."
- **Confidence:** High.

- **Claim:** On volumetric RSNA comps, top solutions typically use 2D CNN + sequence model (BiGRU/LSTM over slice embeddings) and heavy ensembling (7–31 models); windowing of DICOM intensities into 3 "RGB" channels is a winning trick on CT; for MRI the analog is stacking multiple sequences/windows as channels.
- **Source:** PMLR shang25a (RSNA-IHD solutions review) + ynhuhu RSNA repo
- **URL:** https://proceedings.mlr.press/v281/shang25a.html ; https://github.com/ynhuhu/RSNA-Intracranial-Hemorrhage-Detection
- **Date:** 2025-04-22 / 2019-11-15
- **Excerpt:** "Almost all the top solutions rely on 2D convolutional networks and sequential models (Bidirectional GRU or LSTM) to extract intraslice and interslice features, respectively. All the top solutions improve performance by using the ensemble of models, and the number of models varies from 7 to 31." And: "Sequence images on Patient, Study and Series - most sequences were between 24 and 60 images in length ... concat on the deltas between current and previous/next embeddings."
- **Confidence:** High.

---

## 5. EXTERNAL DATA (MRNet / OAI / staging via Kaggle Datasets)

- **Claim:** MRNet = 1,370 knee MRI exams (Stanford), 1,104 abnormal (80.6%), 319 ACL tears (23.3%), 508 meniscal tears (37.1%); axial+coronal+sagittal planes, 256×256, 17–61 slices; official splits 1,130 train / 120 valid / 120 hidden test; requires registration form via Stanford ML Group (competition now closed; ~6GB download).
- **Source:** Stanford ML Group official MRNet page + MDPI MRNet study
- **URL:** https://stanfordmlgroup.github.io/competitions/mrnet/ ; https://www.mdpi.com/2504-4990/3/4/50
- **Date:** accessed 2026-08-10 / 2021-12-16
- **Excerpt:** "The MRNet dataset consists of 1,370 knee MRI exams performed at Stanford University Medical Center. The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports." And: "Each image is 256 × 256, and the number of slices ranges from 17 to 61 ... The dataset contains 1130 trains, 120 valid, and 120 test datasets."
- **Confidence:** High.

- **Claim:** MRNet sequences: coronal T1, coronal T2 FS, sagittal PD, sagittal T2 FS, axial PD FS; GE scanners; 56.6% 3.0T, rest 1.5T. Useful for matching external pretraining data to the competition's sequence mix.
- **Source:** Stanford ML Group official MRNet page
- **URL:** https://stanfordmlgroup.github.io/competitions/mrnet/
- **Date:** accessed 2026-08-10
- **Excerpt:** "Examinations were performed with GE scanners ... coronal T1 weighted, coronal T2 with fat saturation, sagittal proton density (PD) weighted, sagittal T2 with fat saturation, and axial PD weighted with fat saturation. A total of 775 (56.6%) examinations used a 3.0-T magnetic field; the remaining used a 1.5-T magnetic field."
- **Confidence:** High.

- **Claim:** RULES RISK: RSNA Kaggle comps require external data to be "publicly available and equally accessible to use by all participants". MRNet requires an application form (not equally/instantly accessible) and OAI data requires registration/DUA — both are borderline; OAI-ZIB derivative subsets exist publicly on GitLab. Always verify against the 2026 knee comp's specific rules page before use.
- **Source:** RSNA 2024 Lumbar Spine rules page (Kaggle) + OAI-ZIB GitLab + NIAMS OAI page
- **URL:** https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules ; https://gitlab.com/vvr/OActive/osteoarthritis_initiative_zib_dataset/oai_zib_mri_train ; https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative
- **Date:** rules accessed 2026-08-10; NIAMS 2025-06-27
- **Excerpt (rules):** "you will ensure the External Data is publicly available and equally accessible to use by all participants of the Competition for purposes of the ..." Excerpt (NIAMS): "Data and images have been publicly released through the OAI Online website since 2004 ... As of March 23, 2016, there were 4,114 registered users of OAI Online."
- **Confidence:** Medium-High (typical RSNA wording; 2026 comp rules pending — treat MRNet/OAI as "verify before use").

- **Claim:** Typical staging: download external data in an internet-ON notebook (or locally), preprocess/convert, then either (a) save to /kaggle/working and create a Kaggle Dataset from notebook output, or (b) `kaggle datasets create -p folder` via API with metadata JSON; attach the dataset to training/submission notebooks. External libraries travel the same way as wheel datasets.
- **Source:** Kaggle docs + koilerplate repo + ARC Prize 2024 winner repo (da-fr)
- **URL:** https://github.com/AScholan/koilerplate ; https://github.com/da-fr/arc-prize-2024/
- **Date:** 2021-05-23 / 2024-11-12
- **Excerpt (koilerplate):** "it is a non-trivial matter using a library that's not included in the standard notebook image. You need to build yourself a custom dataset containing all the information that's needed for pip to load them into your notebook whilst it's running without an internet connection." Excerpt (ARC winner): "As the competition did not allow internet access, this notebook uses an offline dataset containing various python wheels (which can be created by executing the notebook unsloth-download-2024-9-post4.ipynb and creating a dataset from its output)."
- **Confidence:** High.

---

## 6. AGENTIC CODING PLATFORM (Kimi Code / Claude Code-class) ON A KAGGLE PROJECT

- **Claim:** Structured agentic workflow beats ad-hoc prompting: maintain a persistent project-instructions file (CLAUDE.md-style: environment facts, do-not-touch paths, pinned versions), treat context as a scarce resource, use plan-mode artifacts (markdown plans with checkboxes/status), and keep an issue/memory tracker for the agent.
- **Source:** LevelUp (12 patterns from shanraisshan/claude-code-best-practice, GitHub Trending #1, input from Claude Code creator Boris Cherny) + devas.life note-driven workflow
- **URL:** https://levelup.gitconnected.com/claude-code-best-practices-12-patterns-agentic-engineers-use-65264e3eb919 ; https://www.devas.life/note-driven-agentic-coding-workflow-using-claude-code-and-inkdrop/
- **Date:** 2026-04-15 / 2026-01-29
- **Excerpt:** "Every session started fresh — no structure, no constraints, no reusable configuration ... I was vibe coding ... 'from vibe coding to agentic engineering.' ... 69 actionable tips across 11 categories" and (note-driven) "AI agents take notes for themselves because they forget things due to the context window size, just like us."
- **Confidence:** Medium-High (practitioner consensus, 2026 sources).

- **Claim:** Kaggle-specific agent scaffolding exists: git-tracked `project.yml` for competition metadata (name, platform, is_code_competition) + git-ignored `.env` for KAGGLE_USERNAME/KAGGLE_KEY/WANDB_API_KEY + task runner (Taskfile) — a template for config-driven agentic competition work (ExpAgent, built for Claude Code).
- **Source:** GitHub osushinekotan/ExpAgent
- **URL:** https://github.com/osushinekotan/ExpAgent
- **Date:** 2026-03-14
- **Excerpt:** "project.yml (git-tracked) — Competition settings and metadata: competition_name: 'my-competition'; competition_platform: kaggle ... is_code_competition: false ... .env (git-ignored) — Secrets and GCP settings ... KAGGLE_USERNAME= KAGGLE_KEY= ... WANDB_API_KEY="
- **Confidence:** Medium-High.

- **Claim:** Winner repos separate: input/ (raw), intermediate_output/ (preprocessed pkl per model config), models/ (one folder per experiment: bin/train.sh + config), bin/ (pipeline shell scripts: preprocess.sh, run.sh), output/ (submission). Experiments = config-driven variants of a base model (backbone/size/aug changed per folder), 5-fold split precomputed and shared.
- **Source:** shimacos37/kaggle-rsna-2019-10th-solution (10th place, RSNA-IHD) directory tree
- **URL:** https://github.com/shimacos37/kaggle-rsna-2019-10th-solution
- **Date:** 2019-11-24
- **Excerpt:** "bin : main file ... preprocess.sh ... run.sh ... models/base_cnn : base CNN models ... ricky_se_resnext101_mixup : change backbone of model_base to se_resnext101 and use mixup ... intermediate_output : preprocessed files and intermediate outputs ... train_folds.pkl : splitted train file ... # 5-fold aplit / delete some noise / metadata extraction from dicom: sh bin/preprocess.sh"
- **Confidence:** High.

- **Claim:** Reproducibility practices in winner repos: pin hardware+env in README (docker build script), store per-fold weights/logs/predictions under each model dir, keep training code locally-executable and Kaggle notebooks as thin wrappers, publish the exact wheel/model datasets used by the offline submission notebook.
- **Source:** shimacos37 repo + da-fr/arc-prize-2024 repo
- **URL:** https://github.com/shimacos37/kaggle-rsna-2019-10th-solution ; https://github.com/da-fr/arc-prize-2024/
- **Date:** 2019-11-24 / 2024-11-12
- **Excerpt:** "# prepare docker image: sh bin/build_image.sh" and "Under training_code, you can find our locally executable code ... Under kaggle_notebooks, you can find our notebooks for kaggle ... this notebook uses an offline dataset containing various python wheels ... also available directly on kaggle."
- **Confidence:** High.

### 6.x Recommended repo skeleton for this competition (synthesis)
```
repo/
  configs/            # yaml per experiment (backbone, plane, slices, aug, lr)
  src/dicom_io/       # series grouping, IPP sorting, LUT, photometric fix
  src/preprocess/     # clip+zscore, resize, slice-count normalization (cache to npy)
  src/datasets/       # MONAI/torch datasets, webdataset writers
  src/models/         # 2.5D CNN + sequence aggregator
  src/train.py        # AMP loop, fold from config, CSV/JSON logging
  src/infer.py
  notebooks/          # thin Kaggle wrappers (prep / train / submit)
  kaggle/             # scripts: make wheels dataset, make weights dataset, push via API
  .env.example        # KAGGLE_USERNAME/KAGGLE_KEY placeholders
  CLAUDE.md / AGENTS.md  # environment facts + constraints for the coding agent
```

---

## KEY UNCERTAINTIES / TO VERIFY AT COMPETITION TIME
1. 2026 comp rules on external data & pretraining (MRNet needs application → likely NOT "equally accessible"; check wording).
2. Whether Kaggle offers L4x or other new accelerators by Aug 2026 — official docs still list P100 / T4x2 / TPU v3-8; third-party 2026 sources corroborate 30h/week GPU, 12h session cap.
3. Submission format (notebook vs. API) and runtime limit for this specific code competition.
4. SimpleITK version on current Kaggle docker image (direction-matrix change in 2.4.0) — pin environment.
