# 5. Engineering and Execution Roadmap

This chapter converts the research findings of Chapters 1–4 into a build plan for a solo practitioner working with an agentic coding platform (Kimi Code) plus Kaggle Notebooks. It covers DICOM engineering (§5.1), Kaggle platform mechanics (§5.2), agentic workflow discipline (§5.3), a ten-week plan from 2026-08-10 to the 2026-10-22 final deadline (§5.4), and a risk register (§5.5).

## 5.1 DICOM Engineering Essentials

The competition data is 819,640 DICOM files / 569.76 GB, one slice per file, organized as `<StudyUID>/<SeriesUID>/<SOPUID>.dcm`, in a mix of four transfer syntaxes (uncompressed Explicit VR Little Endian, Implicit VR Little Endian, JPEG Lossless, JPEG 2000), with every file stripped to an allowlisted set of 86 metadata tags — so rich protocol metadata is simply absent and must not be relied upon.[^1^] Series typically contain 20–45 slices (median 30), with a long tail to a few hundred.[^1^]

**Slice ordering.** Never trust InstanceNumber alone. Sort slices by projecting `ImagePositionPatient` (0020,0032) onto the slice normal computed as `cross(row, col)` from `ImageOrientationPatient` (0020,0037); use InstanceNumber only as a fallback, and derive slice spacing from the actual first/last slice positions rather than the `SliceThickness` tag.[^2^] Watch for multi-frame/enhanced DICOMs, where spacing and position live inside `PerFrameFunctionalGroupsSequence`/`SharedFunctionalGroupsSequence` and naive `ds.PixelSpacing` access fails.[^2^]

**Pixel values.** `ds.pixel_array` returns raw stored values. Apply `apply_modality_lut` (rescale slope/intercept) *before* `apply_voi_lut` (windowing), per the official pydicom docs.[^3^] If `PhotometricInterpretation` is `MONOCHROME1`, invert (`arr = arr.max() - arr`); MRI is usually MONOCHROME2, but check per-series.[^4^]

**Reading at scale.** Two viable toolchains: (a) pydicom (+ pylibjpeg for the JPEG transfer syntaxes) with the manual sorting above; (b) SimpleITK `ImageSeriesReader.GetGDCMSeriesFileNames`, which returns GDCM-sorted filenames and enumerates series via `GetGDCMSeriesIDs`.[^5^] **Pin your SimpleITK version**: 2.4.0 changed DICOM series direction handling (Z-component sign flip vs 2.3.1), which silently flips volumes if train and inference environments differ.[^6^] For NIfTI conversion, dcm2niix + nibabel is the de-facto standard and handles JPEG Lossless/JPEG 2000, but it cannot emit fields the anonymized header lacks.[^7^]

```python
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut

def read_series(paths):
    """Read one single-frame MRI series into a geometrically sorted float32 volume."""
    slices = [pydicom.dcmread(p) for p in paths]
    iop = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])              # slice-axis unit vector
    try:
        slices.sort(key=lambda ds: float(np.dot(
            np.asarray(ds.ImagePositionPatient, dtype=float), normal)))
    except AttributeError:                            # tag anonymized away
        slices.sort(key=lambda ds: int(ds.InstanceNumber))
    out = []
    for ds in slices:
        arr = apply_modality_lut(ds.pixel_array, ds)  # rescale FIRST
        arr = apply_voi_lut(arr, ds)                  # windowing SECOND
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr                     # invert
        out.append(arr.astype(np.float32))
    return np.stack(out)                              # (slices, rows, cols)
```

**Preprocessing.** For CNN classification, percentile-clip (e.g., 0.5–99.5%) plus per-volume z-score is sufficient — the peer-reviewed comparison of z-score, WhiteStripe, and Nyul found all three performed similarly for classification accuracy.[^8^] N4 bias-field correction costs minutes per volume; defer it to an offline ablation (precompute and cache, never on-the-fly).[^9^] Normalize slice count by interpolation to a fixed length (24–96 slices per series), the standard MRNet-style practice, instead of full isotropic resampling.[^10^] At 569.76 GB raw, cache preprocessed volumes once as float16 `.npy` files or tar shards; MONAI `CacheDataset`/`PersistentDataset` give up to ~10× training speedup by caching pre-random-transform outputs in RAM or on disk.[^11^] WebDataset-style tar shards (`{000000..N}.tar`) add 3–10× I/O throughput over random file access and conveniently sidestep Kaggle's 1000-file dataset limit.[^12^]

| # | DICOM processing checklist item | Tool / method | Why it matters here |
|---|--------------------------------|---------------|---------------------|
| 1 | Group files by SeriesInstanceUID (0020,000E) | pydicom header scan | Prevents mixing planes/sequences into one volume |
| 2 | Sort by IPP · cross(row, col) of IOP; InstanceNumber fallback | numpy + pydicom | InstanceNumber ordering is unreliable across vendors[^2^] |
| 3 | Spacing from first/last positions, not SliceThickness | numpy | Anonymized/irregular spacing[^2^] |
| 4 | Detect enhanced multi-frame DICOMs | functional-groups sequences | `ds.PixelSpacing` raises on these[^2^] |
| 5 | `apply_modality_lut` → `apply_voi_lut` (in that order) | pydicom.pixels | Raw stored values are not display/analysis values[^3^] |
| 6 | Invert MONOCHROME1 | `arr.max() - arr` | Inverted contrast otherwise[^4^] |
| 7 | Smoke-test all 4 transfer syntaxes | pydicom + pylibjpeg / gdcm | JPEG Lossless + JPEG 2000 present in data[^1^] |
| 8 | Assume only the 86 allowlisted tags exist | defensive `getattr` | Anonymization removed protocol metadata[^1^][^7^] |
| 9 | Percentile-clip + per-volume z-score | numpy | As good as Nyul/WhiteStripe for CNN classification[^8^] |
| 10 | Defer N4 bias field to cached ablation | SimpleITK/ANTs offline | Minutes per volume — kills on-the-fly pipelines[^9^] |
| 11 | Interpolate to fixed slice count (24–96) | `np.interp` over z | MRNet-standard; batching needs fixed shapes[^10^] |
| 12 | Cache volumes as float16 npy/tar shards | MONAI PersistentDataset / webdataset | 569.76 GB raw; 1000-file dataset limit[^11^][^12^] |
| 13 | Pin SimpleITK version in both train and submit envs | requirements pin | 2.4.0 flipped Z direction sign[^6^] |

## 5.2 Kaggle Platform Mechanics

Everything below is from Kaggle's official documentation, accessed 2026-08-10.[^13^] Weekly GPU quota is 30 hours (P100 or T4×2), corroborated by Kaggle staff.[^14^] The competition's code requirements cap submission notebooks at **9 h runtime (CPU or GPU), internet disabled**, output `submission.csv`.[^15^] Trained artifacts move between notebooks as auto-saved output (20 GB) or Kaggle Datasets; datasets have an intentional **1000-file limit**, so package weights and caches as tar/zip archives.[^16^] Offline dependencies install from a wheel dataset via `pip install --no-index --find-links=...`, or via Kaggle's Dependency Manager, which builds a wheel-bearing installation notebook for internet-off submissions.[^17^] Note for the efficiency track: RuntimeSeconds is the full notebook wall time — package installs, model loading, **and DICOM reading** all count.[^18^]

```bash
# Local (or internet-on notebook): build the wheelhouse
pip download monai==1.4.0 pydicom pylibjpeg -d ./wheels
# Upload ./wheels as a Kaggle Dataset, then in the offline submission notebook:
pip install --no-index --find-links=/kaggle/input/my-wheels/wheels monai pydicom pylibjpeg
```

| Kaggle resource / limit | Value | Source |
|--------------------------|-------|--------|
| GPU options | 1× Tesla P100 (16 GB) or 2× Tesla T4; 4 CPU cores, 29 GB RAM | Kaggle docs[^13^] |
| GPU quota | 30 h/week (TPU ~20 h/week); max 2 concurrent batch GPU sessions | Kaggle Book + staff Q&A[^14^] |
| Session runtime | 12 h CPU/GPU notebooks; 9 h TPU | Kaggle docs[^13^] |
| Submission cap | ≤ 9 h CPU or GPU notebook; internet off | Code requirements[^15^] |
| Auto-saved output | 20 GB in /kaggle/working; reusable as input to later notebooks | Kaggle docs[^13^] |
| Dataset file limit | 1000 files per user dataset → ship tar shards | Kaggle staff[^16^] |
| Interactive idle timeout | ~20 min; long runs need Save & Run All (top-to-bottom, ≤12 h) | Kaggle docs[^13^] |
| Docker image | Updated ~every 2 weeks; pin "original environment" in Session options | Kaggle docs[^13^] |
| Extra GPU hours | Colab Pro/Pro+ promo: +15/+30 h/week on the same hardware | Kaggle docs[^13^] |
| Efficiency runtime | Full wall time incl. installs, model load, DICOM decode | Kaggle staff[^18^] |

One mixed-precision note: P100/T4 have no bf16, so use fp16 with `torch.amp.GradScaler('cuda')` (the `torch.cuda.amp` API is deprecated).[^19^] Budget arithmetic: 10 weeks × 30 GPU-h ≈ 300 GPU-hours total — treat this as the hard planning currency in §5.4.

## 5.3 Working Effectively with an Agentic Coding Platform (Kimi Code)

Agentic coding pays off only with structure; practitioners who moved "from vibe coding to agentic engineering" converge on the same patterns: a persistent project-instructions file, plan-mode artifacts with checkbox status, and treating context as a scarce resource.[^20^] Concretely for this project:

1. **Maintain `AGENTS.md` at repo root** with environment facts the agent cannot re-derive: Kaggle GPU/CPU specs, the 30 h/week quota, pinned versions (SimpleITK pin with the 2.4.0 warning[^6^]), the offline-install mechanism, the 86-tag allowlist, and "do-not-touch" paths (gold labels, raw DICOM cache). A Kaggle-specific precedent is ExpAgent: git-tracked `project.yml` for competition metadata plus git-ignored `.env` for `KAGGLE_USERNAME`/`KAGGLE_KEY`.[^21^]
2. **Config-driven experiments.** Winner repos structure work as one folder per experiment with a config (backbone, planes, slices, augmentation, lr), a precomputed shared 5-fold split, and shell entry points (`preprocess.sh`, `run.sh`).[^22^] The agent edits YAML, not training code.
3. **Thin Kaggle notebooks over local code.** Keep training/inference code locally executable in `src/`; Kaggle notebooks are wrappers that pip-install the wheel dataset, attach weight/cache datasets, and call `src/train.py` / `src/infer.py` — the pattern used by the ARC Prize 2024 winner for offline submissions.[^23^]
4. **Reproducibility contract:** fixed seeds logged per config, out-of-fold (OOF) prediction artifacts saved per experiment, per-fold weights and metrics under each model directory.[^22^] Instruct the agent to never overwrite OOF artifacts — they are the ensemble's raw material.

```
rsna-knee/
  AGENTS.md             # env facts, pinned versions, quotas, do-not-touch paths
  project.yml           # competition metadata (git-tracked)
  .env.example          # KAGGLE_USERNAME / KAGGLE_KEY placeholders (git-ignored real .env)
  configs/              # one YAML per experiment: backbone, planes, n_slices, aug, lr, fold
  input/                # raw CSVs + cache manifests (DICOMs stay on Kaggle)
  src/
    dicom_io/           # series grouping, IPP sort, LUT order, photometric fix
    preprocess/         # clip + z-score, resize, slice interpolation -> npy/tar cache
    datasets/           # MONAI CacheDataset/PersistentDataset, shard writers
    models/             # 2.5D CNN encoder + attention-MIL / BiLSTM slice aggregator
    train.py            # fp16 AMP loop, fold + seed from config, CSV/JSON logging
    infer.py            # study-level aggregation, submission.csv writer
  models/               # per-experiment: weights/, oof/, metrics.json
  notebooks/            # thin Kaggle wrappers: prep / train / submit
  kaggle/               # scripts: build wheels dataset, push weights dataset via API
```

## 5.4 Ten-Week Execution Plan

Anchors from the official timeline (all 11:59 PM UTC): **entry/team-merger deadline Oct 15; final submission Oct 22; winners' materials (code, weights as a public dataset, video, method description) Nov 5.**[^24^] The plan below assumes ~30 GPU-hours/week and reflects two strategic facts established earlier: label quality is the largest controllable lever, and commercial LLM APIs are explicitly permitted for report label extraction (host ruling, 2026-08-09).[^25^]

| Week | Dates (2026) | Focus | Deliverables / exit criteria | GPU-h budget |
|------|--------------|-------|------------------------------|--------------|
| W1 | Aug 10–16 | EDA + DICOM pipeline | §5.1 checklist implemented; per-series volumes cached as shards; decode tested on all 4 transfer syntaxes; first dry-run submission (all-0.5 / trivial baseline) to validate the notebook path | ~10 |
| W2 | Aug 17–23 | Label extraction v1 + evaluation harness | Multi-LLM ensemble extractor (multilingual-aware); 58-gold evaluation harness reporting macro AUC per extractor variant — recall LLM labels already beat regex 0.8780 vs 0.8136 against the gold studies, and ~25.4% of cells are "not addressed" (mask, don't zero-fill)[^26^] | ~15 |
| W3 | Aug 24–30 | Baseline image model | MRNet-style 2.5D CNN + gated attention-MIL per plane/sequence; scanner-grouped 5-fold CV running; first real submission | ~30 |
| W4 | Aug 31–Sep 6 | Baseline iteration | Fix data bugs found by CV-vs-LB deltas; augmentation set validated (label-aware hflip, §5.5); OOF artifact store complete | ~30 |
| W5 | Sep 7–13 | Localization stage | ROI/meniscus localizer feeding cropped classification; measure delta on gold + grouped CV | ~30 |
| W6 | Sep 14–20 | Architecture iteration | 1–2 alternatives (3D-ResNet w/ MedicalNet init, BiLSTM aggregator); keep only grouped-CV winners | ~30 |
| W7 | Sep 21–27 | Labels v2 | Refined extractor prompts, targeted gap-filling (never blanket imputation), soft/graded targets; retrain best config | ~30 |
| W8 | Sep 28–Oct 4 | Ensembling + CV hardening | Fold × seed × backbone ensemble; OOF-weighted stacking; final CV vs LB correlation report | ~30 |
| W9 | Oct 5–11 | Efficiency-track candidate | Distill ensemble into small student; fp16 + torch.compile; measure notebook wall time; freeze both candidate pipelines | ~35 |
| W10 | Oct 12–22 | Buffer + submission selection | **Accept rules before Oct 15**; end-to-end dry runs under 9 h; pick 2 final submissions (best-CV ensemble + lean efficient one) by grouped CV + gold, not LB decimals; submit by Oct 22 | ~25 |

Total ≈ 265 GPU-hours planned against ~300 available — keep the ~35 h reserve for reruns and deadline-week failures.

## 5.5 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Silver-label noise caps the image model (reports are a noisy source; image-derived labels are authoritative) | High | High | Multi-LLM ensemble extraction; evaluate every label variant on the 58 gold studies; mask "not addressed" cells; soft targets[^26^] |
| Site/scanner shift across 16 institutions (265 scanner fingerprints; metadata memorization does not transfer) | High | High | Scanner-grouped CV as the primary selection metric; per-volume z-score + aggressive intensity augmentation[^27^] |
| Public LB shake-up: public = ~30% of ~1,300 test studies, prevalence not matched across splits | Medium-High | High | Select finals on grouped CV + gold, not LB; two diverse final submissions; conservative ensembling[^28^] |
| External-data ruling pending: MRNet/OAI are gated click-through datasets, borderline under "equally accessible" wording | Medium | Medium | Treat as unusable until the host rules; clone architectures (MRNet/CoPAS) rather than data; monitor the rules thread[^29^] |
| Compute budget overrun (30 GPU-h/week; 12 h session cap) | Medium | High | Cache preprocessing once (§5.1); iterate on small models; weekly GPU-h ledger; queue long runs via Save & Run All[^14^] |
| TensorRT/INT8 backfire on Kaggle T4 (~147× slower than fp16 when TensorRT libs absent) | Medium | Medium | Efficiency track uses PyTorch fp16/AMP + torch.compile; vendor TensorRT libs or build engines in-notebook only if benchmarked[^30^] |
| Horizontal-flip laterality trap: hflip swaps medial↔lateral labels | Medium | High | Label-aware flip (swap Medial↔Lateral Meniscus and Medial↔Lateral OA) or disable hflip; verify flip direction against ImageOrientationPatient — the trick was worth ~0.01 AUC in RSNA 2025 when done correctly[^31^] |
| Non-Latin report long tail (Greek 7.3%, Cyrillic 5.0%): keyword/regex pipelines silently return confident negatives | High | Medium | Multilingual LLM extractor; script-detection audit of extractor output; no regex-only fallback[^32^] |
| Submission notebook fails or exceeds 9 h at deadline | Medium | High | Dry-run the full notebook weekly from W8; per-study try/except with fallback predictions; pin Docker environment[^13^][^15^] |
| SimpleITK 2.4.0 direction-matrix change flips volumes between train and inference | Low | Medium | Pin SimpleITK in both environments; volume-orientation smoke test in the submission notebook[^6^] |

## Sources

[^1^]: Kaggle Data page, RSNA Knee Abnormality Detection — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data (accessed 2026-08-10)
[^2^]: 3D Slicer community discourse, "DICOM to voxel Python" worked code example — https://discourse.slicer.org/t/dicom-to-voxel-python-landmark-placement-issue-for-dl-dataset/44366 (2025-09-05)
[^3^]: pydicom official API docs, apply_voi_lut / apply_modality_lut — https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html (accessed 2026-08-10)
[^4^]: K-Dense-AI scientific-agent-skills, pydicom SKILL.md — https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/pydicom/SKILL.md (2025-10-19)
[^5^]: SimpleITK official Doxygen, ImageSeriesReader — https://simpleitk.org/doxygen/v2_3/html/classitk_1_1simple_1_1ImageSeriesReader.html (2023-09-12)
[^6^]: SimpleITK GitHub issue #2214 (2.4.0 direction change) — https://github.com/SimpleITK/SimpleITK/issues/2214 (2025-01-08)
[^7^]: dcm2niix BIDS README (rordenlab) — https://github.com/rordenlab/dcm2niix/blob/master/BIDS/README.md (accessed 2026-08-10)
[^8^]: Nature Scientific Reports, "Standardization of brain MR images across machines and protocols" — https://www.nature.com/articles/s41598-020-69298-z (2020-07-23)
[^9^]: DIPY official docs, bias correction guide (N4 cost / fast alternatives) — https://docs.dipy.org/dev/examples_built/preprocessing/bias_correction_dwi.html (accessed 2026-08-10)
[^10^]: GitHub Elzawawy/MRNet (slice-count interpolation to 24) — https://github.com/Elzawawy/MRNet (2019-05-18)
[^11^]: MONAI official docs, Modules Overview + fast model training guide (CacheDataset/PersistentDataset) — https://monai.readthedocs.io/en/0.9.0/highlights.html (2022-06-13)
[^12^]: webdataset official README — https://github.com/webdataset/webdataset/blob/main/README.md (accessed 2026-08-10)
[^13^]: Kaggle official Notebooks documentation — https://www.kaggle.com/docs/notebooks (accessed 2026-08-10)
[^14^]: The Kaggle Book (Bojan Tunguz) + Kaggle Q&A #306441 (staff confirmation of quotas/12 h cap) — https://www.alvinang.sg/s/The-Kaggle-Book-Bojan.pdf ; https://www.kaggle.com/questions-and-answers/306441 (Q&A 2025-04-12)
[^15^]: Kaggle Overview, Code Requirements — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements (accessed 2026-08-10)
[^16^]: Kaggle Product Feedback #162754 (staff: 1000-file dataset limit; chaining notebook output) — https://www.kaggle.com/product-feedback/162754 (accessed 2026-08-10)
[^17^]: Kaggle Q&A #567059 (offline wheel installs) — https://www.kaggle.com/questions-and-answers/567059 (2025-03-07)
[^18^]: Kaggle discussion 733475, Ryan Holbrook (Kaggle Staff) on RuntimeSeconds — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475 (2026-08-07)
[^19^]: mljourney, "Mixed Precision Training with PyTorch AMP" (fp16 vs bf16; GradScaler API) — https://mljourney.com/mixed-precision-training-with-pytorch-amp-fp16-bf16-and-gradscaler/ (2026-05-17)
[^20^]: LevelUp, "Claude Code Best Practices: 12 Patterns Agentic Engineers Use" — https://levelup.gitconnected.com/claude-code-best-practices-12-patterns-agentic-engineers-use-65264e3eb919 (2026-04-15)
[^21^]: GitHub osushinekotan/ExpAgent (Kaggle agent scaffolding: project.yml + .env) — https://github.com/osushinekotan/ExpAgent (2026-03-14)
[^22^]: GitHub shimacos37/kaggle-rsna-2019-10th-solution (top-10 solution repo layout) — https://github.com/shimacos37/kaggle-rsna-2019-10th-solution (2019-11-24)
[^23^]: GitHub da-fr/arc-prize-2024 (thin Kaggle notebooks + offline wheel datasets) — https://github.com/da-fr/arc-prize-2024/ (2024-11-12)
[^24^]: Kaggle Overview, Timeline — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline (accessed 2026-08-10)
[^25^]: Kaggle discussion 733965, "Use of Commercially Hosted LLMs" (host ruling, Po-Hao Chen) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965 (2026-08-09)
[^26^]: Kaggle discussion 733932 (stevenleehans: LLM vs regex labels; "not addressed" cells) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932 (2026-08-09)
[^27^]: Kaggle discussion 733517 (Oleksii Zhukov: metadata shortcut probe; scanner-grouped CV) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517 (2026-08-07)
[^28^]: Kaggle Leaderboard page (public = ~30% of test data) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard (accessed 2026-08-10)
[^29^]: RSNA 2024 Lumbar Spine rules page (external-data wording precedent) — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules (accessed 2026-08-10)
[^30^]: arXiv:2607.08241, "Guidance-Aware Quantization" (TensorRT libs absent on Kaggle T4; INT8-ORT ~147× slower) — https://arxiv.org/html/2607.08241v1 (2026-07-09)
[^31^]: Kaggle writeup, "5th place solution with code", RSNA Intracranial Aneurysm Detection (hflip with left/right label swap) — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution (2025-10-15)
[^32^]: Kaggle discussion 734055 (maximo lorenzo y losada: script distribution; empty ≠ 0; silent regex failure) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055 (2026-08-10)
