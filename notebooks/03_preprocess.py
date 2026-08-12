"""
Preprocess DICOM volumes into cached .npy files.

Run on Kaggle with internet ON (for pip install of dependencies).
Produces: cached volumes in /kaggle/working/volumes/ (float16 .npy).

This is the most time-consuming step. With 4,407 studies × ~5.5 series
× ~30 slices, expect 2-4 hours on Kaggle CPU.

Usage (Kaggle notebook):
    exec(open("notebooks/03_preprocess.py").read())
"""

import logging
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WORK_DIR = Path("/kaggle/working")
VOLUME_DIR = WORK_DIR / "volumes"
VOLUME_DIR.mkdir(exist_ok=True)

TRAIN_SERIES_DIR = DATA_ROOT / "train_series"

# ── Config ───────────────────────────────────────────────────────────
TARGET_SIZE = (224, 224)
NUM_SLICES = 24          # interpolated slices per series
IN_CHANNELS = 3           # 2.5D: adjacent slices as channels
CLIP_PERCENTILES = (0.5, 99.5)

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# ── Install dependencies (internet ON) ──────────────────────────────
logger.info("Installing dependencies...")
import subprocess
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg",
], check=True)

import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut

# ── Load metadata ────────────────────────────────────────────────────
logger.info("Loading metadata...")
train_csv = pd.read_csv(WORK_DIR / "train_parsed.csv")
train_series = pd.read_csv(DATA_ROOT / "train_series.csv")
silver_labels = pd.read_csv(WORK_DIR / "silver_labels.csv")

# Build series lookup: StudyInstanceUID -> list of (SeriesInstanceUID, plane, fluid_sensitive, fat_suppression)
series_lookup = {}
for _, row in train_series.iterrows():
    uid = row["StudyInstanceUID"]
    if uid not in series_lookup:
        series_lookup[uid] = []
    series_lookup[uid].append({
        "series_uid": row["SeriesInstanceUID"],
        "plane": row.get("Anatomical_Plane", "unknown"),
        "fluid_sensitive": row.get("Fluid_Sensitive", 0),
        "fat_suppression": row.get("Fat_Suppression", 0),
    })

# ── DICOM reading functions (inline for Kaggle compatibility) ───────
def read_dicom_volume(paths, target_slices=None):
    """Read a DICOM series into a sorted float32 volume."""
    slices = [pydicom.dcmread(str(p)) for p in paths]

    # Sort by IPP · cross(row, col) of IOP
    first = slices[0]
    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        iop = np.asarray(iop, dtype=np.float64)
        normal = np.cross(iop[:3], iop[3:])
        def sort_key(ds):
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))
            return float(getattr(ds, "InstanceNumber", 0))
        try:
            slices = sorted(slices, key=sort_key)
        except (TypeError, ValueError):
            slices = sorted(slices, key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
    else:
        slices = sorted(slices, key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))

    # Extract pixels
    volume = []
    for ds in slices:
        arr = apply_modality_lut(ds.pixel_array, ds)
        arr = apply_voi_lut(arr, ds)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        volume.append(arr.astype(np.float32))

    volume = np.stack(volume)

    # Interpolate to target slice count
    if target_slices and volume.shape[0] != target_slices:
        old_idx = np.linspace(0, volume.shape[0] - 1, volume.shape[0])
        new_idx = np.linspace(0, volume.shape[0] - 1, target_slices)
        flat = volume.reshape(volume.shape[0], -1)
        result = np.stack([
            np.interp(new_idx, old_idx, flat[:, i])
            for i in range(flat.shape[1])
        ], axis=1)
        volume = result.reshape(target_slices, *volume.shape[1:]).astype(np.float32)

    return volume


def normalize_volume(volume, clip_percentiles=(0.5, 99.5)):
    """Percentile-clip and z-score normalize."""
    lo, hi = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    mean, std = volume.mean(), volume.std()
    if std < 1e-8:
        std = 1.0
    return ((volume - mean) / std).astype(np.float32)


def resize_volume(volume, target_size=(224, 224)):
    """Resize each slice."""
    slices, h, w = volume.shape
    th, tw = target_size
    if (h, w) == (th, tw):
        return volume
    row_idx = (np.arange(th) * (h - 1) / (th - 1)).astype(int)
    col_idx = (np.arange(tw) * (w - 1) / (tw - 1)).astype(int)
    return volume[:, row_idx][:, :, col_idx].astype(np.float32)


# ── Process studies ──────────────────────────────────────────────────
logger.info("Processing studies...")

# Priority: fluid-sensitive fat-suppressed series (T2-FS/PD-FS workhorses)
# For each study, we select the best series per plane

study_ids = silver_labels["StudyInstanceUID"].tolist()
total = len(study_ids)
processed = 0
errors = []

t0 = time.time()

for idx, study_uid in enumerate(study_ids):
    output_path = VOLUME_DIR / f"{study_uid}.npy"

    if output_path.exists():
        processed += 1
        continue

    study_dir = TRAIN_SERIES_DIR / study_uid
    if not study_dir.exists():
        errors.append({"study_uid": study_uid, "error": "directory not found"})
        continue

    series_list = series_lookup.get(study_uid, [])

    # Select best series: prefer fluid-sensitive + fat-suppressed, sagittal plane
    # Score each series
    best_series = None
    best_score = -1

    for s in series_list:
        score = 0
        if s["fluid_sensitive"] == 1:
            score += 2
        if s["fat_suppression"] == 1:
            score += 1
        if s["plane"] == "Sagittal":
            score += 1

        if score > best_score:
            best_score = score
            best_series = s

    if best_series is None:
        # Fallback: use first available series
        series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
        if not series_dirs:
            errors.append({"study_uid": study_uid, "error": "no series found"})
            continue
        best_series = {"series_uid": series_dirs[0].name}

    # Read and preprocess
    series_dir = study_dir / best_series["series_uid"]
    dcm_files = sorted(series_dir.glob("*.dcm"))

    if not dcm_files:
        errors.append({"study_uid": study_uid, "error": "no .dcm files"})
        continue

    try:
        volume = read_dicom_volume(dcm_files, target_slices=NUM_SLICES)
        volume = normalize_volume(volume, CLIP_PERCENTILES)
        volume = resize_volume(volume, TARGET_SIZE)

        # Save as float16 to save space
        np.save(output_path, volume.astype(np.float16))
        processed += 1

    except Exception as e:
        errors.append({"study_uid": study_uid, "error": str(e)})
        logger.error(f"Error processing {study_uid}: {e}")

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = processed / max(elapsed, 1)
        remaining = (total - idx - 1) / max(rate, 0.001)
        logger.info(
            f"  {idx + 1}/{total} studies "
            f"({processed} processed, {len(errors)} errors, "
            f"~{remaining/60:.0f} min remaining)"
        )

logger.info(f"\nPreprocessing complete: {processed}/{total} studies, {len(errors)} errors")

# ── Create tar archive for Kaggle dataset upload ─────────────────────
logger.info("Creating tar archive of volumes...")
tar_path = WORK_DIR / "volumes.tar"
with tarfile.open(tar_path, "w") as tar:
    for npy_file in sorted(VOLUME_DIR.glob("*.npy")):
        tar.add(npy_file, arcname=npy_file.name)

tar_size_mb = tar_path.stat().st_size / (1024 * 1024)
logger.info(f"Archive: {tar_path} ({tar_size_mb:.0f} MB)")

# Save error log
if errors:
    pd.DataFrame(errors).to_csv(WORK_DIR / "preprocess_errors.csv", index=False)

logger.info("\nPreprocessing complete.")
logger.info("Next: upload volumes.tar as Kaggle Dataset, then run 04_train.py")
