"""
Submission script for Kaggle.

Thin notebook that loads trained fold models, runs inference on the
test set, and produces submission.csv.

CRITICAL: Internet is DISABLED during scoring. All dependencies must be
pre-installed from a wheel dataset. All model weights must be loaded
from Kaggle Datasets.

Usage (Kaggle notebook, GPU enabled, internet OFF):
    exec(open("notebooks/05_submit.py").read())
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

t0 = time.time()

# ── Paths ────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WORK_DIR = Path("/kaggle/working")

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# ── Offline dependency install ───────────────────────────────────────
# Dependencies must come from a pre-built wheel dataset (internet is OFF)
WHEELS_DIR = Path("/kaggle/input/rsna-knee-wheels/wheels")
if WHEELS_DIR.exists():
    logger.info("Installing dependencies from wheel dataset...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-index", f"--find-links={WHEELS_DIR}",
        "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg",
        "timm",
    ], check=True)
else:
    logger.info("Wheel dataset not found, assuming dependencies pre-installed")

import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut
import torch

# ── Load test data ───────────────────────────────────────────────────
logger.info("Loading test data...")
test_csv = pd.read_csv(DATA_ROOT / "test.csv")
test_series = pd.read_csv(DATA_ROOT / "test_series.csv")
logger.info(f"Test studies: {len(test_csv)}")

# Build series lookup
series_lookup = {}
for _, row in test_series.iterrows():
    uid = row["StudyInstanceUID"]
    if uid not in series_lookup:
        series_lookup[uid] = []
    series_lookup[uid].append({
        "series_uid": row["SeriesInstanceUID"],
        "plane": row.get("Anatomical_Plane", "unknown"),
        "fluid_sensitive": row.get("Fluid_Sensitive", 0),
        "fat_suppression": row.get("Fat_Suppression", 0),
    })

# ── DICOM reading (inline for offline compatibility) ─────────────────
def read_dicom_volume(paths, target_slices=24):
    """Read a DICOM series into a sorted float32 volume."""
    slices = [pydicom.dcmread(str(p)) for p in paths]

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

    volume = []
    for ds in slices:
        arr = apply_modality_lut(ds.pixel_array, ds)
        arr = apply_voi_lut(arr, ds)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        volume.append(arr.astype(np.float32))

    volume = np.stack(volume)

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
    lo, hi = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    mean, std = volume.mean(), volume.std()
    if std < 1e-8:
        std = 1.0
    return ((volume - mean) / std).astype(np.float32)


def resize_volume(volume, target_size=(224, 224)):
    slices, h, w = volume.shape
    th, tw = target_size
    if (h, w) == (th, tw):
        return volume
    row_idx = (np.arange(th) * (h - 1) / (th - 1)).astype(int)
    col_idx = (np.arange(tw) * (w - 1) / (tw - 1)).astype(int)
    return volume[:, row_idx][:, :, col_idx].astype(np.float32)


# ── Load model ───────────────────────────────────────────────────────
logger.info("Loading model...")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.classifier import KneeClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

config = {
    "backbone": "convnext_small",
    "aggregator": "bilstm",
    "in_channels": 3,
    "num_slices": 24,
    "num_classes": 12,
}

# Load fold models from Kaggle dataset
MODEL_DATASET = Path("/kaggle/input/rsna-knee-models")
models = []

for fold in range(5):
    model_path = MODEL_DATASET / f"fold{fold}_best.pt"
    if not model_path.exists():
        # Try working directory
        model_path = WORK_DIR / "models" / "baseline" / f"fold{fold}_best.pt"

    if not model_path.exists():
        logger.warning(f"Model not found for fold {fold}, skipping")
        continue

    model = KneeClassifier(
        backbone_name=config["backbone"],
        aggregator=config["aggregator"],
        in_channels=config["in_channels"],
        num_slices=config["num_slices"],
        num_classes=config["num_classes"],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models.append(model)

logger.info(f"Loaded {len(models)} fold models")

if not models:
    logger.error("No models loaded! Producing all-0.5 submission.")
    submission = test_csv[["StudyInstanceUID"]].copy()
    for col in LABEL_COLS:
        submission[col] = 0.5
    submission.to_csv(WORK_DIR / "submission.csv", index=False)
    logger.info("Fallback submission saved")
    sys.exit(0)

# ── Inference ────────────────────────────────────────────────────────
logger.info("Running inference...")

TEST_SERIES_DIR = DATA_ROOT / "test_series"
results = []
in_channels = config["in_channels"]
half = in_channels // 2

for idx, row in test_csv.iterrows():
    study_uid = row["StudyInstanceUID"]

    try:
        study_dir = TEST_SERIES_DIR / study_uid
        if not study_dir.exists():
            raise FileNotFoundError(f"Study dir not found: {study_uid}")

        series_list = series_lookup.get(study_uid, [])

        # Select best series (same logic as training)
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
            series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
            if not series_dirs:
                raise FileNotFoundError(f"No series in {study_uid}")
            best_series = {"series_uid": series_dirs[0].name}

        series_dir = study_dir / best_series["series_uid"]
        dcm_files = sorted(series_dir.glob("*.dcm"))

        if not dcm_files:
            raise FileNotFoundError(f"No .dcm files in {series_dir}")

        # Preprocess
        volume = read_dicom_volume(dcm_files, target_slices=config["num_slices"])
        volume = normalize_volume(volume)
        volume = resize_volume(volume)

        # Create 2.5D input
        num_slices = volume.shape[0]
        padded = np.concatenate([
            np.repeat(volume[:1], half, axis=0),
            volume,
            np.repeat(volume[-1:], half, axis=0),
        ], axis=0)

        slices_25d = np.stack([
            padded[i:i + in_channels]
            for i in range(num_slices)
        ], axis=0)

        x = torch.from_numpy(slices_25d).float().unsqueeze(0).to(device)

        # Ensemble prediction
        fold_probs = []
        for model in models:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            fold_probs.append(probs)

        probs = np.mean(fold_probs, axis=0)

    except Exception as e:
        logger.warning(f"Error on {study_uid}: {e}. Using 0.5.")
        probs = np.full(12, 0.5, dtype=np.float32)

    result = {"StudyInstanceUID": study_uid}
    for i, col in enumerate(LABEL_COLS):
        result[col] = float(probs[i])
    results.append(result)

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        logger.info(f"  {idx + 1}/{len(test_csv)} studies ({elapsed:.0f}s)")

# ── Write submission ─────────────────────────────────────────────────
submission = pd.DataFrame(results)
submission = submission[["StudyInstanceUID"] + LABEL_COLS]
submission.to_csv(WORK_DIR / "submission.csv", index=False)

elapsed = time.time() - t0
logger.info(f"\nSubmission saved: {len(submission)} rows")
logger.info(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
logger.info(f"Runtime limit: 9h — {'OK' if elapsed < 32400 else 'EXCEEDED'}")
