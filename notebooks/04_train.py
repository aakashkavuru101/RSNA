"""
Training wrapper for Kaggle.

Thin notebook that installs dependencies, loads cached volumes,
and calls the training loop.

Usage (Kaggle notebook, GPU enabled):
    exec(open("notebooks/04_train.py").read())
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WORK_DIR = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "models" / "baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Install dependencies ─────────────────────────────────────────────
logger.info("Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "timm", "pyyaml",
], check=True)

# ── Load data ────────────────────────────────────────────────────────
logger.info("Loading data...")

silver_labels = pd.read_csv(WORK_DIR / "silver_labels.csv")
folds_df = pd.read_csv(WORK_DIR / "folds.csv")

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

label_matrix = silver_labels[LABEL_COLS].values.astype(np.float32)
study_ids = silver_labels["StudyInstanceUID"].tolist()

# Extract volumes from tar if needed
VOLUME_DIR = WORK_DIR / "volumes"
if not VOLUME_DIR.exists() or not list(VOLUME_DIR.glob("*.npy")):
    tar_path = WORK_DIR / "volumes.tar"
    if tar_path.exists():
        logger.info("Extracting volumes from tar...")
        import tarfile
        VOLUME_DIR.mkdir(exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(VOLUME_DIR)
        logger.info(f"Extracted {len(list(VOLUME_DIR.glob('*.npy')))} volumes")
    else:
        # Try loading from Kaggle dataset
        kaggle_volumes = Path("/kaggle/input/rsna-knee-volumes")
        if kaggle_volumes.exists():
            logger.info("Loading volumes from Kaggle dataset...")
            import tarfile
            for tar_file in kaggle_volumes.glob("*.tar"):
                with tarfile.open(tar_file, "r") as tar:
                    tar.extractall(VOLUME_DIR)
            logger.info(f"Extracted {len(list(VOLUME_DIR.glob('*.npy')))} volumes")

# ── Training config ──────────────────────────────────────────────────
config = {
    "backbone": "convnext_small",
    "aggregator": "bilstm",
    "in_channels": 3,
    "num_slices": 24,
    "num_classes": 12,
    "epochs": 15,
    "batch_size": 16,
    "lr": 1.0e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "loss": "asymmetric",
    "gamma_neg": 4.0,
    "gamma_pos": 1.0,
    "clip": 0.05,
    "label_smoothing": 0.05,
    "num_workers": 2,
}

# ── Train all folds ──────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classifier import KneeClassifier, AsymmetricLoss, MaskedBCELoss
from src.train import KneeMRIDataset, compute_auc, train_fold

logger.info(f"Starting training: {config['backbone']} + {config['aggregator']}")
logger.info(f"Studies: {len(study_ids)}, Folds: 5")

all_metrics = []
t0 = time.time()

for fold in range(5):
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold}")
    logger.info(f"{'='*60}")

    fold_metrics = train_fold(
        config=config,
        fold=fold,
        folds_df=folds_df,
        label_matrix=label_matrix,
        study_ids=study_ids,
        volume_dir=VOLUME_DIR,
        output_dir=MODEL_DIR,
    )
    all_metrics.append(fold_metrics)

# ── Summary ──────────────────────────────────────────────────────────
elapsed = time.time() - t0
logger.info(f"\n{'='*60}")
logger.info(f"TRAINING COMPLETE")
logger.info(f"{'='*60}")

macro_aucs = [m["macro_auc"] for m in all_metrics]
logger.info(f"Fold macro AUCs: {[f'{a:.4f}' for a in macro_aucs]}")
logger.info(f"Mean macro AUC: {np.mean(macro_aucs):.4f} ± {np.std(macro_aucs):.4f}")
logger.info(f"Total time: {elapsed/3600:.1f} hours")

# Save summary
summary = {
    "config": config,
    "fold_metrics": all_metrics,
    "mean_macro_auc": float(np.mean(macro_aucs)),
    "std_macro_auc": float(np.std(macro_aucs)),
    "total_hours": elapsed / 3600,
}
with open(MODEL_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Combine OOF predictions
oof_dfs = []
for fold in range(5):
    oof_path = MODEL_DIR / f"fold{fold}_oof.csv"
    if oof_path.exists():
        oof_dfs.append(pd.read_csv(oof_path))

if oof_dfs:
    oof_combined = pd.concat(oof_dfs, ignore_index=True)
    oof_combined.to_csv(MODEL_DIR / "oof_combined.csv", index=False)
    logger.info(f"OOF predictions saved: {len(oof_combined)} rows")

logger.info("\nNext: run 05_submit.py for inference and submission")
