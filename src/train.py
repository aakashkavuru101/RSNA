"""
Training script for knee MRI abnormality classifier.

Usage:
    python -m src.train --config configs/baseline.yaml --fold 0

Features:
- fp16 AMP (P100/T4 have no bf16)
- Scanner-grouped CV
- Asymmetric loss or masked BCE
- Label-aware horizontal flip (swaps medial↔lateral)
- OOF prediction saving
- CSV/JSON logging per experiment
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from .models.classifier import KneeClassifier, AsymmetricLoss, MaskedBCELoss
from .datasets.folds import create_scanner_grouped_folds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KneeMRIDataset(Dataset):
    """
    Dataset for knee MRI studies.

    Loads preprocessed .npy volumes and silver labels.
    """

    def __init__(
        self,
        study_ids: list,
        label_matrix: np.ndarray,
        volume_dir: Path,
        num_slices: int = 24,
        in_channels: int = 3,
        augment: bool = False,
        label_cols: list = None,
    ):
        self.study_ids = study_ids
        self.label_matrix = label_matrix
        self.volume_dir = Path(volume_dir)
        self.num_slices = num_slices
        self.in_channels = in_channels
        self.augment = augment
        self.label_cols = label_cols or [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA",
            "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture",
        ]

        # Laterality swap indices for label-aware flip
        # Medial Meniscus (2) ↔ Lateral Meniscus (3)
        # Medial OA (4) ↔ Lateral OA (5)
        self.swap_pairs = [(2, 3), (4, 5)]

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        study_id = self.study_ids[idx]
        labels = self.label_matrix[idx].copy()

        # Load preprocessed volume
        volume_path = self.volume_dir / f"{study_id}.npy"
        volume = np.load(volume_path)  # (num_slices, H, W) float32

        # Convert to 2.5D input: (num_slices, in_channels, H, W)
        # For each slice position, stack adjacent slices as channels
        num_slices = volume.shape[0]
        half = self.in_channels // 2
        padded = np.concatenate([
            np.repeat(volume[:1], half, axis=0),
            volume,
            np.repeat(volume[-1:], half, axis=0),
        ], axis=0)

        slices_25d = np.stack([
            padded[i:i + self.in_channels]
            for i in range(num_slices)
        ], axis=0)  # (num_slices, in_channels, H, W)

        # Augmentation
        if self.augment:
            slices_25d, labels = self._augment(slices_25d, labels)

        # Create loss mask: 1 for real labels, 0 for "not addressed" (0.5)
        mask = (labels != 0.5).astype(np.float32)

        return {
            "image": torch.from_numpy(slices_25d).float(),
            "labels": torch.from_numpy(labels).float(),
            "mask": torch.from_numpy(mask).float(),
            "study_id": study_id,
        }

    def _augment(
        self,
        volume: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply label-aware augmentation.

        Horizontal flip swaps medial ↔ lateral labels.
        """
        # Horizontal flip with 50% probability
        if np.random.random() < 0.5:
            volume = volume[:, :, :, ::-1].copy()  # flip W axis
            # Swap medial ↔ lateral labels
            for i, j in self.swap_pairs:
                labels[i], labels[j] = labels[j], labels[i]

        # Brightness/contrast jitter
        if np.random.random() < 0.3:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            volume = volume * contrast + (brightness - 1.0)

        return volume, labels


def compute_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute per-label and macro AUC."""
    results = {}
    aucs = []
    label_names = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]

    for i, name in enumerate(label_names):
        valid = mask[:, i] > 0
        if valid.sum() == 0:
            continue
        yt = y_true[valid, i]
        ys = y_score[valid, i]
        if len(np.unique(yt)) < 2:
            continue
        auc = roc_auc_score(yt, ys)
        results[name] = auc
        aucs.append(auc)

    results["macro_auc"] = np.mean(aucs) if aucs else 0.0
    return results


def train_fold(
    config: dict,
    fold: int,
    folds_df: pd.DataFrame,
    label_matrix: np.ndarray,
    study_ids: list,
    volume_dir: Path,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Train one fold and return validation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training fold {fold} on {device}")

    # Split
    val_mask = folds_df["fold"] == fold
    train_mask = ~val_mask

    train_ids = [s for s, m in zip(study_ids, train_mask) if m]
    val_ids = [s for s, m in zip(study_ids, val_mask) if m]
    train_labels = label_matrix[train_mask.values]
    val_labels = label_matrix[val_mask.values]

    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Datasets
    train_ds = KneeMRIDataset(
        train_ids, train_labels, volume_dir,
        num_slices=config["num_slices"],
        in_channels=config["in_channels"],
        augment=True,
    )
    val_ds = KneeMRIDataset(
        val_ids, val_labels, volume_dir,
        num_slices=config["num_slices"],
        in_channels=config["in_channels"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=True,
    )

    # Model
    model = KneeClassifier(
        backbone_name=config["backbone"],
        aggregator=config.get("aggregator", "bilstm"),
        in_channels=config["in_channels"],
        num_slices=config["num_slices"],
        num_classes=12,
        dropout=config.get("dropout", 0.3),
    ).to(device)

    # Loss
    if config.get("loss") == "asymmetric":
        criterion = AsymmetricLoss(
            gamma_neg=config.get("gamma_neg", 4.0),
            gamma_pos=config.get("gamma_pos", 1.0),
            clip=config.get("clip", 0.05),
        )
    else:
        criterion = MaskedBCELoss(
            label_smoothing=config.get("label_smoothing", 0.05),
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Scheduler
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_auc = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_probs = []
        val_labels_list = []
        val_masks_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)
                val_labels_list.append(batch["labels"].numpy())
                val_masks_list.append(batch["mask"].numpy())

        val_probs = np.concatenate(val_probs)
        val_labels_arr = np.concatenate(val_labels_list)
        val_masks_arr = np.concatenate(val_masks_list)

        metrics = compute_auc(val_labels_arr, val_probs, val_masks_arr)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} "
            f"loss={train_loss/len(train_loader):.4f} "
            f"macro_auc={metrics['macro_auc']:.4f} "
            f"time={elapsed:.1f}s"
        )

        # Save best model
        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            torch.save(
                model.state_dict(),
                output_dir / f"fold{fold}_best.pt",
            )

    # Save OOF predictions
    oof_df = pd.DataFrame(val_probs, columns=[
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ])
    oof_df["StudyInstanceUID"] = val_ids
    oof_df["fold"] = fold
    oof_df.to_csv(output_dir / f"fold{fold}_oof.csv", index=False)

    # Save metrics
    metrics["fold"] = fold
    metrics["best_auc"] = best_auc
    with open(output_dir / f"fold{fold}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="models/baseline")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)
    train_csv = pd.read_csv(data_dir / "train.csv")
    folds_df = pd.read_csv(data_dir / "folds.csv")
    silver_labels = pd.read_csv(data_dir / "silver_labels.csv")

    # Build label matrix
    label_cols = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]
    label_matrix = silver_labels[label_cols].values.astype(np.float32)
    study_ids = silver_labels["StudyInstanceUID"].tolist()

    # Train
    metrics = train_fold(
        config=config,
        fold=args.fold,
        folds_df=folds_df,
        label_matrix=label_matrix,
        study_ids=study_ids,
        volume_dir=data_dir / "volumes",
        output_dir=output_dir,
    )

    logger.info(f"Fold {args.fold} complete: {metrics}")


if __name__ == "__main__":
    main()
