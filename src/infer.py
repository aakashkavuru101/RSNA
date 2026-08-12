"""
Inference script for knee MRI abnormality classifier.

Produces submission.csv in the required format.

Usage:
    python -m src.infer --config configs/baseline.yaml --model-dir models/baseline
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models.classifier import KneeClassifier
from .train import KneeMRIDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def predict_study(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    in_channels: int = 3,
) -> np.ndarray:
    """
    Predict 12-label probabilities for one study volume.

    Args:
        model: Trained KneeClassifier.
        volume: (num_slices, H, W) preprocessed volume.
        device: torch device.
        in_channels: Number of adjacent slices per 2.5D input.

    Returns:
        (12,) array of probabilities.
    """
    model.eval()

    num_slices = volume.shape[0]
    half = in_channels // 2
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

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs


def run_inference(
    config: dict,
    model_dir: Path,
    test_volume_dir: Path,
    test_csv: pd.DataFrame,
    output_path: Path,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Run inference on all test studies using ensemble of fold models.

    Args:
        config: Model config dict.
        model_dir: Directory containing fold*_best.pt files.
        test_volume_dir: Directory containing preprocessed test .npy volumes.
        test_csv: DataFrame with StudyInstanceUID column.
        output_path: Path to write submission.csv.
        n_folds: Number of fold models to ensemble.

    Returns:
        Submission DataFrame.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fold models
    models = []
    for fold in range(n_folds):
        model_path = model_dir / f"fold{fold}_best.pt"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}, skipping")
            continue

        model = KneeClassifier(
            backbone_name=config["backbone"],
            aggregator=config.get("aggregator", "bilstm"),
            in_channels=config["in_channels"],
            num_slices=config["num_slices"],
            num_classes=12,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    logger.info(f"Loaded {len(models)} fold models for ensemble")

    # Predict
    results = []
    for _, row in test_csv.iterrows():
        study_uid = row["StudyInstanceUID"]
        volume_path = test_volume_dir / f"{study_uid}.npy"

        if not volume_path.exists():
            logger.warning(f"Volume not found: {volume_path}, using 0.5")
            probs = np.full(12, 0.5, dtype=np.float32)
        else:
            volume = np.load(volume_path)
            fold_probs = []
            for model in models:
                p = predict_study(model, volume, device, config["in_channels"])
                fold_probs.append(p)
            probs = np.mean(fold_probs, axis=0)

        result = {"StudyInstanceUID": study_uid}
        for i, col in enumerate(LABEL_COLS):
            result[col] = float(probs[i])
        results.append(result)

    submission = pd.DataFrame(results)

    # Ensure column order matches required format
    submission = submission[["StudyInstanceUID"] + LABEL_COLS]
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} ({len(submission)} rows)")

    return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="input")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    test_csv = pd.read_csv(Path(args.data_dir) / "test.csv")

    run_inference(
        config=config,
        model_dir=Path(args.model_dir),
        test_volume_dir=Path(args.data_dir) / "test_volumes",
        test_csv=test_csv,
        output_path=Path(args.output),
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
