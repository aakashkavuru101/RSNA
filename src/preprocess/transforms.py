"""
Preprocessing transforms for knee MRI volumes.

Standard pipeline: percentile-clip → z-score → resize → slice selection → 2.5D stack.
Evidence: percentile-clip + z-score performs as well as Nyul/WhiteStripe for CNN
classification (Nature Sci Rep 2020).
"""

import numpy as np
from typing import Tuple


def normalize_volume(
    volume: np.ndarray,
    clip_percentiles: Tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """
    Percentile-clip and z-score normalize a volume.

    Args:
        volume: (slices, H, W) float32 array.
        clip_percentiles: Lower and upper percentile for clipping.

    Returns:
        Normalized volume, float32, zero mean / unit variance.
    """
    lo, hi = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    mean = volume.mean()
    std = volume.std()
    if std < 1e-8:
        std = 1.0
    return ((volume - mean) / std).astype(np.float32)


def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resize each slice of a volume to target_size.

    Uses simple nearest-neighbor via numpy indexing for speed.
    For higher quality, use cv2.resize or torch F.interpolate.

    Args:
        volume: (slices, H, W) float32 array.
        target_size: (H, W) target dimensions.

    Returns:
        Resized volume, (slices, target_H, target_W), float32.
    """
    slices, h, w = volume.shape
    th, tw = target_size

    if (h, w) == (th, tw):
        return volume

    # Use numpy-based bilinear-like resize via index mapping
    row_idx = (np.arange(th) * (h - 1) / (th - 1)).astype(int)
    col_idx = (np.arange(tw) * (w - 1) / (tw - 1)).astype(int)

    return volume[:, row_idx][:, :, col_idx].astype(np.float32)


def select_central_slices(
    volume: np.ndarray,
    num_slices: int = 3,
) -> np.ndarray:
    """
    Select N central slices from a volume, centered on the middle slice.

    For 2.5D input, we want adjacent slices around the center of the knee joint.
    The center of the series is a reasonable proxy for the joint center.

    Args:
        volume: (slices, H, W) float32 array.
        num_slices: Number of adjacent slices to select (must be odd).

    Returns:
        (num_slices, H, W) array of adjacent central slices.
    """
    assert num_slices % 2 == 1, "num_slices must be odd for symmetric selection"

    total = volume.shape[0]
    mid = total // 2
    half = num_slices // 2

    start = max(0, mid - half)
    end = min(total, mid + half + 1)

    # Pad if we're near the edge
    selected = volume[start:end]
    if selected.shape[0] < num_slices:
        pad_before = (num_slices - selected.shape[0]) // 2
        pad_after = num_slices - selected.shape[0] - pad_before
        selected = np.concatenate([
            np.repeat(selected[:1], pad_before, axis=0),
            selected,
            np.repeat(selected[-1:], pad_after, axis=0),
        ], axis=0)

    return selected


def stack_adjacent_slices(
    volume: np.ndarray,
    num_slices: int = 3,
) -> np.ndarray:
    """
    Create a 2.5D input by stacking N adjacent central slices as channels.

    Args:
        volume: (slices, H, W) float32 array.
        num_slices: Number of adjacent slices to stack (must be odd).

    Returns:
        (num_slices, H, W) array — treat as channels for 2D CNN input.
        For ConvNeXt with 3-channel input, use num_slices=3 and treat as RGB.
    """
    return select_central_slices(volume, num_slices)


def preprocess_series(
    volume: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    num_slices: int = 3,
    clip_percentiles: Tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """
    Full preprocessing pipeline for one series volume.

    Args:
        volume: (slices, H, W) raw float32 volume from DICOM.
        target_size: Spatial resize target.
        num_slices: Number of adjacent slices for 2.5D stacking.
        clip_percentiles: Percentile clipping range.

    Returns:
        (num_slices, target_H, target_W) preprocessed array.
    """
    volume = normalize_volume(volume, clip_percentiles)
    volume = resize_volume(volume, target_size)
    volume = stack_adjacent_slices(volume, num_slices)
    return volume
