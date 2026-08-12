"""
DICOM series reading for RSNA Knee Abnormality Detection.

Handles the competition's specific data format:
- One .dcm file per slice
- 4 transfer syntaxes: Explicit VR LE, Implicit VR LE, JPEG Lossless, JPEG 2000
- 86-tag allowlist (most protocol metadata stripped)
- 20-45 slices per series (median 30), long tail to a few hundred
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut

logger = logging.getLogger(__name__)


def get_series_groups(study_dir: Path) -> Dict[str, List[Path]]:
    """
    Group DICOM files in a study directory by SeriesInstanceUID.

    Args:
        study_dir: Path to <StudyInstanceUID>/ directory containing
                   <SeriesInstanceUID>/ subdirectories.

    Returns:
        Dict mapping SeriesInstanceUID -> list of DICOM file paths.
    """
    groups: Dict[str, List[Path]] = {}
    for series_dir in sorted(study_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        series_uid = series_dir.name
        dcm_files = sorted(series_dir.glob("*.dcm"))
        if dcm_files:
            groups[series_uid] = dcm_files
    return groups


def read_dicom_metadata(ds: pydicom.Dataset) -> dict:
    """
    Extract relevant metadata from a pydicom Dataset.

    Uses defensive getattr throughout because the competition data
    is stripped to an 86-tag allowlist.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)

    return {
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", ""),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", ""),
        "SOPInstanceUID": getattr(ds, "SOPInstanceUID", ""),
        "InstanceNumber": getattr(ds, "InstanceNumber", None),
        "ImageOrientationPatient": [float(x) for x in iop] if iop else None,
        "ImagePositionPatient": [float(x) for x in ipp] if ipp else None,
        "Rows": getattr(ds, "Rows", None),
        "Columns": getattr(ds, "Columns", None),
        "PixelSpacing": [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])],
        "SliceThickness": getattr(ds, "SliceThickness", None),
        "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", "MONOCHROME2"),
        "Manufacturer": getattr(ds, "Manufacturer", ""),
        "ManufacturerModelName": getattr(ds, "ManufacturerModelName", ""),
        "MagneticFieldStrength": getattr(ds, "MagneticFieldStrength", None),
        "PatientSex": getattr(ds, "PatientSex", ""),
    }


def _sort_slices(slices: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """
    Sort DICOM slices geometrically.

    Primary: project ImagePositionPatient onto the slice normal
    computed as cross(row_direction, col_direction) from
    ImageOrientationPatient.

    Fallback: InstanceNumber.

    Never trust InstanceNumber alone — it is unreliable across vendors.
    """
    first = slices[0]
    iop = getattr(first, "ImageOrientationPatient", None)

    if iop is not None and len(iop) == 6:
        iop = np.asarray(iop, dtype=np.float64)
        normal = np.cross(iop[:3], iop[3:])

        def sort_key(ds: pydicom.Dataset) -> float:
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))
            return float(getattr(ds, "InstanceNumber", 0))

        try:
            return sorted(slices, key=sort_key)
        except (TypeError, ValueError):
            pass

    # Fallback: InstanceNumber
    return sorted(slices, key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))


def _pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    """
    Extract pixel array from a DICOM dataset with proper LUT application.

    Order: apply_modality_lut (rescale) FIRST, then apply_voi_lut (windowing).
    Inverts MONOCHROME1 images.
    """
    arr = apply_modality_lut(ds.pixel_array, ds)
    arr = apply_voi_lut(arr, ds)

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    return arr.astype(np.float32)


def read_series(
    paths: List[Path],
    target_slices: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Read one single-frame MRI series into a geometrically sorted float32 volume.

    Args:
        paths: List of .dcm file paths for one series.
        target_slices: If set, interpolate the volume to this many slices.

    Returns:
        volume: np.ndarray of shape (num_slices, rows, cols), float32.
        metadata: dict of series-level metadata from the first slice.
    """
    slices = [pydicom.dcmread(str(p)) for p in paths]
    slices = _sort_slices(slices)

    metadata = read_dicom_metadata(slices[0])

    volume = np.stack([_pixel_array(ds) for ds in slices])

    if target_slices is not None and volume.shape[0] != target_slices:
        volume = _interpolate_slices(volume, target_slices)

    return volume, metadata


def _interpolate_slices(volume: np.ndarray, target: int) -> np.ndarray:
    """
    Interpolate a volume along the slice axis to a fixed number of slices.

    Uses linear interpolation along the z-axis. This is the standard
    MRNet-style practice for normalizing slice count.
    """
    current = volume.shape[0]
    if current == target:
        return volume

    # Create interpolation indices
    old_idx = np.linspace(0, current - 1, current)
    new_idx = np.linspace(0, current - 1, target)

    # Interpolate each pixel along the slice axis
    # Reshape to (slices, -1) for vectorized interpolation
    flat = volume.reshape(current, -1)
    result = np.stack([
        np.interp(new_idx, old_idx, flat[:, i])
        for i in range(flat.shape[1])
    ], axis=1)

    return result.reshape(target, *volume.shape[1:]).astype(np.float32)
