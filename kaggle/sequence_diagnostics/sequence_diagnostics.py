"""Read-only DICOM sequence and laterality diagnostics for RSNA Knee MRI.

This kernel never reads pixels, trains a model, or creates submission.csv.  It scans one
header per training series to quantify contrast mixing in the public plane/fluid slot
scheme and compare single-series laterality with a study-wide geometric consensus.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


THREADS = 24
TAGS = [
    "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
    "RepetitionTime", "EchoTime", "Laterality", "ImageLaterality",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows",
    "Columns",
]
PUBLIC_SLOTS = [
    ("SAG_FLUID", "Sagittal", 1), ("COR_FLUID", "Coronal", 1),
    ("AX_FLUID", "Axial", 1), ("SAG_STRUCT", "Sagittal", 0),
    ("COR_STRUCT", "Coronal", 0), ("AX_STRUCT", "Axial", 0),
]
RECOVERED_SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
FATSAT_OPTIONS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
FATSAT_RE = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
T1_RE = re.compile(r"\bt1\b|\bt1w\b")
T2_RE = re.compile(r"\bt2\b|\bt2w\b")
PD_RE = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


def root() -> Path:
    for path in [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]:
        if (path / "train_series.csv").is_file():
            return path
    raise FileNotFoundError("Competition input not mounted")


def scalar(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(item) for item in value)
    return str(value)


def probe(job: tuple[str, str, str]) -> dict:
    study, series, directory = job
    files = sorted(
        entry.name for entry in os.scandir(directory)
        if entry.is_file() and entry.name.lower().endswith(".dcm")
    )
    row = {
        "StudyInstanceUID": study, "SeriesInstanceUID": series,
        "n_slices": len(files), "error": None,
    }
    if not files:
        row["error"] = "empty series"
        return row
    try:
        ds = pydicom.dcmread(
            os.path.join(directory, files[len(files) // 2]), stop_before_pixels=True,
            force=True, specific_tags=TAGS,
        )
        row.update({tag: scalar(getattr(ds, tag, None)) for tag in TAGS})
    except Exception as exc:
        row["error"] = str(exc)[:160]
    return row


def scan(base: Path) -> pd.DataFrame:
    jobs = []
    for study in os.scandir(base / "train_series"):
        if not study.is_dir():
            continue
        for series in os.scandir(study.path):
            if series.is_dir():
                jobs.append((study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        return pd.DataFrame(pool.map(probe, jobs))


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    text = (frame.SeriesDescription.fillna("") + " " + frame.SequenceName.fillna(""))
    text = text.str.lower().str.replace(r"[_\-.]", " ", regex=True)
    options = frame.ScanOptions.fillna("").str.upper().str.split("|")
    exact_fs = options.apply(
        lambda values: any(value.strip() in FATSAT_OPTIONS for value in values)
    )
    frame["fatsat"] = text.str.contains(FATSAT_RE) | exact_fs
    tr = pd.to_numeric(frame.RepetitionTime, errors="coerce")
    te = pd.to_numeric(frame.EchoTime, errors="coerce")
    gre = frame.ScanningSequence.fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = text.str.contains(T1_RE), text.str.contains(T2_RE), text.str.contains(PD_RE)
    frame["weight"] = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(t2 & ~pdw, "T2", np.where(
            pdw, "PD", np.where(gre, "GRE", np.where(
                tr < 800, "T1", np.where(te > 60, "T2", np.where(tr >= 800, "PD", "UNK"))
            ))
        )),
    )
    frame["fluid_recovered"] = frame.weight.isin(["PD", "T2"])
    return frame


def vector(value, length: int):
    if not isinstance(value, str):
        return None
    try:
        result = np.asarray([float(item) for item in value.split("|")])
    except ValueError:
        return None
    return result if len(result) >= length and np.isfinite(result[:length]).all() else None


def geometric_side(row) -> str | None:
    ipp = vector(row.ImagePositionPatient, 3)
    iop = vector(row.ImageOrientationPatient, 6)
    spacing = vector(row.PixelSpacing, 2)
    try:
        centre = (
            ipp[:3] + iop[:3] * spacing[1] * float(row.Columns) / 2
            + iop[3:6] * spacing[0] * float(row.Rows) / 2
        )
    except (TypeError, ValueError, IndexError):
        return None
    return None if abs(centre[0]) < 20 else ("R" if centre[0] < 0 else "L")


def tagged_side(row) -> str | None:
    for value in [row.Laterality, row.ImageLaterality]:
        if isinstance(value, str) and value.strip().upper()[:1] in ("L", "R"):
            return value.strip().upper()[:1]
    return None


def select_slots(frame: pd.DataFrame, recovered: bool) -> pd.DataFrame:
    rows = []
    definitions = RECOVERED_SLOTS if recovered else PUBLIC_SLOTS
    for study, group in frame.groupby("StudyInstanceUID", sort=True):
        for definition in definitions:
            name, plane = definition[:2]
            candidates = group[group.Anatomical_Plane == plane]
            if recovered:
                _, _, fluid, fatsat = definition
                candidates = candidates[
                    (candidates.fluid_recovered == fluid) & (candidates.fatsat == fatsat)
                ]
            else:
                _, _, fluid = definition
                candidates = candidates[candidates.Fluid_Sensitive == fluid]
            if candidates.empty:
                continue
            chosen = candidates.sort_values(
                ["n_slices", "SeriesInstanceUID"], ascending=[False, True]
            ).iloc[0]
            row = chosen.to_dict()
            row["slot"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def distribution(series: pd.Series) -> dict:
    counts = series.fillna("missing").astype(str).value_counts()
    return {key: {"n": int(value), "share": float(value / len(series))} for key, value in counts.items()}


def main() -> None:
    base = root()
    metadata = pd.read_csv(base / "train_series.csv")
    headers = scan(base)
    frame = metadata.merge(
        headers, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left",
        validate="one_to_one",
    )
    frame = annotate(frame)
    public = select_slots(frame, recovered=False)
    recovered = select_slots(frame, recovered=True)
    studies = int(frame.StudyInstanceUID.nunique())

    coverage = lambda chosen: {
        slot: float((chosen.slot == slot).sum() / studies) for slot in sorted(chosen.slot.unique())
    }
    public_mix = {}
    for slot in [name for name, _, fluid in PUBLIC_SLOTS if fluid == 0]:
        selected = public[public.slot == slot]
        public_mix[slot] = {
            "n": int(len(selected)),
            "recovered_weight": distribution(selected.weight),
            "fatsat": distribution(selected.fatsat),
        }

    frame["tag_side"] = [tagged_side(row) for row in frame.itertuples(index=False)]
    frame["geo_side"] = [geometric_side(row) for row in frame.itertuples(index=False)]
    robust = {}
    for study, group in frame.groupby("StudyInstanceUID"):
        tags = group.tag_side.dropna()
        geo = group.geo_side.dropna()
        robust[study] = tags.iloc[0] if len(tags) else (geo.mode().iloc[0] if len(geo) else None)
    first_public = public.sort_values("slot", key=lambda s: s.map(
        {name: index for index, (name, _, _) in enumerate(PUBLIC_SLOTS)}
    )).drop_duplicates("StudyInstanceUID")
    current = {
        row.StudyInstanceUID: tagged_side(row) or geometric_side(row)
        for row in first_public.itertuples(index=False)
    }
    comparable = [study for study in robust if robust[study] and current.get(study)]
    disagreements = [study for study in comparable if robust[study] != current[study]]

    result = {
        "studies": studies,
        "series": int(len(frame)),
        "header_errors": int(headers.error.notna().sum()),
        "public_slot_coverage": coverage(public),
        "recovered_slot_coverage": coverage(recovered),
        "public_structural_contrast_mix": public_mix,
        "series_weight_distribution": distribution(frame.weight),
        "series_fatsat_distribution": distribution(frame.fatsat),
        "laterality": {
            "robust_resolved": int(sum(value is not None for value in robust.values())),
            "single_series_resolved": int(sum(value is not None for value in current.values())),
            "comparable": len(comparable),
            "disagreements": len(disagreements),
            "disagreement_rate": float(len(disagreements) / max(1, len(comparable))),
        },
    }
    Path("sequence_diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    frame.drop(columns=["files"], errors="ignore").to_csv("series_headers.csv", index=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
