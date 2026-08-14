"""Sequence-routed DINOv2 specialist transfer for RSNA Knee MRI.

The 58 expert-labelled studies are evaluation-only. They are never included in an
optimizer batch. Supervision comes from the independently generated v4 report labels.
The public structural flag mixes T1, PD, T2, and GRE series. This experiment recovers
sequence weighting and fat suppression from DICOM headers, separates those contrasts,
and samples three true contiguous windows per slot. Each fold is warm-started from the
successful localized family, then fine-tuned only for ACL, contusion, and fracture--the
three targets where routed v1 supplied complementary OOF signal. Exact 0.25/0.50 cells
remain masked. The other nine targets stay on the immutable prior 0.8041 OOF blend.
The 58 gold studies remain monitor-only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


T0 = time.time()
SEED = 20260813
EPOCHS = 5
IMG_SIZE = 280
CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
GROUP_SIZE = 3
N_GROUPS = 3
CACHE_SLICES = GROUP_SIZE * N_GROUPS
WINDOW_CENTRES = (0.25, 0.50, 0.75)
BATCH_STUDIES = 4
EVAL_BATCH = 6
UNFREEZE_LAST = 4
LR_BACKBONE = 3.0e-6
LR_HEAD = 2.0e-4
WEIGHT_DECAY = 0.02
HEADER_THREADS = 16
DECODE_THREADS = 10
TIME_LIMIT_S = 7.8 * 3600
MIN_OOF_GAIN = 0.002
MIN_TARGET_GAIN = 0.001
MAX_FOLD_REGRESSION = 0.015
MAX_MACRO_FOLD_REGRESSION = 0.002
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 5)

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SPECIALIST_TARGETS = {"ACL", "Contusion", "Fracture"}

# DICOM-derived slots. Missing acquisitions remain masked and are never substituted
# from a neighbouring predicate because doing so would reintroduce contrast mixing.
SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

# Per-target anatomical-window aggregation. Localized findings use the strongest
# windows; diffuse findings retain the stable mean used by the prior family.
WINDOW_POOL = {
    "ACL": "top2", "MCL": "top2", "Medial Meniscus": "max",
    "Lateral Meniscus": "max", "Contusion": "max", "Fracture": "max",
    "Baker's": "max",
}

FATSAT_OPTIONS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
FATSAT_RE = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
T1_RE = re.compile(r"\bt1\b|\bt1w\b")
T2_RE = re.compile(r"\bt2\b|\bt2w\b")
PD_RE = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("data"),
    ]
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series").is_dir():
            return path
    raise FileNotFoundError("RSNA Knee competition input was not mounted")


def find_v4_labels() -> Path:
    exact = []
    for root, directories, files in os.walk("/kaggle/input"):
        # Never recursively index the raw DICOM trees just to locate a small CSV.
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if "llm_labels_v4_blend.csv" in files:
            exact.append(Path(root) / "llm_labels_v4_blend.csv")
    if len(exact) != 1:
        raise FileNotFoundError(
            f"Expected exactly one llm_labels_v4_blend.csv, found {len(exact)}"
        )
    return exact[0]


def find_dinov2_small() -> Path:
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        text = root.lower()
        if "config.json" in files and "dinov2" in text and "small" in text:
            hits.append(Path(root))
    if not hits:
        raise FileNotFoundError("Attached DINOv2-small model was not found")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def find_unique_artifact(filename: str, marker: str) -> Path:
    """Resolve one attached prior-run artifact without assuming Kaggle mount names."""
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if filename in files and marker in root.lower():
            hits.append(Path(root) / filename)
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected one {filename!r} artifact containing {marker!r}, found {len(hits)}"
        )
    return hits[0]


def find_localized_checkpoints() -> list[Path]:
    """Resolve the successful five-fold localized family used for warm starts."""
    families: dict[Path, dict[int, Path]] = {}
    pattern = re.compile(r"localized_dino_fold([0-4])\.pt$")
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if "localized-dinov2" not in root.lower():
            continue
        for filename in files:
            match = pattern.match(filename)
            if match:
                families.setdefault(Path(root), {})[int(match.group(1))] = Path(root) / filename
    complete = [folds for folds in families.values() if set(folds) == set(range(5))]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"Expected one complete localized checkpoint family, found {len(complete)}"
        )
    return [complete[0][fold] for fold in range(5)]


def find_routed_output() -> Path | None:
    """Find a complete, OOF-approved routed checkpoint family for inference mode."""
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if "weights_manifest.json" not in files:
            continue
        path = Path(root)
        try:
            manifest = json.loads((path / "weights_manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        expected = [path / f"routed_dino_fold{fold}.pt" for fold in range(5)]
        if (
            manifest.get("architecture") == "routed_specialist_transfer_v2"
            and all(checkpoint.is_file() for checkpoint in expected)
        ):
            hits.append(path)
    if len(hits) > 1:
        raise FileNotFoundError(f"Expected at most one routed checkpoint output, found {len(hits)}")
    return hits[0] if hits else None


def safe_target(target: str) -> str:
    return target.lower().replace(" ", "_").replace("'", "")


def validate_and_join_labels(train_df: pd.DataFrame, label_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(label_path)
    expected = ["StudyInstanceUID"] + TARGETS
    if labels.columns.tolist() != expected:
        raise ValueError(f"Unexpected v4 label columns: {labels.columns.tolist()}")
    if labels["StudyInstanceUID"].duplicated().any():
        raise ValueError("v4 labels contain duplicate StudyInstanceUID values")
    merged = train_df[["StudyInstanceUID", "Report"] + TARGETS].merge(
        labels, on="StudyInstanceUID", how="left", validate="one_to_one",
        suffixes=("__gold", ""),
    )
    if merged[TARGETS].isna().any().any():
        raise ValueError("v4 labels do not cover every training study")
    gold_cols = [f"{target}__gold" for target in TARGETS]
    merged["is_gold"] = merged[gold_cols].notna().any(axis=1)
    n_gold = int(merged["is_gold"].sum())
    if n_gold != 58:
        raise ValueError(f"Expected 58 gold studies, found {n_gold}")
    log(f"labels: {len(merged)} studies, {n_gold} gold held out, source={label_path.name}")
    return merged


def training_studies(labels: pd.DataFrame) -> set[str]:
    """Return the complete cohort; fold locking requires complete exact OOF coverage."""
    return set(labels["StudyInstanceUID"].tolist())


def list_series(split_dir: Path, series_csv: pd.DataFrame, studies: set[str] | None) -> pd.DataFrame:
    rows = []
    metadata = series_csv.set_index("SeriesInstanceUID")
    for study_entry in os.scandir(split_dir):
        if not study_entry.is_dir() or (studies is not None and study_entry.name not in studies):
            continue
        for series_entry in os.scandir(study_entry.path):
            if not series_entry.is_dir() or series_entry.name not in metadata.index:
                continue
            files = sorted(
                entry.name for entry in os.scandir(series_entry.path)
                if entry.is_file() and entry.name.lower().endswith(".dcm")
            )
            if not files:
                continue
            meta = metadata.loc[series_entry.name]
            rows.append({
                "StudyInstanceUID": study_entry.name,
                "SeriesInstanceUID": series_entry.name,
                "dir": series_entry.path,
                "files": files,
                "n_slices": len(files),
                "Anatomical_Plane": meta["Anatomical_Plane"],
                "Fluid_Sensitive": int(meta["Fluid_Sensitive"]),
            })
    out = pd.DataFrame(rows)
    log(f"filesystem: found {len(out)} series for {out.StudyInstanceUID.nunique()} studies")
    return out


HEADER_TAGS = [
    "Laterality", "ImageLaterality", "ImagePositionPatient", "ImageOrientationPatient",
    "PixelSpacing", "Rows", "Columns", "Manufacturer", "ManufacturerModelName",
    "MagneticFieldStrength", "StationName", "SeriesDescription", "SequenceName",
    "ScanOptions", "ScanningSequence", "RepetitionTime", "EchoTime",
]


def _header_probe(record: dict) -> dict:
    result = {
        "StudyInstanceUID": record["StudyInstanceUID"],
        "SeriesInstanceUID": record["SeriesInstanceUID"],
    }
    try:
        middle = record["files"][len(record["files"]) // 2]
        ds = pydicom.dcmread(
            os.path.join(record["dir"], middle), stop_before_pixels=True,
            force=True, specific_tags=HEADER_TAGS,
        )
        for tag in HEADER_TAGS:
            value = getattr(ds, tag, None)
            if value is None:
                result[tag] = None
            elif isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
                result[tag] = "|".join(str(x) for x in value)
            else:
                result[tag] = str(value)
    except Exception as exc:
        result["header_error"] = str(exc)[:160]
    return result


def annotate_series(series_df: pd.DataFrame) -> pd.DataFrame:
    """Read one header per series and recover weighting plus fat suppression."""
    records = series_df.to_dict("records")
    with ThreadPoolExecutor(max_workers=HEADER_THREADS) as pool:
        rows = list(pool.map(_header_probe, records))
    headers = pd.DataFrame(rows)
    result = series_df.merge(
        headers, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left",
        validate="one_to_one",
    )
    errors = result.get("header_error", pd.Series(dtype=object)).notna().sum()
    text = (result.SeriesDescription.fillna("") + " " + result.SequenceName.fillna(""))
    text = text.str.lower().str.replace(r"[_\-.]", " ", regex=True)
    options = result.ScanOptions.fillna("").str.upper().str.split("|")
    exact_fs = options.apply(
        lambda values: any(value.strip() in FATSAT_OPTIONS for value in values)
    )
    result["fatsat"] = text.str.contains(FATSAT_RE) | exact_fs
    tr = pd.to_numeric(result.RepetitionTime, errors="coerce")
    te = pd.to_numeric(result.EchoTime, errors="coerce")
    gre = result.ScanningSequence.fillna("").str.upper().str.contains("GR")
    t1 = text.str.contains(T1_RE)
    t2 = text.str.contains(T2_RE)
    pdw = text.str.contains(PD_RE)
    result["weight"] = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(t2 & ~pdw, "T2", np.where(
            pdw, "PD", np.where(gre, "GRE", np.where(
                tr < 800, "T1", np.where(te > 60, "T2", np.where(tr >= 800, "PD", "UNK"))
            ))
        )),
    )
    result["fluid_recovered"] = result.weight.isin(["PD", "T2"])
    log(f"headers: {len(result)} series, {int(errors)} read errors")
    return result


def choose_slots(series_df: pd.DataFrame) -> dict[str, list[dict | None]]:
    output = {}
    for study, group in series_df.groupby("StudyInstanceUID", sort=True):
        chosen = []
        for _, plane, fluid, fatsat in SLOTS:
            candidates = group[
                (group.Anatomical_Plane == plane)
                & (group.fluid_recovered == fluid)
                & (group.fatsat == fatsat)
            ]
            if candidates.empty:
                chosen.append(None)
            else:
                row = candidates.sort_values(
                    ["n_slices", "SeriesInstanceUID"], ascending=[False, True]
                ).iloc[0]
                chosen.append(row.to_dict())
        output[study] = chosen
    coverage = np.array([[slot is not None for slot in slots] for slots in output.values()])
    log("routed slot coverage: " + ", ".join(
        f"{SLOTS[i][0]}={coverage[:, i].mean():.1%}" for i in range(len(SLOTS))
    ))
    return output


def _numbers(value, n: int) -> np.ndarray | None:
    if not isinstance(value, str):
        return None
    try:
        array = np.asarray([float(x) for x in value.split("|")], dtype=np.float64)
    except ValueError:
        return None
    return array if len(array) >= n and np.isfinite(array[:n]).all() else None


def _row_side(row) -> tuple[str | None, str | None]:
    values = [getattr(row, "Laterality", None), getattr(row, "ImageLaterality", None)]
    tags = [str(value).strip().upper()[:1] for value in values if value is not None]
    tagged = next((value for value in tags if value in ("L", "R")), None)
    ipp = _numbers(getattr(row, "ImagePositionPatient", None), 3)
    iop = _numbers(getattr(row, "ImageOrientationPatient", None), 6)
    spacing = _numbers(getattr(row, "PixelSpacing", None), 2)
    try:
        rows = float(getattr(row, "Rows"))
        cols = float(getattr(row, "Columns"))
        centre = ipp[:3] + iop[:3] * spacing[1] * cols / 2 + iop[3:6] * spacing[0] * rows / 2
        geometric = None if abs(centre[0]) < 20 else ("R" if centre[0] < 0 else "L")
    except (TypeError, ValueError, IndexError):
        geometric = None
    return tagged, geometric


def laterality_map(headers: pd.DataFrame) -> dict[str, str | None]:
    """Resolve laterality from all series in a study, preferring explicit tags."""
    output = {}
    tagged = geometric = unresolved = 0
    disagreements = 0
    for study, group in headers.groupby("StudyInstanceUID", sort=True):
        tagged_sides, geometric_sides = [], []
        for row in group.itertuples(index=False):
            tag_side, geo_side = _row_side(row)
            if tag_side is not None:
                tagged_sides.append(tag_side)
            if geo_side is not None:
                geometric_sides.append(geo_side)
        if tagged_sides:
            side = pd.Series(tagged_sides).mode().iloc[0]
            tagged += 1
            if geometric_sides and pd.Series(geometric_sides).mode().iloc[0] != side:
                disagreements += 1
        elif geometric_sides:
            side = pd.Series(geometric_sides).mode().iloc[0]
            geometric += 1
        else:
            side = None
            unresolved += 1
        output[study] = side
    log(
        f"laterality: {tagged} tagged, {geometric} geometric, {unresolved} unresolved, "
        f"tag/geometry disagreements={disagreements}"
    )
    return output


def scanner_groups(headers: pd.DataFrame) -> dict[str, str]:
    fields = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "StationName"]

    def clean(value) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "unknown"
        return re.sub(r"\s+", " ", str(value).strip().lower()) or "unknown"

    groups = {}
    for study, frame in headers.groupby("StudyInstanceUID", sort=True):
        values = []
        for field in fields:
            cleaned = frame[field].map(clean)
            non_unknown = cleaned[cleaned != "unknown"]
            values.append((non_unknown if len(non_unknown) else cleaned).mode().iloc[0])
        groups[study] = "|".join(values)
    log(f"scanner groups: {len(set(groups.values()))} unique fingerprints")
    return groups


ORDER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"]


def ordered_files(record: dict) -> list[str]:
    keyed = []
    for position, filename in enumerate(record["files"]):
        coordinate = None
        instance = None
        try:
            ds = pydicom.dcmread(
                os.path.join(record["dir"], filename), stop_before_pixels=True,
                force=True, specific_tags=ORDER_TAGS,
            )
            iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
            coordinate = float(np.dot(ipp, np.cross(iop[:3], iop[3:6])))
            instance = float(getattr(ds, "InstanceNumber", position))
        except Exception:
            try:
                instance = float(getattr(ds, "InstanceNumber"))
            except Exception:
                pass
        keyed.append((filename, coordinate, instance, position))
    if all(row[1] is not None for row in keyed):
        return [row[0] for row in sorted(keyed, key=lambda row: row[1])]
    if sum(row[2] is not None for row in keyed) >= max(2, int(0.8 * len(keyed))):
        return [row[0] for row in sorted(keyed, key=lambda row: (
            row[2] if row[2] is not None else float("inf"), row[3]
        ))]
    return sorted(record["files"], key=lambda name: [
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)
    ])


def _read_slot(
    job: tuple[str, int, dict, str | None]
) -> tuple[str, int, np.ndarray | None]:
    study, slot_index, record, side = job
    files = ordered_files(record)
    count = len(files)
    lo = int(SLICE_BAND[0] * (count - 1))
    hi = int(SLICE_BAND[1] * (count - 1))
    indices = []
    for fraction in WINDOW_CENTRES:
        centre = int(round(fraction * (count - 1)))
        centre = int(np.clip(centre, lo, max(lo, hi)))
        # A true 2.5D view: adjacent physical slices, with edge replication only
        # for unusually short stacks.
        indices.extend([
            int(np.clip(centre + offset, 0, count - 1))
            for offset in (-1, 0, 1)
        ])
    if len(indices) != CACHE_SLICES:
        raise ValueError(f"Expected {CACHE_SLICES} cached slices, got {len(indices)}")
    images = []
    spacing = None
    for index in indices:
        try:
            ds = pydicom.dcmread(os.path.join(record["dir"], files[int(index)]), force=True)
            image = ds.pixel_array.astype(np.float32)
            image = image * float(getattr(ds, "RescaleSlope", 1) or 1)
            image = image + float(getattr(ds, "RescaleIntercept", 0) or 0)
            raw_spacing = getattr(ds, "PixelSpacing", None)
            if raw_spacing is not None:
                spacing = float(raw_spacing[0])
            images.append(image)
        except Exception:
            images.append(None)
    good = [i for i, image in enumerate(images) if image is not None]
    if not good:
        return study, slot_index, None
    for i, image in enumerate(images):
        if image is None:
            images[i] = images[min(good, key=lambda j: abs(j - i))]
    shape = images[0].shape
    images = [image if image.shape == shape else np.zeros(shape, np.float32) for image in images]
    volume = np.stack(images)
    if spacing is not None and np.isfinite(spacing) and spacing > 0:
        wanted = int(round(CROP_MM / spacing))
        height, width = shape
        if 16 < wanted < min(height, width):
            cy, cx = height // 2, width // 2
            half = wanted // 2
            volume = volume[:, cy - half:cy + half, cx - half:cx + half]
    low, high = np.percentile(volume, [1, 99])
    volume = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    high_res = F.interpolate(
        tensor, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    output = (high_res.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    plane = SLOTS[slot_index][1]
    if side == "R":
        if plane in ("Coronal", "Axial"):
            output = output[:, :, ::-1].copy()
        elif plane == "Sagittal":
            output = output[::-1].copy()
    return study, slot_index, output


def build_cache(slot_map: dict[str, list[dict | None]], sides: dict[str, str | None], tag: str):
    studies = sorted(slot_map)
    index = {study: i for i, study in enumerate(studies)}
    cache = np.zeros(
        (len(studies), len(SLOTS), CACHE_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8
    )
    mask = np.zeros((len(studies), len(SLOTS)), dtype=np.float32)
    jobs = [
        (study, slot_index, record, sides.get(study))
        for study, slots in slot_map.items()
        for slot_index, record in enumerate(slots)
        if record is not None
    ]
    log(f"{tag}: decoding {len(jobs)} selected series")
    failures = 0
    with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
        for done, (study, slot_index, output) in enumerate(
            pool.map(_read_slot, jobs), 1
        ):
            if output is None:
                failures += 1
            else:
                cache[index[study], slot_index] = output
                mask[index[study], slot_index] = 1
            if done % 1000 == 0:
                log(f"{tag}: decoded {done}/{len(jobs)}")
            if time.time() - T0 > TIME_LIMIT_S:
                raise TimeoutError("Time budget exhausted during DICOM decoding")
    if np.any(mask.sum(axis=1) == 0):
        raise RuntimeError(f"{tag}: at least one study has no decodable slot")
    total_gib = cache.nbytes / 1024**3
    log(f"{tag}: contiguous-window cache {total_gib:.2f} GiB, failures={failures}")
    return studies, cache, mask


class LocalizedSlotHead(nn.Module):
    def __init__(self, dim: int, n_slots: int, n_outputs: int, hidden: int = 256):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_embedding = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.target_embedding = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.slot_query = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.dropout = nn.Dropout(0.2)
        self.output_weight = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.output_bias = nn.Parameter(torch.zeros(n_outputs))
        self.hidden = hidden

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[None, :, None, :]
        hidden = hidden + self.target_embedding[None, None, :, :]
        attention = torch.einsum("bsoh,oh->bos", hidden, self.slot_query) / self.hidden**0.5
        attention = attention.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        context = self.dropout(torch.einsum("bos,bsoh->boh", attention, hidden))
        return (context * self.output_weight.unsqueeze(0)).sum(-1) + self.output_bias


class KneeDINO(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        dim = backbone.config.hidden_size
        self.patch_query = nn.Parameter(torch.randn(len(TARGETS), dim) * 0.02)
        self.head = LocalizedSlotHead(dim * 2, len(SLOTS), len(TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slots = images.shape[:2]
        pixels = images.reshape(batch * slots, *images.shape[2:]).float().div_(255.0)
        pixels = (pixels - self.mean) / self.std
        tokens = self.backbone(pixel_values=pixels).last_hidden_state
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        patch_attention = torch.einsum(
            "npd,od->nop", patches, self.patch_query
        ).div_(patches.shape[-1] ** 0.5).softmax(-1)
        localized = torch.einsum("nop,npd->nod", patch_attention, patches)
        cls = cls[:, None, :].expand(-1, len(TARGETS), -1)
        features = torch.cat([cls, localized], dim=-1)
        return self.head(features.reshape(batch, slots, len(TARGETS), -1), mask)


def build_model(model_path: Path) -> KneeDINO:
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    layers = backbone.encoder.layer
    for block in layers[max(0, len(layers) - UNFREEZE_LAST):]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in backbone.layernorm.parameters():
        parameter.requires_grad = True
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    log(f"DINOv2: {len(layers)} blocks, last {UNFREEZE_LAST} open, {trainable/1e6:.1f}M trainable")
    return KneeDINO(backbone)


def initialize_from_localized(model: KneeDINO, checkpoint_path: Path, fold: int) -> None:
    """Warm start routed slots from the successful localized model for this fold."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "localized_target_patch_attention_v1":
        raise ValueError(f"Fold {fold} localized checkpoint has unexpected architecture")
    if checkpoint.get("fold") != fold or checkpoint.get("gold_training_count") != 0:
        raise ValueError(f"Fold {fold} localized checkpoint failed provenance validation")
    state = checkpoint["state_dict"]
    # Localized slot order: SAG/COR/AX fluid, then SAG/COR/AX structural.
    # Routed order: SAG/COR/AX fluid-FS, SAG fluid-noFS, COR T1, SAG T1.
    # Reuse the anatomically closest learned slot embedding for each routed slot.
    state["head.slot_embedding"] = state["head.slot_embedding"][[0, 1, 2, 0, 4, 3]].clone()
    model.load_state_dict(state, strict=True)
    log(f"fold {fold}: warm-started from {checkpoint_path.name}")


def macro_auc(truth: np.ndarray, score: np.ndarray) -> tuple[float, dict[str, float]]:
    values = {}
    for i, target in enumerate(TARGETS):
        if len(np.unique(truth[:, i])) == 2:
            values[target] = float(roc_auc_score(truth[:, i], score[:, i]))
    return float(np.mean(list(values.values()))) if values else float("nan"), values


def masked_macro_auc(
    truth: np.ndarray, score: np.ndarray, valid: np.ndarray
) -> tuple[float, dict[str, float]]:
    values = {}
    for index, target in enumerate(TARGETS):
        rows = valid[:, index]
        if rows.sum() >= 2 and len(np.unique(truth[rows, index])) == 2:
            values[target] = float(roc_auc_score(truth[rows, index], score[rows, index]))
    return float(np.mean(list(values.values()))) if values else float("nan"), values


def rank_predictions(predictions: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(predictions)
    return frame.rank(method="average", pct=True).values.astype(np.float32)


def take_group(cache_rows: np.ndarray, group: int) -> np.ndarray:
    """Take one true contiguous three-slice anatomical window."""
    start = group * GROUP_SIZE
    output = cache_rows[:, :, start:start + GROUP_SIZE]
    if output.shape[2] != GROUP_SIZE:
        raise ValueError(f"Expected {GROUP_SIZE} channels, got {output.shape[2]}")
    return output


@torch.no_grad()
def predict(model, cache, mask, indices, device) -> np.ndarray:
    model.eval()
    group_outputs = []
    for group in range(N_GROUPS):
        outputs = []
        for start in range(0, len(indices), EVAL_BATCH):
            selected = indices[start:start + EVAL_BATCH]
            images = torch.from_numpy(take_group(cache[selected], group)).to(
                device, non_blocking=True
            )
            present = torch.from_numpy(mask[selected]).to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, present)
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
        group_outputs.append(np.concatenate(outputs))
    stacked = np.stack(group_outputs, axis=0)
    result = stacked.mean(axis=0)
    for target, mode in WINDOW_POOL.items():
        index = TARGETS.index(target)
        values = stacked[:, :, index]
        if mode == "max":
            result[:, index] = values.max(axis=0)
        elif mode == "top2":
            count = min(2, len(values))
            result[:, index] = np.sort(values, axis=0)[-count:].mean(axis=0)
        else:
            raise ValueError(f"Unknown window pool {mode!r} for {target}")
    return result


def affine_augment(images: torch.Tensor) -> torch.Tensor:
    batch, slots, channels, height, width = images.shape
    flat = images.reshape(batch * slots, channels, height, width).float()
    count = flat.shape[0]
    angles = (torch.rand(count, device=flat.device) - 0.5) * (12 * math.pi / 180)
    scales = 0.94 + torch.rand(count, device=flat.device) * 0.12
    tx = (torch.rand(count, device=flat.device) - 0.5) * 0.08
    ty = (torch.rand(count, device=flat.device) - 0.5) * 0.08
    theta = torch.zeros(count, 2, 3, device=flat.device)
    theta[:, 0, 0] = scales * torch.cos(angles)
    theta[:, 0, 1] = -scales * torch.sin(angles)
    theta[:, 1, 0] = scales * torch.sin(angles)
    theta[:, 1, 1] = scales * torch.cos(angles)
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    gain = 0.9 + torch.rand(count, 1, 1, 1, device=flat.device) * 0.2
    bias = (torch.rand(count, 1, 1, 1, device=flat.device) - 0.5) * 16
    return (flat * gain + bias).clamp(0, 255).reshape(batch, slots, channels, height, width)


def make_clean_split(
    studies: list[str], labels: pd.DataFrame, groups: dict[str, str], fold_index: int = 0,
    locked_folds: dict[str, int] | None = None,
):
    table = labels.set_index("StudyInstanceUID").loc[studies]
    eligible = np.flatnonzero(~table["is_gold"].values)
    group_values = np.array([groups.get(study, "unknown") for study in studies])
    if locked_folds is None:
        unique = np.unique(group_values[eligible])
        if len(unique) < 2:
            raise ValueError("Need at least two scanner groups for honest validation")
        folds = min(5, len(unique))
        splitter = GroupKFold(n_splits=folds)
        splits = list(splitter.split(eligible, groups=group_values[eligible]))
        if not 0 <= fold_index < len(splits):
            raise ValueError(f"fold_index {fold_index} is outside 0..{len(splits)-1}")
        train_local, val_local = splits[fold_index]
        train_indices = eligible[train_local]
        val_indices = eligible[val_local]
    else:
        assigned = np.asarray([locked_folds.get(study, -1) for study in studies])
        if (assigned[eligible] < 0).any():
            raise ValueError("Locked OOF artifact does not cover every non-gold study")
        folds = len(np.unique(assigned[eligible]))
        val_indices = eligible[assigned[eligible] == fold_index]
        train_indices = eligible[assigned[eligible] != fold_index]

    target_values = table[TARGETS].values.astype(np.float32)
    addressed = (~np.isclose(target_values, 0.25)) & (~np.isclose(target_values, 0.50))
    specialist_columns = np.asarray([target in SPECIALIST_TARGETS for target in TARGETS])
    before_mask = len(train_indices)
    train_indices = train_indices[addressed[train_indices][:, specialist_columns].any(axis=1)]
    excluded_unaddressed = before_mask - len(train_indices)

    # Do not score duplicate report text whose target source appears in the other side.
    reports = table["Report"].fillna("").astype(str)
    report_hash = reports.map(lambda text: hashlib.sha256(text.encode()).hexdigest()).values
    val_hashes = set(report_hash[val_indices])
    train_indices = np.array([i for i in train_indices if report_hash[i] not in val_hashes])
    gold_indices = np.flatnonzero(table["is_gold"].values)
    if set(train_indices) & set(gold_indices):
        raise AssertionError("Gold leakage into training split")
    if set(group_values[train_indices]) & set(group_values[val_indices]):
        raise AssertionError("Scanner leakage between training and validation")
    log(
        f"fold {fold_index}: train={len(train_indices)}, val={len(val_indices)}, "
        f"gold={len(gold_indices)}, "
        f"all-unaddressed excluded={excluded_unaddressed}, "
        f"scanner train/val={len(set(group_values[train_indices]))}/{len(set(group_values[val_indices]))}"
    )
    return table, train_indices, val_indices, gold_indices, folds


def scanner_balance_weights(
    studies: list[str], groups: dict[str, str], train_indices: np.ndarray
) -> np.ndarray:
    """Mildly upweight underrepresented scanners without letting tiny sites dominate."""
    group_values = np.asarray([groups.get(study, "unknown") for study in studies])
    counts = pd.Series(group_values[train_indices]).value_counts()
    reference = float(np.median(counts.values))
    weights = np.ones(len(studies), dtype=np.float32)
    raw = np.asarray([
        math.sqrt(reference / float(counts[group_values[index]]))
        for index in train_indices
    ], dtype=np.float32)
    raw = np.clip(raw, 0.75, 1.50)
    raw = np.clip(raw / raw.mean(), 0.75, 1.50)
    weights[train_indices] = raw
    log(
        f"scanner balance: train weights {raw.min():.2f}..{raw.max():.2f}, "
        f"median group size={reference:.0f}"
    )
    return weights


def train_model(
    model, cache, mask, table, train_indices, val_indices, gold_indices,
    study_weights, device,
):
    targets = table[TARGETS].values.astype(np.float32)
    confidence = 0.25 + 0.75 * np.abs(targets - 0.5) * 2.0
    # The v4 table encodes report targets that were not addressed as exact 0.25;
    # exact 0.50 is also explicitly uninformative. Neither is a negative label.
    addressed = (~np.isclose(targets, 0.25)) & (~np.isclose(targets, 0.50))
    specialist_mask = np.asarray(
        [target in SPECIALIST_TARGETS for target in TARGETS], dtype=np.float32
    )
    confidence *= addressed.astype(np.float32)
    confidence *= specialist_mask[None, :]
    if np.any(confidence[train_indices].sum(axis=1) == 0):
        raise ValueError("At least one training study has no addressed target")
    coverage = addressed[train_indices].mean(axis=0)
    log("addressed label coverage: " + ", ".join(
        f"{target}={coverage[index]:.1%}" for index, target in enumerate(TARGETS)
    ))
    gold_columns = [f"{target}__gold" for target in TARGETS]
    gold_truth = table[gold_columns].values.astype(np.float32)

    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": LR_BACKBONE,
        },
        {
            "params": list(model.head.parameters()) + [model.patch_query],
            "lr": LR_HEAD,
        },
    ], weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_indices) // BATCH_STUDIES)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=steps_per_epoch * EPOCHS, pct_start=0.15,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc = -1.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        permutation = np.random.permutation(train_indices)
        losses = []
        for start in range(0, len(permutation) - BATCH_STUDIES + 1, BATCH_STUDIES):
            selected = permutation[start:start + BATCH_STUDIES]
            group = int(np.random.randint(N_GROUPS))
            images = torch.from_numpy(take_group(cache[selected], group)).to(
                device, non_blocking=True
            )
            images = affine_augment(images)
            present = torch.from_numpy(mask[selected]).to(device, non_blocking=True)
            truth = torch.from_numpy(targets[selected]).to(device, non_blocking=True)
            weights = confidence[selected] * study_weights[selected, None]
            weights = torch.from_numpy(weights).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, present)
                cell_loss = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
                loss = (cell_loss * weights).sum() / weights.sum().clamp_min(1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if time.time() - T0 > TIME_LIMIT_S:
                raise TimeoutError("Time budget exhausted during training")

        val_predictions = predict(model, cache, mask, val_indices, device)
        val_auc, per_target = masked_macro_auc(
            (targets[val_indices] > 0.5).astype(int), val_predictions,
            addressed[val_indices] & specialist_mask[None, :].astype(bool),
        )
        gold_predictions = predict(model, cache, mask, gold_indices, device)
        gold_auc, gold_per_target = macro_auc(gold_truth[gold_indices].astype(int), gold_predictions)
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "silver_scanner_auc": val_auc,
            "gold_auc_monitor_only": gold_auc,
            "silver_per_target": per_target,
            "gold_per_target": gold_per_target,
        }
        history.append(record)
        log(
            f"epoch {epoch+1}/{EPOCHS}: loss={record['loss']:.4f}, "
            f"scanner-val={val_auc:.4f}, gold-monitor={gold_auc:.4f}"
        )
        # Model selection uses scanner-isolated silver validation only, never gold.
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_state, history


def write_candidate_preview(predictions, studies, test_df, path="candidate_preview.csv"):
    prediction_df = pd.DataFrame(rank_predictions(predictions), columns=TARGETS)
    prediction_df.insert(0, "StudyInstanceUID", studies)
    output = test_df[["StudyInstanceUID"]].merge(
        prediction_df, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    if output[TARGETS].isna().any().any():
        raise ValueError("Missing test predictions; refusing to write a fallback submission")
    output.to_csv(path, index=False)
    log(f"wrote {path}: {output.shape}")
    return output


def inference_main(output_root: Path) -> None:
    """Run hidden-test inference from approved checkpoints without any retraining."""
    seed_everything()
    manifest = json.loads((output_root / "weights_manifest.json").read_text())
    if not manifest.get("candidate_ready", False):
        raise RuntimeError("Routed OOF gates did not pass; refusing to create submission.csv")
    if float(manifest.get("guarded_blend_oof_auc", 0.0)) <= 0.0:
        raise RuntimeError("Routed output is missing guarded-blend OOF evidence")
    if manifest.get("targets") != TARGETS or manifest.get("slots") != [list(row) for row in SLOTS]:
        raise ValueError("Checkpoint manifest does not match this inference implementation")

    root = find_competition_root()
    dino_path = find_dinov2_small()
    test_df = pd.read_csv(root / "test.csv")
    test_series_csv = pd.read_csv(root / "test_series.csv")
    test_keep = set(test_df["StudyInstanceUID"])
    test_series = list_series(root / "test_series", test_series_csv, test_keep)
    test_series = annotate_series(test_series)
    test_slots = choose_slots(test_series)
    test_sides = laterality_map(test_series)
    test_studies, test_cache, test_mask = build_cache(test_slots, test_sides, "hidden-test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This inference kernel requires a GPU")
    fold_predictions = []
    for fold in range(5):
        checkpoint_path = output_root / f"routed_dino_fold{fold}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != manifest["architecture"]:
            raise ValueError(f"Fold {fold} architecture does not match manifest")
        if checkpoint.get("gold_training_count") != 0:
            raise ValueError(f"Fold {fold} reports gold-label training leakage")
        model = build_model(dino_path)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(device)
        fold_predictions.append(
            rank_predictions(predict(
                model, test_cache, test_mask, np.arange(len(test_studies)), device
            ))
        )
        log(f"hidden-test: completed fold {fold + 1}/5")
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    ensemble = rank_predictions(np.mean(fold_predictions, axis=0))
    submission = write_candidate_preview(
        ensemble, test_studies, test_df, path="submission.csv"
    )
    if submission.shape != (len(test_df), len(TARGETS) + 1):
        raise RuntimeError("Final submission shape validation failed")
    log(f"inference-only submission complete in {(time.time() - T0) / 3600:.2f} hours")


def load_previous_candidate(
    oof_path: Path,
    eligible_studies: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    oof = pd.read_csv(oof_path)
    if oof["StudyInstanceUID"].duplicated().any():
        raise ValueError("Previous OOF artifact contains duplicate studies")
    oof = oof.set_index("StudyInstanceUID").reindex(eligible_studies)
    if oof.isna().any().any():
        raise ValueError("Previous OOF artifact does not align exactly with this run")
    baseline_oof = np.column_stack([
        oof[f"candidate_blend_{safe_target(target)}"].values for target in TARGETS
    ]).astype(np.float32)
    return oof, baseline_oof


def choose_target_blend(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Choose coarse target weights with pooled-gain and fold-regression guards."""
    weights = np.zeros(len(TARGETS), dtype=np.float32)
    blended = baseline.copy()
    details = {}
    for target_index, target in enumerate(TARGETS):
        y = truth[:, target_index]
        baseline_auc = float(roc_auc_score(y, baseline[:, target_index]))
        if target not in SPECIALIST_TARGETS:
            details[target] = {
                "baseline_auc": baseline_auc,
                "selected": {"weight": 0.0, "auc": baseline_auc, "gain": 0.0},
                "grid": [],
                "reason": "preserved exact baseline; outside specialist target set",
            }
            continue
        candidates = []
        for weight in BLEND_WEIGHTS:
            prediction = (
                (1.0 - weight) * baseline[:, target_index]
                + weight * candidate[:, target_index]
            )
            pooled = float(roc_auc_score(y, prediction))
            fold_deltas = []
            for fold in sorted(np.unique(fold_assignment)):
                rows = fold_assignment == fold
                if len(np.unique(y[rows])) != 2:
                    continue
                old_auc = roc_auc_score(y[rows], baseline[rows, target_index])
                new_auc = roc_auc_score(y[rows], prediction[rows])
                fold_deltas.append(float(new_auc - old_auc))
            worst_fold_delta = min(fold_deltas) if fold_deltas else 0.0
            valid = (
                weight == 0.0
                or (
                    pooled >= baseline_auc + MIN_TARGET_GAIN
                    and worst_fold_delta >= -MAX_FOLD_REGRESSION
                )
            )
            candidates.append({
                "weight": float(weight),
                "auc": pooled,
                "gain": pooled - baseline_auc,
                "worst_fold_delta": worst_fold_delta,
                "valid": bool(valid),
            })
        valid_candidates = [row for row in candidates if row["valid"]]
        selected = max(valid_candidates, key=lambda row: (row["auc"], -row["weight"]))
        weight = selected["weight"]
        weights[target_index] = weight
        blended[:, target_index] = (
            (1.0 - weight) * baseline[:, target_index]
            + weight * candidate[:, target_index]
        )
        details[target] = {
            "baseline_auc": baseline_auc,
            "selected": selected,
            "grid": candidates,
        }
    return weights, blended, details


def main() -> None:
    seed_everything()
    root = find_competition_root()
    label_path = find_v4_labels()
    dino_path = find_dinov2_small()
    previous_oof_path = find_unique_artifact(
        "localized_oof_predictions.csv", "localized-dinov2"
    )
    localized_checkpoints = find_localized_checkpoints()
    train_df = pd.read_csv(root / "train.csv")
    labels = validate_and_join_labels(train_df, label_path)
    train_keep = training_studies(labels)

    train_series_csv = pd.read_csv(root / "train_series.csv")
    train_series = list_series(root / "train_series", train_series_csv, train_keep)
    train_series = annotate_series(train_series)
    train_slots = choose_slots(train_series)
    train_sides = laterality_map(train_series)
    groups = scanner_groups(train_series)
    studies, train_cache, train_mask = build_cache(train_slots, train_sides, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training kernel requires a GPU")
    run_folds = 5
    table = labels.set_index("StudyInstanceUID").loc[studies]
    targets = table[TARGETS].values.astype(np.float32)
    eligible = np.flatnonzero(~table["is_gold"].values)
    gold_indices = np.flatnonzero(table["is_gold"].values)
    eligible_studies = np.asarray(studies)[eligible]

    previous_oof, baseline_oof = load_previous_candidate(
        previous_oof_path, eligible_studies
    )
    # Lock the scanner fingerprints to the exact prior OOF artifact. Header routing may
    # resolve metadata that was previously missing; changing group strings here would
    # silently change validation folds and invalidate the paired OOF comparison.
    for study, scanner in previous_oof["scanner_group"].items():
        groups[study] = scanner
    locked_folds = {
        study: int(fold) for study, fold in previous_oof["fold"].items()
    }
    log("scanner fold fingerprints locked to the previous exact OOF artifact")

    fold_histories = []
    fold_gold_predictions = []
    fold_assignment = np.full(len(studies), -1, dtype=np.int8)
    new_oof = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)

    for fold in range(run_folds):
        seed_everything(SEED + fold)
        table, train_indices, val_indices, gold_indices, available_folds = make_clean_split(
            studies, labels, groups, fold, locked_folds=locked_folds
        )
        if available_folds != run_folds:
            raise ValueError(f"Expected {run_folds} scanner folds, found {available_folds}")
        expected_fold = previous_oof.loc[np.asarray(studies)[val_indices], "fold"].values
        if not np.all(expected_fold == fold):
            raise ValueError("Scored OOF fold assignment differs from reconstructed split")
        fold_assignment[val_indices] = fold
        study_weights = scanner_balance_weights(studies, groups, train_indices)
        model = build_model(dino_path)
        initialize_from_localized(model, localized_checkpoints[fold], fold)
        model, state, history = train_model(
            model, train_cache, train_mask, table,
            train_indices, val_indices, gold_indices, study_weights, device,
        )

        checkpoint = {
            "state_dict": state,
            "architecture": "routed_specialist_transfer_v2",
            "targets": TARGETS,
            "slots": SLOTS,
            "img_size": IMG_SIZE,
            "crop_mm": CROP_MM,
            "slice_band": SLICE_BAND,
            "window_centres": WINDOW_CENTRES,
            "window_pool": WINDOW_POOL,
            "specialist_targets": sorted(SPECIALIST_TARGETS),
            "warm_start": "localized_target_patch_attention_v1",
            "view_layout": "contiguous_centre_minus_one_centre_plus_one",
            "group_size": GROUP_SIZE,
            "n_groups": N_GROUPS,
            "cache_slices": CACHE_SLICES,
            "fold": fold,
            "train_studies": [studies[i] for i in train_indices],
            "validation_studies": [studies[i] for i in val_indices],
            "gold_training_count": 0,
            "label_source": label_path.name,
            "label_mask": "exact_0.25_and_0.50",
        }
        checkpoint_name = f"routed_dino_fold{fold}.pt"
        torch.save(checkpoint, checkpoint_name)

        new_oof[val_indices] = predict(model, train_cache, train_mask, val_indices, device)
        fold_gold_predictions.append(
            predict(model, train_cache, train_mask, gold_indices, device)
        )
        fold_histories.append({
            "fold": fold,
            "train_studies": len(train_indices),
            "validation_studies": len(val_indices),
            "history": history,
        })
        del model, state, study_weights
        gc.collect()
        torch.cuda.empty_cache()

    if not np.isfinite(new_oof[eligible]).all() or (fold_assignment[eligible] < 0).any():
        raise RuntimeError("New OOF predictions do not cover every non-gold study")

    oof_truth = (targets[eligible] > 0.5).astype(int)
    new_oof_rank = rank_predictions(new_oof[eligible])
    baseline_oof_auc, baseline_per_target = macro_auc(oof_truth, baseline_oof)
    new_oof_auc, new_per_target = macro_auc(oof_truth, new_oof_rank)
    target_weights, blend_oof, blend_details = choose_target_blend(
        oof_truth, baseline_oof, new_oof_rank, fold_assignment[eligible]
    )
    best_oof_auc, blend_per_target = macro_auc(oof_truth, blend_oof)
    addressed_oof = (
        (~np.isclose(targets[eligible], 0.25))
        & (~np.isclose(targets[eligible], 0.50))
    )
    baseline_addressed_auc, baseline_addressed_per_target = masked_macro_auc(
        oof_truth, baseline_oof, addressed_oof
    )
    routed_addressed_auc, routed_addressed_per_target = masked_macro_auc(
        oof_truth, new_oof_rank, addressed_oof
    )
    blend_addressed_auc, blend_addressed_per_target = masked_macro_auc(
        oof_truth, blend_oof, addressed_oof
    )
    fold_guard = []
    routed_fold_guard = []
    for fold in range(run_folds):
        rows = fold_assignment[eligible] == fold
        previous_fold_auc, _ = macro_auc(oof_truth[rows], baseline_oof[rows])
        routed_fold_auc, _ = macro_auc(oof_truth[rows], new_oof_rank[rows])
        blend_fold_auc, _ = macro_auc(oof_truth[rows], blend_oof[rows])
        routed_fold_guard.append({
            "fold": fold,
            "previous_auc": previous_fold_auc,
            "candidate_auc": routed_fold_auc,
            "delta": routed_fold_auc - previous_fold_auc,
        })
        fold_guard.append({
            "fold": fold,
            "previous_auc": previous_fold_auc,
            "candidate_auc": blend_fold_auc,
            "delta": blend_fold_auc - previous_fold_auc,
        })
    log(
        f"pooled OOF: previous={baseline_oof_auc:.4f}, routed={new_oof_auc:.4f}, "
        f"guarded blend={best_oof_auc:.4f}"
    )
    incremental_gate = best_oof_auc >= baseline_oof_auc + MIN_OOF_GAIN
    selected_specialists = [
        target for target in SPECIALIST_TARGETS
        if target_weights[TARGETS.index(target)] > 0
    ]
    target_gate = len(selected_specialists) >= 2
    fold_gate = all(
        row["delta"] >= -MAX_MACRO_FOLD_REGRESSION for row in fold_guard
    )
    candidate_ready = bool(incremental_gate and target_gate and fold_gate)
    log(
        f"candidate gates: blend-gain={incremental_gate}, specialists="
        f"{selected_specialists}, fold-tolerance={fold_gate}"
    )

    gold_columns = [f"{target}__gold" for target in TARGETS]
    gold_truth = table[gold_columns].values[gold_indices].astype(int)
    new_gold_ensemble = rank_predictions(np.mean(fold_gold_predictions, axis=0))
    gold_auc, gold_per_target = macro_auc(gold_truth, new_gold_ensemble)
    exact_oof = pd.DataFrame({
        "StudyInstanceUID": eligible_studies,
        "fold": fold_assignment[eligible],
        "scanner_group": [groups.get(study, "unknown") for study in eligible_studies],
    })
    for index, target in enumerate(TARGETS):
        safe = safe_target(target)
        exact_oof[f"target_{safe}"] = targets[eligible, index]
        exact_oof[f"previous_blend_{safe}"] = baseline_oof[:, index]
        exact_oof[f"routed_{safe}"] = new_oof_rank[:, index]
        exact_oof[f"candidate_blend_{safe}"] = blend_oof[:, index]
    exact_oof.to_csv("routed_oof_predictions.csv", index=False)

    Path("metrics.json").write_text(json.dumps({
        "folds": fold_histories,
        "previous_candidate_oof_auc": baseline_oof_auc,
        "routed_oof_auc": new_oof_auc,
        "guarded_blend_oof_auc": best_oof_auc,
        "previous_candidate_per_target": baseline_per_target,
        "routed_per_target": new_per_target,
        "guarded_blend_per_target": blend_per_target,
        "addressed_only_oof": {
            "previous_candidate": baseline_addressed_auc,
            "routed": routed_addressed_auc,
            "guarded_blend": blend_addressed_auc,
            "previous_per_target": baseline_addressed_per_target,
            "routed_per_target": routed_addressed_per_target,
            "blend_per_target": blend_addressed_per_target,
        },
        "routed_fold_guard": routed_fold_guard,
        "fold_guard": fold_guard,
        "candidate_ready": candidate_ready,
        "specialist_targets": sorted(SPECIALIST_TARGETS),
        "selected_specialists": selected_specialists,
        "routed_target_weights": {
            target: float(target_weights[index]) for index, target in enumerate(TARGETS)
        },
        "target_blend_selection": blend_details,
        "minimum_required_oof_gain": MIN_OOF_GAIN,
        "minimum_target_gain": MIN_TARGET_GAIN,
        "maximum_fold_regression": MAX_FOLD_REGRESSION,
        "maximum_macro_fold_regression": MAX_MACRO_FOLD_REGRESSION,
        "routed_gold_auc_monitor_only": gold_auc,
        "routed_gold_per_target": gold_per_target,
        "gold_eval_studies": len(gold_indices),
        "gold_training_studies": 0,
        "elapsed_seconds_before_submission": time.time() - T0,
    }, indent=2))
    Path("weights_manifest.json").write_text(json.dumps({
        "architecture": "routed_specialist_transfer_v2",
        "checkpoint_pattern": "routed_dino_fold{fold}.pt",
        "folds": run_folds,
        "targets": TARGETS,
        "slots": SLOTS,
        "img_size": IMG_SIZE,
        "crop_mm": CROP_MM,
        "slice_band": SLICE_BAND,
        "window_centres": WINDOW_CENTRES,
        "window_pool": WINDOW_POOL,
        "specialist_targets": sorted(SPECIALIST_TARGETS),
        "warm_start": "localized_target_patch_attention_v1",
        "group_size": GROUP_SIZE,
        "n_groups": N_GROUPS,
        "routed_target_weights": {
            target: float(target_weights[index]) for index, target in enumerate(TARGETS)
        },
        "gold_training_count": 0,
        "label_mask": "exact_0.25_and_0.50",
        "candidate_ready": candidate_ready,
        "routed_oof_auc": new_oof_auc,
        "guarded_blend_oof_auc": best_oof_auc,
        "minimum_required_oof_gain": MIN_OOF_GAIN,
        "maximum_macro_fold_regression": MAX_MACRO_FOLD_REGRESSION,
    }, indent=2))
    log(f"routed family gold monitor: {gold_auc:.4f}")
    log("training output contains checkpoints and OOF evidence only; hidden inference is separate")
    log(f"complete in {(time.time() - T0) / 3600:.2f} hours")


if __name__ == "__main__":
    routed_output = find_routed_output()
    if routed_output is None:
        main()
    else:
        log(f"approved routed output detected at {routed_output}; entering inference-only mode")
        inference_main(routed_output)
