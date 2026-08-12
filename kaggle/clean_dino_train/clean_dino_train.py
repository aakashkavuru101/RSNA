"""Clean DINOv2 multi-plane training and inference for RSNA Knee MRI.

The 58 expert-labelled studies are evaluation-only. They are never included in an
optimizer batch. Supervision comes from the independently generated v4 report labels.
The full configuration is enabled only after a bounded smoke run has validated the raw
DICOM, label, model, checkpoint, and submission paths.
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
SMOKE = False
SMOKE_NON_GOLD = 768
EPOCHS = 2 if SMOKE else 8
IMG_SIZE = 336
BASELINE_IMG_SIZE = 224
CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
GROUP_SIZE = 3
N_GROUPS = 1 if SMOKE else 2
CACHE_SLICES = GROUP_SIZE * N_GROUPS
BATCH_STUDIES = 3
EVAL_BATCH = 4
UNFREEZE_LAST = 4
LR_BACKBONE = 1.0e-5
LR_HEAD = 8.0e-4
WEIGHT_DECAY = 0.02
HEADER_THREADS = 16
DECODE_THREADS = 10
TIME_LIMIT_S = 7.8 * 3600
MIN_OOF_GAIN = 0.001
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 11)

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# Public competition metadata has two acquisition axes: plane and fluid sensitivity.
# A missing slot remains masked; it is never filled with a sequence of another type.
SLOTS = [
    ("SAG_FLUID", "Sagittal", 1),
    ("COR_FLUID", "Coronal", 1),
    ("AX_FLUID", "Axial", 1),
    ("SAG_STRUCT", "Sagittal", 0),
    ("COR_STRUCT", "Coronal", 0),
    ("AX_STRUCT", "Axial", 0),
]


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


def find_baseline_checkpoints() -> list[Path]:
    """Find the five validated 224 px fold checkpoints from the prior kernel."""
    candidates: dict[Path, dict[int, Path]] = {}
    pattern = re.compile(r"clean_dino_fold([0-4])\.pt$")
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        for filename in files:
            match = pattern.fullmatch(filename)
            if match:
                candidates.setdefault(Path(root), {})[int(match.group(1))] = Path(root) / filename
    complete = [files for files in candidates.values() if set(files) == set(range(5))]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"Expected one complete five-fold 224 px checkpoint package, found {len(complete)}"
        )
    return [complete[0][fold] for fold in range(5)]


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


def choose_smoke_studies(labels: pd.DataFrame) -> set[str]:
    gold = labels.loc[labels["is_gold"], "StudyInstanceUID"].tolist()
    non_gold = labels.loc[~labels["is_gold"], "StudyInstanceUID"].tolist()
    if not SMOKE:
        return set(gold + non_gold)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(non_gold, size=min(SMOKE_NON_GOLD, len(non_gold)), replace=False)
    keep = set(gold) | set(chosen.tolist())
    log(f"SMOKE mode: caching {len(chosen)} non-gold + {len(gold)} gold studies")
    return keep


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


def choose_slots(series_df: pd.DataFrame) -> dict[str, list[dict | None]]:
    output = {}
    for study, group in series_df.groupby("StudyInstanceUID", sort=True):
        chosen = []
        for _, plane, fluid in SLOTS:
            candidates = group[
                (group["Anatomical_Plane"] == plane)
                & (group["Fluid_Sensitive"] == fluid)
            ]
            if candidates.empty:
                chosen.append(None)
            else:
                # Denser stacks preserve more anatomy; UID makes ties deterministic.
                row = candidates.sort_values(
                    ["n_slices", "SeriesInstanceUID"], ascending=[False, True]
                ).iloc[0]
                chosen.append(row.to_dict())
        output[study] = chosen
    coverage = np.array([[slot is not None for slot in slots] for slots in output.values()])
    log("slot coverage: " + ", ".join(
        f"{SLOTS[i][0]}={coverage[:, i].mean():.1%}" for i in range(len(SLOTS))
    ))
    return output


HEADER_TAGS = [
    "Laterality", "ImageLaterality", "ImagePositionPatient", "ImageOrientationPatient",
    "PixelSpacing", "Rows", "Columns", "Manufacturer", "ManufacturerModelName",
    "MagneticFieldStrength", "StationName",
]


def _header_probe(item: tuple[str, dict]) -> tuple[str, dict]:
    study, record = item
    result = {"StudyInstanceUID": study}
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
    return study, result


def study_headers(slot_map: dict[str, list[dict | None]]) -> pd.DataFrame:
    items = []
    for study, slots in slot_map.items():
        record = next((slot for slot in slots if slot is not None), None)
        if record is not None:
            items.append((study, record))
    with ThreadPoolExecutor(max_workers=HEADER_THREADS) as pool:
        rows = dict(pool.map(_header_probe, items))
    result = pd.DataFrame([rows[s] for s in sorted(rows)])
    errors = result.get("header_error", pd.Series(dtype=object)).notna().sum()
    log(f"headers: {len(result)} studies, {int(errors)} read errors")
    return result


def _numbers(value, n: int) -> np.ndarray | None:
    if not isinstance(value, str):
        return None
    try:
        array = np.asarray([float(x) for x in value.split("|")], dtype=np.float64)
    except ValueError:
        return None
    return array if len(array) >= n and np.isfinite(array[:n]).all() else None


def laterality_map(headers: pd.DataFrame) -> dict[str, str | None]:
    output = {}
    tagged = geometric = unresolved = 0
    for row in headers.itertuples(index=False):
        values = [getattr(row, "Laterality", None), getattr(row, "ImageLaterality", None)]
        tags = [str(v).strip().upper()[:1] for v in values if v is not None]
        side = next((v for v in tags if v in ("L", "R")), None)
        if side is not None:
            tagged += 1
        else:
            ipp = _numbers(getattr(row, "ImagePositionPatient", None), 3)
            iop = _numbers(getattr(row, "ImageOrientationPatient", None), 6)
            spacing = _numbers(getattr(row, "PixelSpacing", None), 2)
            try:
                rows = float(getattr(row, "Rows"))
                cols = float(getattr(row, "Columns"))
                centre = ipp[:3] + iop[:3] * spacing[1] * cols / 2 + iop[3:6] * spacing[0] * rows / 2
                side = None if abs(centre[0]) < 20 else ("R" if centre[0] < 0 else "L")
            except (TypeError, ValueError, IndexError):
                side = None
            geometric += side is not None
            unresolved += side is None
        output[row.StudyInstanceUID] = side
    log(f"laterality: {tagged} tagged, {geometric} geometric, {unresolved} unresolved")
    return output


def scanner_groups(headers: pd.DataFrame) -> dict[str, str]:
    fields = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "StationName"]

    def clean(value) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "unknown"
        return re.sub(r"\s+", " ", str(value).strip().lower()) or "unknown"

    groups = {}
    for row in headers.itertuples(index=False):
        groups[row.StudyInstanceUID] = "|".join(clean(getattr(row, field, None)) for field in fields)
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
) -> tuple[str, int, np.ndarray | None, np.ndarray | None]:
    study, slot_index, record, side = job
    files = ordered_files(record)
    count = len(files)
    lo = int(SLICE_BAND[0] * (count - 1))
    hi = int(SLICE_BAND[1] * (count - 1))
    indices = np.linspace(lo, max(lo, hi), CACHE_SLICES).round().astype(int)
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
        return study, slot_index, None, None
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
    baseline = F.interpolate(
        tensor, (BASELINE_IMG_SIZE, BASELINE_IMG_SIZE), mode="bilinear", align_corners=False
    )
    output = (high_res.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    baseline_output = (
        (baseline.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    )
    plane = SLOTS[slot_index][1]
    if side == "R":
        if plane in ("Coronal", "Axial"):
            output = output[:, :, ::-1].copy()
            baseline_output = baseline_output[:, :, ::-1].copy()
        elif plane == "Sagittal":
            output = output[::-1].copy()
            baseline_output = baseline_output[::-1].copy()
    return study, slot_index, output, baseline_output


def build_cache(slot_map: dict[str, list[dict | None]], sides: dict[str, str | None], tag: str):
    studies = sorted(slot_map)
    index = {study: i for i, study in enumerate(studies)}
    cache = np.zeros(
        (len(studies), len(SLOTS), CACHE_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8
    )
    baseline_cache = np.zeros(
        (len(studies), len(SLOTS), CACHE_SLICES, BASELINE_IMG_SIZE, BASELINE_IMG_SIZE),
        dtype=np.uint8,
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
        for done, (study, slot_index, output, baseline_output) in enumerate(
            pool.map(_read_slot, jobs), 1
        ):
            if output is None:
                failures += 1
            else:
                cache[index[study], slot_index] = output
                baseline_cache[index[study], slot_index] = baseline_output
                mask[index[study], slot_index] = 1
            if done % 1000 == 0:
                log(f"{tag}: decoded {done}/{len(jobs)}")
            if time.time() - T0 > TIME_LIMIT_S:
                raise TimeoutError("Time budget exhausted during DICOM decoding")
    if np.any(mask.sum(axis=1) == 0):
        raise RuntimeError(f"{tag}: at least one study has no decodable slot")
    total_gib = (cache.nbytes + baseline_cache.nbytes) / 1024**3
    log(f"{tag}: dual cache {total_gib:.2f} GiB, failures={failures}")
    return studies, cache, baseline_cache, mask


class SlotHead(nn.Module):
    def __init__(self, dim: int, n_slots: int, n_outputs: int, hidden: int = 256):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_embedding = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden, n_outputs)
        self.hidden = hidden

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(features) + self.slot_embedding
        attention = torch.einsum("bsh,oh->bos", hidden, self.query) / self.hidden**0.5
        attention = attention.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        context = self.dropout(torch.einsum("bos,bsh->boh", attention, hidden))
        return (context * self.output.weight.unsqueeze(0)).sum(-1) + self.output.bias


class KneeDINO(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(backbone.config.hidden_size * 2, len(SLOTS), len(TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slots = images.shape[:2]
        pixels = images.reshape(batch * slots, *images.shape[2:]).float().div_(255.0)
        pixels = (pixels - self.mean) / self.std
        output = self.backbone(pixel_values=pixels).last_hidden_state
        features = torch.cat([output[:, 0], output[:, 1:].mean(1)], dim=1)
        return self.head(features.reshape(batch, slots, -1), mask)


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


def load_baseline_model(
    model_path: Path, checkpoint_path: Path, expected_fold: int, device: torch.device
) -> KneeDINO:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "fold": expected_fold,
        "img_size": BASELINE_IMG_SIZE,
        "gold_training_count": 0,
        "targets": TARGETS,
        "slots": SLOTS,
        "group_size": GROUP_SIZE,
        "n_groups": N_GROUPS,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Baseline fold {expected_fold} has {key}={checkpoint.get(key)!r}, "
                f"expected {expected!r}"
            )
    model = build_model(model_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def macro_auc(truth: np.ndarray, score: np.ndarray) -> tuple[float, dict[str, float]]:
    values = {}
    for i, target in enumerate(TARGETS):
        if len(np.unique(truth[:, i])) == 2:
            values[target] = float(roc_auc_score(truth[:, i], score[:, i]))
    return float(np.mean(list(values.values()))) if values else float("nan"), values


def rank_predictions(predictions: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(predictions)
    return frame.rank(method="average", pct=True).values.astype(np.float32)


def take_group(cache_rows: np.ndarray, group: int) -> np.ndarray:
    """Take one interleaved three-slice view spanning the central slice band."""
    output = cache_rows[:, :, group::N_GROUPS]
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
    return np.mean(group_outputs, axis=0)


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
    studies: list[str], labels: pd.DataFrame, groups: dict[str, str], fold_index: int = 0
):
    table = labels.set_index("StudyInstanceUID").loc[studies]
    eligible = np.flatnonzero(~table["is_gold"].values)
    group_values = np.array([groups.get(study, "unknown") for study in studies])
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
        f"scanner train/val={len(set(group_values[train_indices]))}/{len(set(group_values[val_indices]))}"
    )
    return table, train_indices, val_indices, gold_indices, len(splits)


def train_model(model, cache, mask, table, train_indices, val_indices, gold_indices, device):
    targets = table[TARGETS].values.astype(np.float32)
    confidence = 0.25 + 0.75 * np.abs(targets - 0.5) * 2.0
    gold_columns = [f"{target}__gold" for target in TARGETS]
    gold_truth = table[gold_columns].values.astype(np.float32)

    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": LR_BACKBONE,
        },
        {"params": model.head.parameters(), "lr": LR_HEAD},
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
            weights = torch.from_numpy(confidence[selected]).to(device, non_blocking=True)
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
        val_auc, per_target = macro_auc((targets[val_indices] > 0.5).astype(int), val_predictions)
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


def write_submission(predictions, studies, test_df, path="submission.csv"):
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


def main() -> None:
    seed_everything()
    root = find_competition_root()
    label_path = find_v4_labels()
    dino_path = find_dinov2_small()
    baseline_checkpoints = find_baseline_checkpoints()
    train_df = pd.read_csv(root / "train.csv")
    labels = validate_and_join_labels(train_df, label_path)
    train_keep = choose_smoke_studies(labels)

    train_series_csv = pd.read_csv(root / "train_series.csv")
    train_series = list_series(root / "train_series", train_series_csv, train_keep)
    train_slots = choose_slots(train_series)
    train_headers = study_headers(train_slots)
    train_sides = laterality_map(train_headers)
    groups = scanner_groups(train_headers)
    studies, train_cache, train_baseline_cache, train_mask = build_cache(
        train_slots, train_sides, "train"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training kernel requires a GPU")
    run_folds = 1 if SMOKE else 5
    table = labels.set_index("StudyInstanceUID").loc[studies]
    targets = table[TARGETS].values.astype(np.float32)
    eligible = np.flatnonzero(~table["is_gold"].values)
    gold_indices = np.flatnonzero(table["is_gold"].values)

    # Reproduce the validated 224 px ensemble on pixels resized directly from the same
    # normalized volumes. Its OOF predictions are the regression guard for this run.
    baseline_oof = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)
    baseline_gold_predictions = []
    for fold in range(run_folds):
        table, _, val_indices, gold_indices, available_folds = make_clean_split(
            studies, labels, groups, fold
        )
        if available_folds != run_folds:
            raise ValueError(f"Expected {run_folds} scanner folds, found {available_folds}")
        baseline_model = load_baseline_model(
            dino_path, baseline_checkpoints[fold], fold, device
        )
        baseline_oof[val_indices] = predict(
            baseline_model, train_baseline_cache, train_mask, val_indices, device
        )
        baseline_gold_predictions.append(
            predict(baseline_model, train_baseline_cache, train_mask, gold_indices, device)
        )
        del baseline_model
        gc.collect()
        torch.cuda.empty_cache()
    if not np.isfinite(baseline_oof[eligible]).all():
        raise RuntimeError("Baseline OOF predictions do not cover every non-gold study")
    del train_baseline_cache
    gc.collect()

    test_df = pd.read_csv(root / "test.csv")
    test_series_csv = pd.read_csv(root / "test_series.csv")
    test_keep = set(test_df["StudyInstanceUID"])
    test_series = list_series(root / "test_series", test_series_csv, test_keep)
    test_slots = choose_slots(test_series)
    test_headers = study_headers(test_slots)
    test_sides = laterality_map(test_headers)
    test_studies, test_cache, test_baseline_cache, test_mask = build_cache(
        test_slots, test_sides, "test"
    )

    baseline_test_predictions = []
    for fold in range(run_folds):
        baseline_model = load_baseline_model(
            dino_path, baseline_checkpoints[fold], fold, device
        )
        baseline_test_predictions.append(
            predict(
                baseline_model, test_baseline_cache, test_mask,
                np.arange(len(test_studies)), device,
            )
        )
        del baseline_model
        gc.collect()
        torch.cuda.empty_cache()
    del test_baseline_cache
    gc.collect()

    fold_histories = []
    fold_gold_predictions = []
    fold_test_predictions = []
    new_oof = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)

    for fold in range(run_folds):
        seed_everything(SEED + fold)
        table, train_indices, val_indices, gold_indices, available_folds = make_clean_split(
            studies, labels, groups, fold
        )
        if not SMOKE and available_folds != 5:
            raise ValueError(f"Full ensemble requires 5 scanner folds, found {available_folds}")
        model = build_model(dino_path)
        model, state, history = train_model(
            model, train_cache, train_mask, table,
            train_indices, val_indices, gold_indices, device,
        )

        checkpoint = {
            "state_dict": state,
            "targets": TARGETS,
            "slots": SLOTS,
            "img_size": IMG_SIZE,
            "crop_mm": CROP_MM,
            "slice_band": SLICE_BAND,
            "group_size": GROUP_SIZE,
            "n_groups": N_GROUPS,
            "cache_slices": CACHE_SLICES,
            "fold": fold,
            "train_studies": [studies[i] for i in train_indices],
            "validation_studies": [studies[i] for i in val_indices],
            "gold_training_count": 0,
            "label_source": label_path.name,
            "smoke": SMOKE,
        }
        checkpoint_name = "clean_dino_smoke.pt" if SMOKE else f"clean_dino_fold{fold}.pt"
        torch.save(checkpoint, checkpoint_name)

        new_oof[val_indices] = predict(
            model, train_cache, train_mask, val_indices, device
        )
        fold_gold_predictions.append(
            predict(model, train_cache, train_mask, gold_indices, device)
        )
        fold_test_predictions.append(
            predict(model, test_cache, test_mask, np.arange(len(test_studies)), device)
        )
        fold_histories.append({
            "fold": fold,
            "train_studies": len(train_indices),
            "validation_studies": len(val_indices),
            "history": history,
        })
        del model, state
        gc.collect()
        torch.cuda.empty_cache()

    if not np.isfinite(new_oof[eligible]).all():
        raise RuntimeError("New OOF predictions do not cover every non-gold study")

    # Select one conservative family-level blend on pooled scanner-isolated OOF. Gold
    # predictions are deliberately not available to this decision.
    oof_truth = (targets[eligible] > 0.5).astype(int)
    baseline_oof_rank = rank_predictions(baseline_oof[eligible])
    new_oof_rank = rank_predictions(new_oof[eligible])
    baseline_oof_auc, _ = macro_auc(oof_truth, baseline_oof_rank)
    new_oof_auc, _ = macro_auc(oof_truth, new_oof_rank)
    blend_scores = {}
    for weight in BLEND_WEIGHTS:
        blended = (1.0 - weight) * baseline_oof_rank + weight * new_oof_rank
        blend_scores[float(weight)], _ = macro_auc(oof_truth, blended)
    best_weight = max(blend_scores, key=blend_scores.get)
    best_oof_auc = blend_scores[best_weight]
    log(
        f"pooled OOF: baseline={baseline_oof_auc:.4f}, new={new_oof_auc:.4f}, "
        f"best blend={best_oof_auc:.4f} at new_weight={best_weight:.1f}"
    )
    if best_oof_auc < baseline_oof_auc + MIN_OOF_GAIN:
        raise RuntimeError(
            f"No honest OOF improvement ({best_oof_auc:.6f} vs "
            f"{baseline_oof_auc:.6f}); refusing to write submission.csv"
        )

    gold_columns = [f"{target}__gold" for target in TARGETS]
    gold_truth = table[gold_columns].values[gold_indices].astype(int)
    baseline_gold_ensemble = rank_predictions(np.mean(baseline_gold_predictions, axis=0))
    new_gold_ensemble = rank_predictions(np.mean(fold_gold_predictions, axis=0))
    gold_ensemble = (
        (1.0 - best_weight) * baseline_gold_ensemble + best_weight * new_gold_ensemble
    )
    gold_auc, gold_per_target = macro_auc(gold_truth, gold_ensemble)
    baseline_test_ensemble = rank_predictions(
        np.mean([rank_predictions(p) for p in baseline_test_predictions], axis=0)
    )
    new_test_ensemble = rank_predictions(np.mean(
        [rank_predictions(predictions) for predictions in fold_test_predictions], axis=0
    ))
    test_ensemble = (
        (1.0 - best_weight) * baseline_test_ensemble + best_weight * new_test_ensemble
    )
    Path("metrics.json").write_text(json.dumps({
        "folds": fold_histories,
        "baseline_oof_auc": baseline_oof_auc,
        "new_oof_auc": new_oof_auc,
        "best_blend_oof_auc": best_oof_auc,
        "new_model_blend_weight": best_weight,
        "blend_oof_scores": {str(k): v for k, v in blend_scores.items()},
        "minimum_required_oof_gain": MIN_OOF_GAIN,
        "ensemble_gold_auc_monitor_only": gold_auc,
        "ensemble_gold_per_target": gold_per_target,
        "gold_eval_studies": len(gold_indices),
        "gold_training_studies": 0,
        "elapsed_seconds_before_submission": time.time() - T0,
    }, indent=2))
    log(f"blended ensemble gold monitor: {gold_auc:.4f}")
    write_submission(test_ensemble, test_studies, test_df)
    log(f"complete in {(time.time() - T0) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
