"""FAST fold-0 A/B diagnostic — multimodal (image x report) vs plain base.

One scanner fold (fold 0, ~870 val studies), full 8-epoch training for the
multimodal model, and a prediction-only pass for the existing dino2b_fold0
(plain base, full-gold, TTA) checkpoint attached as the
`rsna-knee-ensemble-checkpoints` dataset. Both predict the SAME fold-0
validation studies; the two macro AUCs and per-target deltas are the honest
answer to "does report-guided vision strength help?" in ~3h instead of 7h.

No test decode, no submission — this is a diagnostic only.
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
EPOCHS = 8
IMG_SIZE = 336
CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
GROUP_SIZE = 3
N_GROUPS = 2
CACHE_SLICES = GROUP_SIZE * N_GROUPS
BATCH_STUDIES = 3
EVAL_BATCH = 4
UNFREEZE_LAST = 4
LR_BACKBONE = 1.0e-5
LR_HEAD = 8.0e-4
WEIGHT_DECAY = 0.02
HEADER_THREADS = 16
DECODE_THREADS = 10
TIME_LIMIT_S = 6.0 * 3600

MODEL_KEY = "base"          # both sides use DINOv2-base
LABELS_FILE = "silver_labels_v5.csv"
EMBEDDINGS_FILE = "report_embeddings.csv"
TEXT_DIM = 384
CLIP_WEIGHT = 0.25
CLIP_TEMP = 0.07
GOLD_LAMBDA = 2.0
TTA = True
FOLD = 0

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

SLOTS = [
    ("SAG_FLUID", "Sagittal", 1),
    ("COR_FLUID", "Coronal", 1),
    ("AX_FLUID", "Axial", 1),
    ("SAG_STRUCT", "Sagittal", 0),
    ("COR_STRUCT", "Coronal", 0),
    ("AX_STRUCT", "Axial", 0),
]

FLIP_SWAP = [TARGETS.index(t) for t in TARGETS]
SWAP_PAIRS = {
    "Medial Meniscus": "Lateral Meniscus",
    "Lateral Meniscus": "Medial Meniscus",
    "Medial OA": "Lateral OA",
    "Lateral OA": "Medial OA",
}
for a, b in SWAP_PAIRS.items():
    FLIP_SWAP[TARGETS.index(a)] = TARGETS.index(b)


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


def find_single_file(filename: str, tag: str) -> Path:
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if filename in files:
            hits.append(Path(root) / filename)
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename} ({tag}), found {len(hits)}")
    return hits[0]


def find_backbone() -> Path:
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        text = root.lower()
        if "config.json" in files and "dinov2" in text and MODEL_KEY in text:
            hits.append(Path(root))
    if not hits:
        raise FileNotFoundError(f"Attached DINOv2-{MODEL_KEY} model was not found")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def find_base_fold0_pt() -> Path:
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if "dino2b_fold0.pt" in files:
            hits.append(Path(root) / "dino2b_fold0.pt")
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one dino2b_fold0.pt, found {len(hits)}")
    return hits[0]


def validate_and_join_labels(train_df: pd.DataFrame, label_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(label_path)
    expected = ["StudyInstanceUID"] + TARGETS
    if labels.columns.tolist() != expected:
        raise ValueError(f"Unexpected label columns: {labels.columns.tolist()}")
    merged = train_df[["StudyInstanceUID", "Report"] + TARGETS].merge(
        labels, on="StudyInstanceUID", how="left", validate="one_to_one",
        suffixes=("__gold", ""),
    )
    if merged[TARGETS].isna().any().any():
        raise ValueError("labels do not cover every training study")
    gold_cols = [f"{target}__gold" for target in TARGETS]
    merged["is_gold"] = merged[gold_cols].notna().any(axis=1)
    n_gold = int(merged["is_gold"].sum())
    if n_gold != 58:
        raise ValueError(f"Expected 58 gold studies, found {n_gold}")
    log(f"labels: {len(merged)} studies, {n_gold} gold, source={label_path.name}")
    return merged


def list_series(split_dir: Path, series_csv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metadata = series_csv.set_index("SeriesInstanceUID")
    for study_entry in os.scandir(split_dir):
        if not study_entry.is_dir():
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
                rows_n = float(getattr(row, "Rows"))
                cols_n = float(getattr(row, "Columns"))
                centre = ipp[:3] + iop[:3] * spacing[1] * cols_n / 2 + iop[3:6] * spacing[0] * rows_n / 2
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


def _read_slot(job: tuple[str, int, dict, str | None]) -> tuple[str, int, np.ndarray | None]:
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
    resized = F.interpolate(tensor, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    output = (resized.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
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
        for done, (study, slot_index, output) in enumerate(pool.map(_read_slot, jobs), 1):
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
    log(f"{tag}: cache {cache.nbytes / 1024**3:.2f} GiB, failures={failures}")
    return studies, cache, mask


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


class KneeDINOMM(nn.Module):
    def __init__(self, backbone: nn.Module, text_dim: int = TEXT_DIM):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(backbone.config.hidden_size * 2, len(SLOTS), len(TARGETS))
        feature_dim = backbone.config.hidden_size * 2
        self.text_proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, text_dim)
        )
        self.clip_temp = nn.Parameter(torch.tensor(math.log(1.0 / CLIP_TEMP)))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, mask: torch.Tensor):
        batch, slots = images.shape[:2]
        pixels = images.reshape(batch * slots, *images.shape[2:]).float().div_(255.0)
        pixels = (pixels - self.mean) / self.std
        output = self.backbone(pixel_values=pixels).last_hidden_state
        features = torch.cat([output[:, 0], output[:, 1:].mean(1)], dim=1)
        features = features.reshape(batch, slots, -1)
        logits = self.head(features, mask)
        pooled = features.mean(dim=1)
        text_emb = F.normalize(self.text_proj(pooled), dim=-1)
        return logits, text_emb

    def inference_forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(images, mask)
        return logits


def infonce(image_emb, text_emb, temp):
    logits = image_emb @ text_emb.T / temp
    labels = torch.arange(len(image_emb), device=image_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def build_backbone(model_path: Path):
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
    return backbone


def macro_auc(truth, score):
    values = {}
    for i, target in enumerate(TARGETS):
        if len(np.unique(truth[:, i])) == 2:
            values[target] = float(roc_auc_score(truth[:, i], score[:, i]))
    return float(np.mean(list(values.values()))) if values else float("nan"), values


def rank_predictions(predictions):
    frame = pd.DataFrame(predictions)
    return frame.rank(method="average", pct=True).values.astype(np.float32)


def take_group(cache_rows: np.ndarray, group: int) -> np.ndarray:
    output = cache_rows[:, :, group::N_GROUPS]
    if output.shape[2] != GROUP_SIZE:
        raise ValueError(f"Expected {GROUP_SIZE} channels, got {output.shape[2]}")
    return output


@torch.no_grad()
def predict(model, cache, mask, indices, device, tta: bool = False) -> np.ndarray:
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
                logits = model.inference_forward(images, present) if isinstance(model, KneeDINOMM) else model(images, present)
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
        group_outputs.append(np.concatenate(outputs))
    probs = np.mean(group_outputs, axis=0)
    if tta:
        flipped = []
        for group in range(N_GROUPS):
            outputs = []
            for start in range(0, len(indices), EVAL_BATCH):
                selected = indices[start:start + EVAL_BATCH]
                images = torch.from_numpy(take_group(cache[selected], group)).to(
                    device, non_blocking=True
                )
                images = images.flip(dims=[-1])
                present = torch.from_numpy(mask[selected]).to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model.inference_forward(images, present) if isinstance(model, KneeDINOMM) else model(images, present)
                outputs.append(torch.sigmoid(logits).float().cpu().numpy())
            flipped.append(np.concatenate(outputs))
        flipped_probs = np.mean(flipped, axis=0)[:, FLIP_SWAP]
        probs = (probs + flipped_probs) / 2.0
    return probs


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


def make_clean_split(studies: list[str], labels: pd.DataFrame, groups: dict[str, str], fold_index: int = 0):
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


def train_mm(model, cache, mask, table, train_indices, val_indices, gold_indices, device,
             text_emb_np, uid_order):
    targets = table[TARGETS].values.astype(np.float32)
    confidence = 0.25 + 0.75 * np.abs(targets - 0.5) * 2.0
    gold_columns = [f"{target}__gold" for target in TARGETS]
    gold_truth = table[gold_columns].values.astype(np.float32)
    targets = targets.copy()
    confidence = confidence.copy()
    targets[gold_indices] = gold_truth[gold_indices].astype(np.float32)
    confidence[gold_indices] *= GOLD_LAMBDA
    full_train = np.concatenate([train_indices, gold_indices])

    report_pos = {uid: i for i, uid in enumerate(uid_order)}
    train_report_ids = np.array([report_pos[str(table.index[i])] for i in full_train])
    train_text = torch.from_numpy(text_emb_np[train_report_ids]).float()

    model = model.to(device)
    non_backbone = [p for name, p in model.named_parameters()
                    if p.requires_grad and not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": non_backbone, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(full_train) // BATCH_STUDIES)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=steps_per_epoch * EPOCHS,
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc = -1.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        permutation = np.random.permutation(len(full_train))
        losses = []
        for start in range(0, len(permutation) - BATCH_STUDIES + 1, BATCH_STUDIES):
            sel = permutation[start:start + BATCH_STUDIES]
            indices = full_train[sel]
            group = int(np.random.randint(N_GROUPS))
            images = torch.from_numpy(take_group(cache[indices], group)).to(device, non_blocking=True)
            images = affine_augment(images)
            present = torch.from_numpy(mask[indices]).to(device, non_blocking=True)
            truth = torch.from_numpy(targets[indices]).to(device, non_blocking=True)
            weights = torch.from_numpy(confidence[indices]).to(device, non_blocking=True)
            txt = train_text[sel].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits, img_emb = model(images, present)
                cell_loss = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
                bce = (cell_loss * weights).sum() / weights.sum().clamp_min(1)
                cl = infonce(img_emb, F.normalize(txt, dim=-1), torch.exp(model.clip_temp).item())
                loss = bce + CLIP_WEIGHT * cl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if time.time() - T0 > TIME_LIMIT_S:
                raise TimeoutError("Time budget exhausted during training")

        val_predictions = predict(model, cache, mask, val_indices, device, tta=False)
        val_auc, per_target = macro_auc((targets[val_indices] > 0.5).astype(int), val_predictions)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "silver_scanner_auc": val_auc, "silver_per_target": per_target})
        log(f"epoch {epoch+1}/{EPOCHS}: loss={history[-1]['loss']:.4f}, scanner-val={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_state, history


def main() -> None:
    seed_everything()
    root = find_competition_root()
    label_path = find_single_file(LABELS_FILE, "labels")
    emb_path = find_single_file(EMBEDDINGS_FILE, "text embeddings")
    dino_path = find_backbone()
    base_ckpt_path = find_base_fold0_pt()
    train_df = pd.read_csv(root / "train.csv")
    labels = validate_and_join_labels(train_df, label_path)

    emb_df = pd.read_csv(emb_path).dropna(subset=["StudyInstanceUID"])
    emb_np = emb_df.drop(columns=["StudyInstanceUID"]).to_numpy(dtype=np.float32)
    emb_np = emb_np / np.clip(np.linalg.norm(emb_np, axis=1, keepdims=True), 1e-9, None)
    uid_order = list(emb_df["StudyInstanceUID"].astype(str))
    log(f"text embeddings: {emb_np.shape}")

    train_series_csv = pd.read_csv(root / "train_series.csv")
    train_series = list_series(root / "train_series", train_series_csv)
    train_slots = choose_slots(train_series)
    train_headers = study_headers(train_slots)
    train_sides = laterality_map(train_headers)
    groups = scanner_groups(train_headers)
    studies, train_cache, train_mask = build_cache(train_slots, train_sides, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This diagnostic kernel requires a GPU")
    table = labels.set_index("StudyInstanceUID").loc[studies]
    targets = table[TARGETS].values.astype(np.float32)
    gold_columns = [f"{target}__gold" for target in TARGETS]
    gold_truth = table[gold_columns].values.astype(np.float32)
    gold_indices_all = np.flatnonzero(table["is_gold"].values)

    table, train_indices, val_indices, gold_indices, available_folds = make_clean_split(
        studies, labels, groups, FOLD
    )
    if available_folds != 5:
        raise ValueError(f"Expected 5 scanner folds, found {available_folds}")

    # ---- Side A: existing plain base fold-0 (from checkpoints dataset) ----
    base_backbone = build_backbone(dino_path)
    base_model = KneeDINO(base_backbone)
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    if base_ckpt.get("fold") != FOLD:
        raise ValueError(f"base ckpt fold={base_ckpt.get('fold')}, expected {FOLD}")
    base_model.load_state_dict(base_ckpt["state_dict"], strict=True)
    base_model = base_model.to(device).eval()
    base_val = predict(base_model, train_cache, train_mask, val_indices, device, tta=TTA)
    base_gold = predict(base_model, train_cache, train_mask, gold_indices, device, tta=TTA)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Side B: multimodal trained on the same fold-0 split ----
    seed_everything(SEED + FOLD)
    mm_backbone = build_backbone(dino_path)
    mm_model = KneeDINOMM(mm_backbone)
    mm_model, mm_state, mm_history = train_mm(
        mm_model, train_cache, train_mask, table,
        train_indices, val_indices, gold_indices, device, emb_np, uid_order,
    )
    mm_val = predict(mm_model, train_cache, train_mask, val_indices, device, tta=TTA)
    mm_gold = predict(mm_model, train_cache, train_mask, gold_indices, device, tta=TTA)

    silver_truth = (targets[val_indices] > 0.5).astype(int)
    base_auc, base_pa = macro_auc(silver_truth, base_val)
    mm_auc, mm_pa = macro_auc(silver_truth, mm_val)
    gold_base_auc, _ = macro_auc(gold_truth[gold_indices].astype(int), base_gold)
    gold_mm_auc, _ = macro_auc(gold_truth[gold_indices].astype(int), mm_gold)

    report = {
        "fold": FOLD,
        "val_studies": int(len(val_indices)),
        "gold_studies": int(len(gold_indices)),
        "base_oof_macro_auc": float(base_auc),
        "multimodal_oof_macro_auc": float(mm_auc),
        "delta_macro": float(mm_auc - base_auc),
        "per_target_base": base_pa,
        "per_target_multimodal": mm_pa,
        "per_target_delta": {t: round(mm_pa.get(t, float("nan")) - base_pa.get(t, float("nan")), 4) for t in TARGETS},
        "base_gold_monitor_auc": float(gold_base_auc),
        "multimodal_gold_monitor_auc": float(gold_mm_auc),
        "gold_delta": float(gold_mm_auc - gold_base_auc),
        "multimodal_training_history": mm_history,
        "clip_weight": CLIP_WEIGHT,
        "elapsed_seconds": time.time() - T0,
        "status": "DIAGNOSTIC_COMPLETE",
    }
    Path("diagnostic.json").write_text(json.dumps(report, indent=2, default=float))
    log("=" * 70)
    log(f"BASE       OOF macro = {base_auc:.4f} | gold = {gold_base_auc:.4f}")
    log(f"MULTIMODAL OOF macro = {mm_auc:.4f} | gold = {gold_mm_auc:.4f}")
    log(f"DELTA      OOF = {mm_auc - base_auc:+.4f} | gold = {gold_mm_auc - gold_base_auc:+.4f}")
    log("per-target delta:")
    for t in TARGETS:
        log(f"  {t:18s} {base_pa.get(t, 0):.4f} -> {mm_pa.get(t, 0):.4f} "
            f"({mm_pa.get(t, 0) - base_pa.get(t, 0):+.4f})")
    log(f"complete in {(time.time() - T0) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
