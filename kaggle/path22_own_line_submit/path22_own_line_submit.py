"""Path22 v2 — inference-only submission vehicle for the Path20 stage-0 line.

The Path20 training kernel (``aakashkavuru/rsna-knee-path20-gold-backbone-train``)
hit its time guard (PATH20_PARTIAL_TIMEOUT) before the gold-at-backbone
deployment stage ran, but its stage-0 pass completed and passed its own
quality gate. Path20's own output is unsubmittable (638 min > 540 min GPU
cap), so this kernel is the submittable vehicle: it loads the five stage-0
(gold-free) 336 px checkpoints ``clean_dino_fold{0..4}.pt`` from Path20's
output, loads the five 224 px parent-baseline checkpoints from the
``aakashkavuru/rsna-knee-clean-dinov2-full`` kernel output, reproduces the
gated baseline/new rank blend exactly, and writes ``submission.csv``.

Preprocessing, model, and predict code below is copied verbatim from the
training script. Integrity gates hard-fail; the single sanctioned fallback
is baseline-kernel unavailability, which degrades to the new-folds-only
ensemble and is recorded loudly in the audit.

Gates:
- exactly one complete five-fold ``clean_dino_fold[0-4].pt`` set inside the
  Path20 kernel output directory (disambiguated from the baseline kernel's
  identically named files by directory slug),
- per-checkpoint preprocessing metadata equals this kernel's constants
  (img_size, crop_mm, slice_band, group_size, n_groups, cache_slices,
  targets, slots), gold_training_count == 0, label_source ==
  "silver_labels_v5.csv", state_dict loads with strict=True,
- metrics.json next to the checkpoints passes the stage-0 quality gate:
  best_blend_oof_auc >= baseline_oof_auc + minimum_required_oof_gain.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
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


T0 = time.time()
SEED = 20260813
IMG_SIZE = 336
BASELINE_IMG_SIZE = 224
CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
GROUP_SIZE = 3
N_GROUPS = 2
CACHE_SLICES = GROUP_SIZE * N_GROUPS
EVAL_BATCH = 4
UNFREEZE_LAST = 4
HEADER_THREADS = 16
DECODE_THREADS = 10
TIME_LIMIT_S = 8.0 * 3600  # submission cap is 9 h

# --- Path22 v2: stage-0 checkpoint contract ---
CHECKPOINT_PATTERN = re.compile(r"clean_dino_fold([0-4])\.pt$")
N_FOLDS = 5
PATH20_SLUG = "path20-gold-backbone-train"
BASELINE_SLUG = "clean-dinov2-full"
EXPECTED_GOLD_TRAINING_COUNT = 0
EXPECTED_LABEL_SOURCE = "silver_labels_v5.csv"
METRICS_FILE = "metrics.json"
PATH20_AUDIT_FILE = "path20_audit.json"

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
    import random

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


def find_fold_checkpoints(slug: str, label: str) -> list[Path]:
    """Find one complete five-fold clean_dino package under the given kernel slug.

    Both the Path20 output and the baseline kernel output ship files named
    clean_dino_fold{0..4}.pt; the directory path fragment keeps them apart.
    """
    candidates: dict[Path, dict[int, Path]] = {}
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if slug not in root:
            continue
        for filename in files:
            match = CHECKPOINT_PATTERN.fullmatch(filename)
            if match:
                candidates.setdefault(Path(root), {})[int(match.group(1))] = Path(root) / filename
    complete = [files for files in candidates.values() if set(files) == set(range(N_FOLDS))]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"Expected one complete five-fold {label} checkpoint package under "
            f"*{slug}*, found {len(complete)}"
        )
    return [complete[0][fold] for fold in range(N_FOLDS)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics(checkpoint_paths: list[Path]) -> dict:
    """Load metrics.json next to the stage-0 checkpoints and apply the gate."""
    checkpoint_dir = checkpoint_paths[0].parent
    metrics_path = checkpoint_dir / METRICS_FILE
    if not metrics_path.is_file():
        matches = [
            path for path in checkpoint_dir.parent.rglob(METRICS_FILE)
            if PATH20_SLUG in str(path)
        ] if checkpoint_dir.parent.is_dir() else []
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected {METRICS_FILE} next to the stage-0 checkpoints, "
                f"fallback search found {len(matches)}"
            )
        metrics_path = matches[0]
    metrics = json.loads(metrics_path.read_text())
    best_blend = float(metrics["best_blend_oof_auc"])
    baseline_oof = float(metrics["baseline_oof_auc"])
    min_gain = float(metrics["minimum_required_oof_gain"])
    if not best_blend >= baseline_oof + min_gain:
        raise RuntimeError(
            f"stage-0 quality gate failed: best_blend_oof_auc={best_blend:.6f} < "
            f"baseline_oof_auc={baseline_oof:.6f} + min_gain={min_gain:.6f}"
        )
    blend_weight = float(metrics["new_model_blend_weight"])
    if not 0.0 <= blend_weight <= 1.0:
        raise RuntimeError(f"metrics.json blend weight {blend_weight} outside [0, 1]")
    log(
        f"metrics gate passed: blend={best_blend:.4f} >= baseline={baseline_oof:.4f} "
        f"+ {min_gain:.3f}; new_model_blend_weight={blend_weight}"
    )
    return metrics


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


def load_stage0_model(
    model_path: Path, checkpoint_path: Path, expected_fold: int, device: torch.device
) -> KneeDINO:
    """Load one Path20 stage-0 (336 px, gold-free) checkpoint after all gates."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "fold": expected_fold,
        "img_size": IMG_SIZE,
        "crop_mm": CROP_MM,
        "slice_band": SLICE_BAND,
        "group_size": GROUP_SIZE,
        "n_groups": N_GROUPS,
        "cache_slices": CACHE_SLICES,
        "targets": TARGETS,
        "slots": SLOTS,
        "gold_training_count": EXPECTED_GOLD_TRAINING_COUNT,
        "label_source": EXPECTED_LABEL_SOURCE,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Stage-0 fold {expected_fold} has {key}={checkpoint.get(key)!r}, "
                f"expected {expected!r}"
            )
    model = build_model(model_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    log(f"stage-0 fold {expected_fold}: gates passed")
    return model.to(device).eval()


def load_baseline_model(
    model_path: Path, checkpoint_path: Path, expected_fold: int, device: torch.device
) -> KneeDINO:
    """Load one 224 px parent-baseline checkpoint (same gate as the training script)."""
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
    dino_path = find_dinov2_small()
    stage0_paths = find_fold_checkpoints(PATH20_SLUG, "stage-0")
    metrics = load_metrics(stage0_paths)
    blend_weight = float(metrics["new_model_blend_weight"])
    stage0_sha256 = {path.name: _sha256(path) for path in stage0_paths}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This submission kernel requires a GPU")

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

    prediction_hashes = {}

    # Stage-0 (336 px) fold predictions.
    fold_test_predictions = []
    for fold, checkpoint_path in enumerate(stage0_paths):
        model = load_stage0_model(dino_path, checkpoint_path, fold, device)
        predictions = predict(
            model, test_cache, test_mask, np.arange(len(test_studies)), device
        )
        fold_test_predictions.append(predictions)
        prediction_hashes[f"stage0/{checkpoint_path.name}"] = hashlib.sha256(
            np.ascontiguousarray(predictions).tobytes()
        ).hexdigest()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Parent-baseline (224 px) fold predictions, reproduced from the same
    # normalized volumes. If the baseline kernel's checkpoints do not load
    # cleanly, degrade to the new-folds-only ensemble and say so loudly.
    baseline_test_predictions = []
    baseline_sha256 = {}
    baseline_deviation = None
    try:
        baseline_paths = find_fold_checkpoints(BASELINE_SLUG, "baseline")
        baseline_sha256 = {path.name: _sha256(path) for path in baseline_paths}
        for fold, checkpoint_path in enumerate(baseline_paths):
            baseline_model = load_baseline_model(dino_path, checkpoint_path, fold, device)
            predictions = predict(
                baseline_model, test_baseline_cache, test_mask,
                np.arange(len(test_studies)), device,
            )
            baseline_test_predictions.append(predictions)
            prediction_hashes[f"baseline/{checkpoint_path.name}"] = hashlib.sha256(
                np.ascontiguousarray(predictions).tobytes()
            ).hexdigest()
            del baseline_model
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        baseline_deviation = (
            f"BASELINE_UNAVAILABLE_NEW_FOLDS_ONLY: {type(exc).__name__}: {exc}"
        )
        log(f"WARNING: {baseline_deviation}")
        baseline_test_predictions = []

    # Blend exactly as the training script did: per-fold ranks averaged and
    # re-ranked per family, then the gated (1-w)/w rank blend; the final
    # rank transform happens inside write_submission.
    new_test_ensemble = rank_predictions(np.mean(
        [rank_predictions(p) for p in fold_test_predictions], axis=0
    ))
    if baseline_test_predictions:
        baseline_test_ensemble = rank_predictions(
            np.mean([rank_predictions(p) for p in baseline_test_predictions], axis=0)
        )
        test_ensemble = (
            (1.0 - blend_weight) * baseline_test_ensemble
            + blend_weight * new_test_ensemble
        )
        ensemble_mode = f"blend_new_weight={blend_weight}"
    else:
        test_ensemble = new_test_ensemble
        ensemble_mode = "new_folds_only_baseline_unavailable"
    output = write_submission(test_ensemble, test_studies, test_df)

    audit = {
        "mode": "path22-stage0-own-line-submit",
        "parents": {
            "stage0_checkpoints": f"aakashkavuru/rsna-knee-{PATH20_SLUG}",
            "baseline_checkpoints": f"aakashkavuru/rsna-knee-{BASELINE_SLUG}",
        },
        "stage0_checkpoints": stage0_sha256,
        "baseline_checkpoints": baseline_sha256,
        "baseline_deviation": baseline_deviation,
        "ensemble_mode": ensemble_mode,
        "blend_weight_new": blend_weight,
        "metrics_json": metrics,
        "prediction_hashes": prediction_hashes,
        "ensemble_sha256": hashlib.sha256(
            np.ascontiguousarray(test_ensemble).tobytes()
        ).hexdigest(),
        "submission_sha256": _sha256(Path("submission.csv")),
        "submission_rows": int(len(output)),
        "test_studies": len(test_studies),
        "gold_training_count": EXPECTED_GOLD_TRAINING_COUNT,
        "label_source": EXPECTED_LABEL_SOURCE,
        "elapsed_seconds": time.time() - T0,
        "status": "PATH22_STAGE0_OWN_LINE_SUBMIT_WRITTEN",
    }
    audit_text = json.dumps(audit, indent=2, default=float)
    Path("path22_audit.json").write_text(audit_text)
    log("path22_audit.json:\n" + audit_text)
    log(f"complete in {(time.time() - T0) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
