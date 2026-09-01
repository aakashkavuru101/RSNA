"""OOF harvest — regenerate honest out-of-fold predictions for a member.

Loads one complete five-fold checkpoint package (default: dino2b folds from
the `rsna-knee-ensemble-checkpoints` dataset), decodes the TRAIN cache once,
rebuilds the deterministic scanner-grouped splits, and predicts each fold's
model on its own validation studies (plus the 58 gold as a monitor). Writes
<member>_oof.csv (StudyInstanceUID + 12 targets, OOF rows filled; gold rows
filled with the gold-folds-ensemble prediction where available).

This is predict-only (~2h): the base training run completed all five folds
but errored on a guard before its OOF CSV was written; this kernel recovers
the validation signal that the consensus validator consumes.
"""

from __future__ import annotations

import gc
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
IMG_SIZE = 336
CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
GROUP_SIZE = 3
N_GROUPS = 2
CACHE_SLICES = GROUP_SIZE * N_GROUPS
EVAL_BATCH = 4
UNFREEZE_LAST = 4
HEADER_THREADS = 16
DECODE_THREADS = 10
TIME_LIMIT_S = 6.5 * 3600

MEMBER_TAG = "dino2b"
MODEL_KEY = "base"
LABELS_FILE = "silver_labels_v5.csv"
TTA = True

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


def find_member_checkpoints() -> dict[int, Path]:
    pattern = re.compile(rf"{re.escape(MEMBER_TAG)}_fold([0-4])\.pt$")
    found: dict[int, Path] = {}
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        for filename in files:
            match = pattern.fullmatch(filename)
            if match:
                found[int(match.group(1))] = Path(root) / filename
    if set(found) != set(range(5)):
        raise FileNotFoundError(f"incomplete {MEMBER_TAG} folds: {sorted(found)}")
    return {fold: found[fold] for fold in range(5)}


def validate_and_join_labels(train_df: pd.DataFrame, label_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(label_path)
    merged = train_df[["StudyInstanceUID", "Report"] + TARGETS].merge(
        labels, on="StudyInstanceUID", how="left", validate="one_to_one",
        suffixes=("__gold", ""),
    )
    if merged[TARGETS].isna().any().any():
        raise ValueError("labels do not cover every training study")
    gold_cols = [f"{target}__gold" for target in TARGETS]
    merged["is_gold"] = merged[gold_cols].notna().any(axis=1)
    if int(merged["is_gold"].sum()) != 58:
        raise ValueError("expected 58 gold studies")
    log(f"labels: {len(merged)} studies, source={label_path.name}")
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
    log(f"filesystem: {len(out)} series for {out.StudyInstanceUID.nunique()} studies")
    return out


def choose_slots(series_df: pd.DataFrame) -> dict[str, list[dict | None]]:
    output = {}
    for study, group in series_df.groupby("StudyInstanceUID", sort=True):
        chosen = []
        for _, plane, fluid in SLOTS:
            candidates = group[
                (group["Anatomical_Plane"] == plane) & (group["Fluid_Sensitive"] == fluid)
            ]
            if candidates.empty:
                chosen.append(None)
            else:
                row = candidates.sort_values(
                    ["n_slices", "SeriesInstanceUID"], ascending=[False, True]
                ).iloc[0]
                chosen.append(row.to_dict())
        output[study] = chosen
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
    log(f"headers: {len(result)} studies")
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
    for row in headers.itertuples(index=False):
        values = [getattr(row, "Laterality", None), getattr(row, "ImageLaterality", None)]
        tags = [str(v).strip().upper()[:1] for v in values if v is not None]
        side = next((v for v in tags if v in ("L", "R")), None)
        if side is None:
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
        output[row.StudyInstanceUID] = side
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


def load_member_model(model_key: str, checkpoint_path: Path, expected_fold: int, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("fold") != expected_fold:
        raise ValueError(f"fold mismatch: {checkpoint.get('fold')} != {expected_fold}")
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        text = root.lower()
        if "config.json" in files and "dinov2" in text and model_key in text:
            hits.append(Path(root))
    if not hits:
        raise FileNotFoundError(f"DINOv2-{model_key} not found")
    model_path = sorted(hits, key=lambda p: len(str(p)))[0]
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model = KneeDINO(backbone)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


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
                logits = model(images, present)
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
                    logits = model(images, present)
                outputs.append(torch.sigmoid(logits).float().cpu().numpy())
            flipped.append(np.concatenate(outputs))
        flipped_probs = np.mean(flipped, axis=0)[:, FLIP_SWAP]
        probs = (probs + flipped_probs) / 2.0
    return probs


def make_split(studies: list[str], labels: pd.DataFrame, groups: dict[str, str]):
    table = labels.set_index("StudyInstanceUID").loc[studies]
    eligible = np.flatnonzero(~table["is_gold"].values)
    group_values = np.array([groups.get(study, "unknown") for study in studies])
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(eligible, groups=group_values[eligible]))
    return table, eligible, splits, group_values


def main() -> None:
    seed_everything()
    root = find_competition_root()
    label_path = find_single_file(LABELS_FILE, "labels")
    train_df = pd.read_csv(root / "train.csv")
    labels = validate_and_join_labels(train_df, label_path)
    checkpoints = find_member_checkpoints()

    train_series_csv = pd.read_csv(root / "train_series.csv")
    train_series = list_series(root / "train_series", train_series_csv)
    train_slots = choose_slots(train_series)
    train_headers = study_headers(train_slots)
    train_sides = laterality_map(train_headers)
    groups = scanner_groups(train_headers)
    studies, train_cache, train_mask = build_cache(train_slots, train_sides, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, eligible, splits, group_values = make_split(studies, labels, groups)
    gold_indices = np.flatnonzero(table["is_gold"].values)
    targets = table[TARGETS].values.astype(np.float32)
    gold_columns = [f"{target}__gold" for target in TARGETS]
    gold_truth = table[gold_columns].values.astype(np.float32)

    oof = np.full((len(studies), len(TARGETS)), np.nan, dtype=np.float32)
    fold_of = np.full(len(studies), -1, dtype=np.int64)
    gold_preds = []
    for fold in range(5):
        train_local, val_local = splits[fold]
        val_indices = eligible[val_local]
        model = load_member_model(MODEL_KEY, checkpoints[fold], fold, device)
        oof[val_indices] = predict(model, train_cache, train_mask, val_indices, device, tta=TTA)
        fold_of[val_indices] = fold
        gold_preds.append(predict(model, train_cache, train_mask, gold_indices, device, tta=TTA))
        log(f"fold {fold}: predicted {len(val_indices)} val studies")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if not np.isfinite(oof[eligible]).all():
        raise RuntimeError("OOF predictions do not cover every non-gold study")

    oof_auc, oof_per_target = {}, {}
    vals = []
    for i, t in enumerate(TARGETS):
        if len(np.unique((targets[eligible] > 0.5).astype(int)[:, i])) == 2:
            a = float(roc_auc_score((targets[eligible] > 0.5).astype(int)[:, i], oof[eligible][:, i]))
            oof_per_target[t] = a
            vals.append(a)
    oof_auc = float(np.mean(vals))
    gold_ens = np.mean(gold_preds, axis=0)
    gold_auc, gold_per_target = {}, []
    gv = []
    for i, t in enumerate(TARGETS):
        if len(np.unique(gold_truth[gold_indices].astype(int)[:, i])) == 2:
            a = float(roc_auc_score(gold_truth[gold_indices].astype(int)[:, i], gold_ens[:, i]))
            gold_per_target.append((t, a))
            gv.append(a)
    gold_auc = float(np.mean(gv))
    log(f"silver OOF macro={oof_auc:.4f}; gold ensemble monitor={gold_auc:.4f}")

    frame = pd.DataFrame(oof, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", np.asarray(studies, dtype=str))
    frame["fold"] = fold_of
    frame["is_gold"] = table["is_gold"].values.astype(int)
    frame.to_csv(f"{MEMBER_TAG}_oof.csv", index=False)
    Path("harvest_metrics.json").write_text(json.dumps({
        "member": MEMBER_TAG,
        "silver_oof_macro_auc": oof_auc,
        "silver_oof_per_target": oof_per_target,
        "gold_ensemble_monitor_auc": gold_auc,
        "gold_ensemble_per_target": dict(gold_per_target),
        "elapsed_seconds": time.time() - T0,
    }, indent=2))
    log(f"wrote {MEMBER_TAG}_oof.csv; complete in {(time.time() - T0)/3600:.2f}h")


if __name__ == "__main__":
    main()
