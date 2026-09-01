"""Combined ensemble inference kernel for RSNA Knee MRI.

Loads all complete five-fold checkpoint packages (dino2b, dino2b_s2, dino2b_mm6)
found under /kaggle/input, decodes the TEST set once, and rank-mean averages
per-member fold predictions (with label-aware horizontal-flip TTA) into the
final submission.csv.

Two model architectures are supported:
  * Standard KneeDINO (KneeDINO + SlotHead): checkpoints without text_proj
  * Multimodal KneeDINO (adds text_proj + clip_temp): checkpoints with text_proj

Integrity gates hard-fail; a member whose package is incomplete or whose
preprocessing contract mismatches is skipped and recorded loudly in the audit
(never silently blended).
"""

from __future__ import annotations

import gc
import json
import math
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
CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
GROUP_SIZE = 3
N_GROUPS = 2
CACHE_SLICES = GROUP_SIZE * N_GROUPS
EVAL_BATCH = 4
UNFREEZE_LAST = 4
HEADER_THREADS = 16
DECODE_THREADS = 10
TIME_LIMIT_S = 8.0 * 3600
TTA = True

CHECKPOINT_PATTERN = re.compile(r"(.+)_fold([0-4])\.pt$")
N_FOLDS = 5

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


def find_backbone_dir(model_key: str) -> Path:
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        text = root.lower()
        if "config.json" in files and "dinov2" in text and model_key in text:
            hits.append(Path(root))
    if not hits:
        raise FileNotFoundError(f"Attached DINOv2-{model_key} model was not found")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def discover_members() -> dict[str, dict]:
    """Find complete five-fold checkpoint packages keyed by member tag."""
    packages: dict[str, dict[int, Path]] = {}
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        for filename in files:
            match = CHECKPOINT_PATTERN.fullmatch(filename)
            if match:
                tag, fold = match.group(1), int(match.group(2))
                packages.setdefault(tag, {})[fold] = Path(root) / filename
    members = {}
    for tag, folds in packages.items():
        if set(folds) == set(range(N_FOLDS)):
            members[tag] = {fold: folds[fold] for fold in range(N_FOLDS)}
        else:
            log(f"SKIP member '{tag}': incomplete folds {sorted(folds)}")
    if not members:
        raise RuntimeError("No complete member checkpoint package found")
    log(f"members discovered: {sorted(members)}")
    return members


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
    """Combined standard + multimodal architecture.

    If text_dim is provided and > 0, also builds text_proj + clip_temp
    (multimodal variant). state_dict loading tolerates missing/extra keys.
    """
    def __init__(self, backbone: nn.Module, text_dim: int = 0):
        super().__init__()
        self.backbone = backbone
        feature_dim = backbone.config.hidden_size * 2
        self.head = SlotHead(feature_dim, len(SLOTS), len(TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.text_dim = int(text_dim)
        if self.text_dim > 0:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, self.text_dim)
            )
            self.clip_temp = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, images: torch.Tensor, mask: torch.Tensor):
        batch, slots = images.shape[:2]
        pixels = images.reshape(batch * slots, *images.shape[2:]).float().div_(255.0)
        pixels = (pixels - self.mean) / self.std
        output = self.backbone(pixel_values=pixels).last_hidden_state
        features = torch.cat([output[:, 0], output[:, 1:].mean(1)], dim=1)
        features = features.reshape(batch, slots, -1)
        logits = self.head(features, mask)
        if self.text_dim > 0:
            pooled = features.mean(dim=1)
            text_emb = F.normalize(self.text_proj(pooled), dim=-1)
            return logits, text_emb
        return logits, None


def build_model(model_path: Path, text_dim: int = 0) -> KneeDINO:
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
    return KneeDINO(backbone, text_dim=text_dim)


def load_member_model(model_key: str, checkpoint_path: Path, expected_fold: int, device) -> KneeDINO:
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
        "model_key": model_key,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"fold {expected_fold} has {key}={checkpoint.get(key)!r}, expected {expected!r}"
            )
    text_dim = int(checkpoint.get("text_dim", 0) or 0)
    model_path = find_backbone_dir(model_key)
    model = build_model(model_path, text_dim=text_dim)
    state = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        log(f"  fold {expected_fold}: unexpected keys={unexpected[:4]}")
    if missing and not (text_dim == 0 and all(k.startswith("text_proj") or k == "clip_temp" for k in missing)):
        log(f"  fold {expected_fold}: missing keys={missing[:4]}")
    return model.to(device).eval()


def rank_predictions(predictions: np.ndarray) -> np.ndarray:
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
                logits, _ = model(images, present)
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
                    logits, _ = model(images, present)
                outputs.append(torch.sigmoid(logits).float().cpu().numpy())
            flipped.append(np.concatenate(outputs))
        flipped_probs = np.mean(flipped, axis=0)[:, FLIP_SWAP]
        probs = (probs + flipped_probs) / 2.0
    return probs


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
    members = discover_members()

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
    test_studies, test_cache, test_mask = build_cache(test_slots, test_sides, "test")

    member_predictions = {}
    skipped = {}
    member_arch = {}
    for tag, folds in sorted(members.items()):
        try:
            fold_preds = []
            arch = None
            for fold in range(N_FOLDS):
                checkpoint_path = folds[fold]
                ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model_key = ck.get("model_key")
                if model_key is None:
                    raise ValueError(f"{tag} fold {fold} has no model_key")
                model = load_member_model(model_key, checkpoint_path, fold, device)
                preds = predict(
                    model, test_cache, test_mask, np.arange(len(test_studies)), device,
                    tta=TTA,
                )
                fold_preds.append(preds)
                arch = "multimodal" if getattr(model, "text_dim", 0) > 0 else "standard"
                del model
                gc.collect()
                torch.cuda.empty_cache()
            member_predictions[tag] = np.mean(fold_preds, axis=0)
            member_arch[tag] = arch
            log(f"member '{tag}' [{arch}]: {N_FOLDS} folds averaged (TTA={TTA})")
        except Exception as exc:
            skipped[tag] = f"{type(exc).__name__}: {exc}"
            log(f"SKIP member '{tag}': {skipped[tag]}")

    if not member_predictions:
        raise RuntimeError("No member produced test predictions")

    if len(member_predictions) == 1:
        tag = next(iter(member_predictions))
        final = member_predictions[tag]
        mode = f"single_member={tag}"
    else:
        ranked = [rank_predictions(p) for p in member_predictions.values()]
        final = np.mean(ranked, axis=0)
        mode = f"rank_mean_over_{len(member_predictions)}_members"

    output = write_submission(final, test_studies, test_df)

    audit = {
        "mode": "combined-ensemble-inference-v2",
        "members": sorted(member_predictions),
        "member_arch": member_arch,
        "skipped": skipped,
        "ensemble_mode": mode,
        "tta": TTA,
        "test_studies": len(test_studies),
        "submission_rows": int(len(output)),
        "elapsed_seconds": time.time() - T0,
        "status": "COMBINED_ENSEMBLE_SUBMISSION_WRITTEN",
    }
    Path("combined_ensemble_audit.json").write_text(json.dumps(audit, indent=2, default=float))
    log("combined_ensemble_audit.json:\n" + json.dumps(audit, indent=2))
    log(f"complete in {(time.time() - T0) / 3600:.2f} hours")


if __name__ == "__main__":
    main()