"""Inference-only hidden-test scorer for the validated RSNA Knee ensemble."""

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

import __main__ as p


ROUTED_SIZE = 280
LOCALIZED_SIZE = 336
GROUP_SIZE = 3
LOCALIZED_CENTRES = (0.35, 0.65)
ROUTED_CENTRES = (0.25, 0.50, 0.75)
SLICE_BAND = (0.20, 0.80)
DECODE_THREADS = 10
HEADER_THREADS = 16
EVAL_BATCH = 4
SCORED_WEIGHTS = (0.20, 0.80)

ROUTED_SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
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
HEADER_TAGS = [
    "Laterality", "ImageLaterality", "ImagePositionPatient", "ImageOrientationPatient",
    "PixelSpacing", "Rows", "Columns", "SeriesDescription", "SequenceName",
    "ScanOptions", "ScanningSequence", "RepetitionTime", "EchoTime",
]


def checkpoint_family(marker: str, filename_prefix: str) -> list[Path]:
    packages: dict[Path, dict[int, Path]] = {}
    pattern = re.compile(re.escape(filename_prefix) + r"([0-4])\.pt$")
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if marker not in root.lower():
            continue
        for filename in files:
            match = pattern.fullmatch(filename)
            if match:
                packages.setdefault(Path(root), {})[int(match.group(1))] = Path(root) / filename
    complete = [family for family in packages.values() if set(family) == set(range(5))]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"Expected one complete {filename_prefix!r} family containing {marker!r}; "
            f"found {len(complete)}"
        )
    return [complete[0][fold] for fold in range(5)]


def unique_json(marker: str, filename: str) -> tuple[Path, dict]:
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if marker in root.lower() and filename in files:
            hits.append(Path(root) / filename)
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected one {filename!r} containing {marker!r}; found {len(hits)}"
        )
    return hits[0], json.loads(hits[0].read_text())


def _series_header_probe(record: dict) -> dict:
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
                result[tag] = "|".join(str(item) for item in value)
            else:
                result[tag] = str(value)
    except Exception as exc:
        result["header_error"] = str(exc)[:160]
    return result


def annotate_series(series: pd.DataFrame) -> pd.DataFrame:
    records = series.to_dict("records")
    with ThreadPoolExecutor(max_workers=HEADER_THREADS) as pool:
        headers = pd.DataFrame(pool.map(_series_header_probe, records))
    result = series.merge(
        headers, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left",
        validate="one_to_one",
    )
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
    t1, t2, pdw = text.str.contains(T1_RE), text.str.contains(T2_RE), text.str.contains(PD_RE)
    result["weight"] = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(t2 & ~pdw, "T2", np.where(
            pdw, "PD", np.where(gre, "GRE", np.where(
                tr < 800, "T1", np.where(te > 60, "T2", np.where(tr >= 800, "PD", "UNK"))
            ))
        )),
    )
    result["fluid_recovered"] = result.weight.isin(["PD", "T2"])
    p.log(f"routed headers: {len(result)} series")
    return result


def choose_routed_slots(series: pd.DataFrame) -> dict[str, list[dict | None]]:
    output = {}
    for study, group in series.groupby("StudyInstanceUID", sort=True):
        slots = []
        for _, plane, fluid, fatsat in ROUTED_SLOTS:
            candidates = group[
                (group.Anatomical_Plane == plane)
                & (group.fluid_recovered == fluid)
                & (group.fatsat == fatsat)
            ]
            if candidates.empty:
                slots.append(None)
            else:
                slots.append(candidates.sort_values(
                    ["n_slices", "SeriesInstanceUID"], ascending=[False, True]
                ).iloc[0].to_dict())
        output[study] = slots
    return output


def _series_numbers(value, n: int) -> np.ndarray | None:
    if not isinstance(value, str):
        return None
    try:
        values = np.asarray([float(item) for item in value.split("|")], dtype=np.float64)
    except ValueError:
        return None
    return values if len(values) >= n and np.isfinite(values[:n]).all() else None


def robust_laterality(headers: pd.DataFrame) -> dict[str, str | None]:
    output = {}
    for study, group in headers.groupby("StudyInstanceUID", sort=True):
        tags, geometry = [], []
        for row in group.itertuples(index=False):
            values = [getattr(row, "Laterality", None), getattr(row, "ImageLaterality", None)]
            tags.extend(
                side for side in [str(value).strip().upper()[:1] for value in values if value]
                if side in ("L", "R")
            )
            ipp = _series_numbers(getattr(row, "ImagePositionPatient", None), 3)
            iop = _series_numbers(getattr(row, "ImageOrientationPatient", None), 6)
            spacing = _series_numbers(getattr(row, "PixelSpacing", None), 2)
            try:
                rows, cols = float(getattr(row, "Rows")), float(getattr(row, "Columns"))
                centre = ipp[:3] + iop[:3] * spacing[1] * cols / 2 + iop[3:6] * spacing[0] * rows / 2
                if abs(centre[0]) >= 20:
                    geometry.append("R" if centre[0] < 0 else "L")
            except (TypeError, ValueError, IndexError):
                pass
        output[study] = (
            pd.Series(tags).mode().iloc[0] if tags else
            (pd.Series(geometry).mode().iloc[0] if geometry else None)
        )
    return output


def _read_contiguous(job):
    study, slot_index, record, side, centres, size, slot_defs = job
    files = p.ordered_files(record)
    count = len(files)
    lo, hi = int(SLICE_BAND[0] * (count - 1)), int(SLICE_BAND[1] * (count - 1))
    indices = []
    for fraction in centres:
        centre = int(np.clip(round(fraction * (count - 1)), lo, max(lo, hi)))
        indices.extend(int(np.clip(centre + offset, 0, count - 1)) for offset in (-1, 0, 1))
    images, spacing = [], None
    for index in indices:
        try:
            ds = pydicom.dcmread(os.path.join(record["dir"], files[index]), force=True)
            image = ds.pixel_array.astype(np.float32)
            image = image * float(getattr(ds, "RescaleSlope", 1) or 1)
            image += float(getattr(ds, "RescaleIntercept", 0) or 0)
            if getattr(ds, "PixelSpacing", None) is not None:
                spacing = float(ds.PixelSpacing[0])
            images.append(image)
        except Exception:
            images.append(None)
    good = [index for index, image in enumerate(images) if image is not None]
    if not good:
        return study, slot_index, None
    for index, image in enumerate(images):
        if image is None:
            images[index] = images[min(good, key=lambda other: abs(other - index))]
    shape = images[0].shape
    images = [image if image.shape == shape else np.zeros(shape, np.float32) for image in images]
    volume = np.stack(images)
    if spacing is not None and np.isfinite(spacing) and spacing > 0:
        wanted = int(round(p.CROP_MM / spacing))
        height, width = shape
        if 16 < wanted < min(height, width):
            cy, cx, half = height // 2, width // 2, wanted // 2
            volume = volume[:, cy-half:cy+half, cx-half:cx+half]
    low, high = np.percentile(volume, [1, 99])
    volume = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    tensor = F.interpolate(tensor, (size, size), mode="bilinear", align_corners=False)
    output = (tensor.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    plane = slot_defs[slot_index][1]
    if side == "R":
        if plane in ("Coronal", "Axial"):
            output = output[:, :, ::-1].copy()
        elif plane == "Sagittal":
            output = output[::-1].copy()
    return study, slot_index, output


def build_contiguous_cache(slot_map, sides, centres, size, slot_defs, tag):
    studies = sorted(slot_map)
    index = {study: position for position, study in enumerate(studies)}
    slices = GROUP_SIZE * len(centres)
    cache = np.zeros((len(studies), len(slot_defs), slices, size, size), dtype=np.uint8)
    mask = np.zeros((len(studies), len(slot_defs)), dtype=np.float32)
    jobs = [
        (study, slot_index, record, sides.get(study), centres, size, slot_defs)
        for study, slots in slot_map.items()
        for slot_index, record in enumerate(slots) if record is not None
    ]
    with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
        for done, (study, slot_index, output) in enumerate(pool.map(_read_contiguous, jobs), 1):
            if output is not None:
                cache[index[study], slot_index] = output
                mask[index[study], slot_index] = 1
            if done % 1000 == 0:
                p.log(f"{tag}: decoded {done}/{len(jobs)}")
    if np.any(mask.sum(1) == 0):
        raise RuntimeError(f"{tag}: study without a decodable slot")
    p.log(f"{tag}: cache {cache.nbytes / 1024**3:.2f} GiB")
    return studies, cache, mask


class LocalizedSlotHead(nn.Module):
    def __init__(self, dim, n_slots, n_outputs, hidden=256):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_embedding = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.target_embedding = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.slot_query = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.dropout = nn.Dropout(0.2)
        self.output_weight = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.output_bias = nn.Parameter(torch.zeros(n_outputs))
        self.hidden = hidden

    def forward(self, features, mask):
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[None, :, None, :]
        hidden = hidden + self.target_embedding[None, None, :, :]
        attention = torch.einsum("bsoh,oh->bos", hidden, self.slot_query) / self.hidden**0.5
        attention = attention.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        context = self.dropout(torch.einsum("bos,bsoh->boh", attention, hidden))
        return (context * self.output_weight.unsqueeze(0)).sum(-1) + self.output_bias


class LocalizedDINO(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        dim = backbone.config.hidden_size
        self.patch_query = nn.Parameter(torch.randn(len(p.TARGETS), dim) * 0.02)
        self.head = LocalizedSlotHead(dim * 2, 6, len(p.TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, images, mask):
        batch, slots = images.shape[:2]
        pixels = images.reshape(batch * slots, *images.shape[2:]).float().div_(255.0)
        tokens = self.backbone(pixel_values=(pixels-self.mean)/self.std).last_hidden_state
        cls, patches = tokens[:,0], tokens[:,1:]
        attention = torch.einsum("npd,od->nop", patches, self.patch_query)
        attention = attention.div_(patches.shape[-1] ** 0.5).softmax(-1)
        localized = torch.einsum("nop,npd->nod", attention, patches)
        cls = cls[:,None,:].expand(-1,len(p.TARGETS),-1)
        features = torch.cat([cls,localized],-1).reshape(batch,slots,len(p.TARGETS),-1)
        return self.head(features,mask)


def load_localized(dino_path, checkpoint_path, fold, architecture, device):
    from transformers import AutoModel
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"fold": fold, "architecture": architecture, "gold_training_count": 0}
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"{checkpoint_path.name}: invalid {key}")
    model = LocalizedDINO(AutoModel.from_pretrained(str(dino_path), local_files_only=True))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_contiguous(model, cache, mask, device, routed=False):
    groups = cache.shape[2] // GROUP_SIZE
    group_outputs = []
    indices = np.arange(len(cache))
    for group in range(groups):
        outputs = []
        for start in range(0, len(indices), EVAL_BATCH):
            selected = indices[start:start+EVAL_BATCH]
            images = torch.from_numpy(
                cache[selected, :, group*GROUP_SIZE:(group+1)*GROUP_SIZE]
            ).to(device, non_blocking=True)
            present = torch.from_numpy(mask[selected]).to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, present)
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
        group_outputs.append(np.concatenate(outputs))
    stacked = np.stack(group_outputs)
    result = stacked.mean(0)
    if routed:
        for target, mode in WINDOW_POOL.items():
            index = p.TARGETS.index(target)
            values = stacked[:,:,index]
            if mode == "max":
                result[:,index] = values.max(0)
            elif mode == "top2":
                result[:,index] = np.sort(values,axis=0)[-min(2,len(values)):].mean(0)
    return result


def ranked_fold_ensemble(predictions):
    return p.rank_predictions(np.mean([p.rank_predictions(value) for value in predictions], axis=0))


def main():
    p.seed_everything()
    root, dino_path = p.find_competition_root(), p.find_dinov2_small()
    checkpoints_224 = checkpoint_family("clean-dinov2-full", "clean_dino_fold")
    checkpoints_336 = checkpoint_family("resolution-blend", "clean_dino_fold")
    localized = checkpoint_family("localized-dinov2", "localized_dino_fold")
    routed = checkpoint_family("routed-dinov2", "routed_dino_fold")
    _, localized_metrics = unique_json("localized-dinov2", "metrics.json")
    _, routed_manifest = unique_json("routed-dinov2", "weights_manifest.json")
    _, routed_metrics = unique_json("routed-dinov2", "metrics.json")
    if not routed_manifest.get("candidate_ready") or not routed_metrics.get("candidate_ready"):
        raise RuntimeError("Specialist OOF gates failed; refusing hidden inference")
    if routed_manifest.get("architecture") != "routed_specialist_transfer_v2":
        raise RuntimeError("Unexpected specialist architecture")
    if any(row["delta"] < -0.002 for row in routed_metrics["fold_guard"]):
        raise RuntimeError("Specialist scanner-fold gate failed")

    test_df = pd.read_csv(root / "test.csv")
    series_csv = pd.read_csv(root / "test_series.csv")
    test_series = p.list_series(root / "test_series", series_csv, set(test_df.StudyInstanceUID))
    public_slots = p.choose_slots(test_series)
    public_headers = p.study_headers(public_slots)
    public_sides = p.laterality_map(public_headers)
    studies, cache_336, cache_224, public_mask = p.build_cache(
        public_slots, public_sides, "hidden-public"
    )
    localized_studies, localized_cache, localized_mask = build_contiguous_cache(
        public_slots, public_sides, LOCALIZED_CENTRES, LOCALIZED_SIZE, p.SLOTS,
        "hidden-localized",
    )
    if studies != localized_studies:
        raise RuntimeError("Public cache study ordering mismatch")

    routed_series = annotate_series(test_series)
    routed_slots = choose_routed_slots(routed_series)
    routed_sides = robust_laterality(routed_series)
    routed_studies, routed_cache, routed_mask = build_contiguous_cache(
        routed_slots, routed_sides, ROUTED_CENTRES, ROUTED_SIZE, ROUTED_SLOTS,
        "hidden-routed",
    )
    if studies != routed_studies:
        raise RuntimeError("Routed cache study ordering mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Inference requires a GPU")
    indices = np.arange(len(studies))
    family_224, family_336, family_localized, family_routed = [], [], [], []
    for fold in range(5):
        model = p.load_baseline_model(dino_path, checkpoints_224[fold], fold, device)
        family_224.append(p.predict(model, cache_224, public_mask, indices, device))
        del model; gc.collect(); torch.cuda.empty_cache()

        checkpoint = torch.load(checkpoints_336[fold], map_location="cpu", weights_only=False)
        if checkpoint.get("img_size") != 336 or checkpoint.get("fold") != fold:
            raise ValueError("Invalid 336 checkpoint")
        model = p.build_model(dino_path)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(device).eval()
        family_336.append(p.predict(model, cache_336, public_mask, indices, device))
        del model, checkpoint; gc.collect(); torch.cuda.empty_cache()

        model = load_localized(
            dino_path, localized[fold], fold, "localized_target_patch_attention_v1", device
        )
        family_localized.append(
            predict_contiguous(model, localized_cache, localized_mask, device)
        )
        del model; gc.collect(); torch.cuda.empty_cache()

        model = load_localized(
            dino_path, routed[fold], fold, "routed_specialist_transfer_v2", device
        )
        family_routed.append(
            predict_contiguous(model, routed_cache, routed_mask, device, routed=True)
        )
        del model; gc.collect(); torch.cuda.empty_cache()
        p.log(f"completed inference fold {fold+1}/5")

    pred_224 = ranked_fold_ensemble(family_224)
    pred_336 = ranked_fold_ensemble(family_336)
    scored = SCORED_WEIGHTS[0] * pred_224 + SCORED_WEIGHTS[1] * pred_336
    pred_localized = ranked_fold_ensemble(family_localized)
    localized_weights = np.asarray([
        localized_metrics["localized_target_weights"][target] for target in p.TARGETS
    ], dtype=np.float32)
    prior = (1-localized_weights[None,:])*scored + localized_weights[None,:]*pred_localized
    pred_routed = ranked_fold_ensemble(family_routed)
    routed_weights = np.asarray([
        routed_manifest["routed_target_weights"][target] for target in p.TARGETS
    ], dtype=np.float32)
    final = (1-routed_weights[None,:])*prior + routed_weights[None,:]*pred_routed
    submission = p.write_submission(final, studies, test_df, "submission.csv")
    if submission.shape != (len(test_df), len(p.TARGETS)+1):
        raise RuntimeError("Submission shape validation failed")
    Path("inference_manifest.json").write_text(json.dumps({
        "scored_weights": SCORED_WEIGHTS,
        "localized_weights": dict(zip(p.TARGETS, localized_weights.tolist())),
        "routed_weights": dict(zip(p.TARGETS, routed_weights.tolist())),
        "guarded_blend_oof_auc": routed_metrics["guarded_blend_oof_auc"],
        "gold_training_count": 0,
        "elapsed_seconds": time.time()-p.T0,
    }, indent=2))


if __name__ == "__main__":
    main()
