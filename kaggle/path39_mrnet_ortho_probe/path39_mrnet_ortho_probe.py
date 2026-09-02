#!/usr/bin/env python3
"""Path39: cheap external-label probe before any full competition retraining.

Train a physically ordered study head on MRNet's real ACL/meniscus diagnoses
using frozen OrthoFoundation slice features, then evaluate once on the 58 RSNA
gold studies.  This kernel never creates submission.csv and has a hard 75-minute
wall; it exists only to decide whether supervised external knee MRI transfers.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260902
IMAGE_SIZE = 224
FEATURE_DIM = 2048
HEAD_DIM = 256
SLICES_PER_PLANE = 8
FEATURE_BATCH_PER_GPU = 32
HEAD_BATCH_SIZE = 64
HEAD_EPOCHS = 40
HEAD_PATIENCE = 6
WALL_SECONDS = 75 * 60
ORTHO_SHA256 = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"
PLANES = ("axial", "coronal", "sagittal")
MRNET_TARGETS = ("abnormal", "acl", "meniscus")
RSNA_TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)
SLOT_BOUNDS = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
PATH17_REFERENCE = {
    "ACL": 0.9276960784313725,
    "Medial Meniscus": 0.891826923076923,
    "Lateral Meniscus": 0.7900621118012422,
}

STARTED = time.monotonic()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def elapsed() -> float:
    return time.monotonic() - STARTED


def check_wall(stage: str) -> None:
    if elapsed() >= WALL_SECONDS:
        raise TimeoutError(f"Path39 75-minute wall reached during {stage}")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def find_file(name: str) -> Path:
    matches = []
    for current, directories, files in os.walk("/kaggle/input"):
        if Path(current) == Path("/kaggle/input"):
            directories[:] = [d for d in directories if d != "competitions"]
        else:
            directories[:] = [d for d in directories if d not in (
                "train_series", "test_series", "train_images", "test_images"
            )]
        if name in files:
            matches.append(Path(current) / name)
    matches.sort()
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def find_competition_root() -> Path:
    for root in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (root / "train.csv").is_file():
            return root
    raise FileNotFoundError("Competition train.csv not found")


class TwoPartArray:
    def __init__(self, first: np.ndarray, second: np.ndarray):
        self.first, self.second, self.cut = first, second, len(first)
        self.shape = (len(first) + len(second),) + first.shape[1:]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.first[index] if index < self.cut else self.second[index - self.cut]


def load_corpus() -> tuple[TwoPartArray, np.ndarray]:
    first = np.load(find_file("all_vols.npy"), mmap_mode="r")
    second = np.load(find_file("extra_vols.npy"), mmap_mode="r")
    ids = np.concatenate([
        np.load(find_file("all_ids.npy"), allow_pickle=True).astype(str),
        np.load(find_file("extra_ids.npy"), allow_pickle=True).astype(str),
    ])
    volumes = TwoPartArray(first, second)
    if volumes.shape != (4407, 44, 336, 336) or len(set(ids)) != 4407:
        raise ValueError(f"Unexpected physical corpus contract: {volumes.shape}, ids={len(set(ids))}")
    return volumes, ids


def read_study_side(raw_root: Path, study: str) -> str | None:
    import pydicom

    files = sorted((raw_root / "train_series" / study).glob("*/*.dcm"))
    if not files:
        return None
    tags = [
        "Laterality", "ImageLaterality", "ImagePositionPatient",
        "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns",
    ]
    try:
        ds = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True,
                             force=True, specific_tags=tags)
    except Exception:
        return None
    for tag in ("Laterality", "ImageLaterality"):
        value = str(getattr(ds, tag, "")).strip().upper()[:1]
        if value in ("L", "R"):
            return value
    try:
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        spacing = np.asarray(ds.PixelSpacing, dtype=float)
        centre = (ipp + iop[:3] * spacing[1] * float(ds.Columns) / 2
                  + iop[3:] * spacing[0] * float(ds.Rows) / 2)
        return None if abs(centre[0]) < 20 else ("R" if centre[0] < 0 else "L")
    except Exception:
        return None


def canonicalize_right(volume: np.ndarray) -> np.ndarray:
    output = volume.copy()
    for start, stop in SLOT_BOUNDS[:2]:
        output[start:stop] = output[start:stop][::-1]
    output[22:44] = output[22:44, :, ::-1]
    return output


class OrthoEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=images.is_cuda):
            output = self.backbone.forward_features(images)
        return torch.cat([
            output["x_norm_clstoken"], output["x_norm_patchtokens"].amax(1)
        ], dim=1)


def load_encoder(device: torch.device) -> tuple[nn.Module, dict]:
    weights_path = find_file("OrthoFoudation-L.pth")
    digest = sha256(weights_path)
    if digest != ORTHO_SHA256:
        raise RuntimeError(f"OrthoFoundation checkpoint hash drift: {digest}")
    source_files = list(weights_path.parent.glob("**/dinov3/hub/backbones.py"))
    if len(source_files) != 1:
        raise FileNotFoundError(f"Expected one mounted DINOv3 source tree, found {source_files}")
    sys.path.insert(0, str(source_files[0].parents[2]))
    from dinov3.hub.backbones import dinov3_vitl16

    backbone = dinov3_vitl16(pretrained=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in state.items()}, strict=True)
    encoder: nn.Module = OrthoEncoder(backbone).eval().to(device)
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
    return encoder, {
        "architecture": "OrthoFoundation DINOv3 ViT-L/16",
        "sha256": digest,
        "parameters": sum(p.numel() for p in backbone.parameters()),
        "trainable_parameters": 0,
    }


def uniform_indices(length: int, count: int = SLICES_PER_PLANE) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample an empty MRI plane")
    return np.rint(np.linspace(0, length - 1, count)).astype(int)


@torch.inference_mode()
def encode_slices(encoder: nn.Module, slices: np.ndarray, device: torch.device) -> np.ndarray:
    batch_size = FEATURE_BATCH_PER_GPU * max(torch.cuda.device_count(), 1)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    source = torch.from_numpy(np.asarray(slices, dtype=np.float32)).unsqueeze(1)
    outputs = []
    for start in range(0, len(source), batch_size):
        images = torch.nn.functional.interpolate(
            source[start:start + batch_size].to(device, non_blocking=True),
            size=(IMAGE_SIZE, IMAGE_SIZE), mode="area",
        ).repeat(1, 3, 1, 1).div_(255)
        images = images.sub_(mean).div_(std)
        outputs.append(encoder(images).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float16)


def read_mrnet_labels(path: Path) -> pd.Series:
    # This mirror encodes the original first row as `_0000,_0`; header=None is intentional.
    frame = pd.read_csv(path, header=None, names=("study", "label"), dtype=str)
    frame["study"] = frame.study.str.replace("_", "", regex=False).str.zfill(4)
    frame["label"] = pd.to_numeric(frame.label.str.replace("_", "", regex=False), errors="raise")
    if len(frame) != 1130 or frame.study.nunique() != 1130 or not frame.label.isin((0, 1)).all():
        raise ValueError(f"Bad MRNet label table {path}: rows={len(frame)}")
    return frame.set_index("study").label.astype(np.float32)


def discover_mrnet() -> tuple[dict[str, dict[str, Path]], np.ndarray, np.ndarray, dict]:
    label_paths = {target: find_file(f"train-{target}.csv") for target in MRNET_TARGETS}
    roots = {path.parent for path in label_paths.values()}
    if len(roots) != 1:
        raise RuntimeError(f"MRNet label files span multiple roots: {roots}")
    root = roots.pop()
    labels = {target: read_mrnet_labels(path) for target, path in label_paths.items()}
    studies = sorted(set.intersection(*(set(series.index) for series in labels.values())))
    if len(studies) != 1130:
        raise ValueError(f"MRNet label intersection drift: {len(studies)}")

    manifest_path = root / "train.csv"
    manifest = pd.read_csv(manifest_path)
    required = {"file_name", "md5_hash"}
    if len(manifest) != 3390 or not required.issubset(manifest.columns):
        raise ValueError(f"MRNet manifest drift: rows={len(manifest)}, cols={manifest.columns.tolist()}")
    digest_to_plane = {
        str(row.md5_hash): str(row.file_name).split("/")[0].lower()
        for row in manifest.itertuples(index=False)
    }
    arrays = sorted(root.rglob("*.npy"))
    paths: dict[str, dict[str, Path]] = {study: {} for study in studies}
    matched = 0
    for index, path in enumerate(arrays):
        check_wall("MRNet layout verification")
        plane = digest_to_plane.get(md5_b64(path))
        if plane is None:
            continue
        study = path.stem.zfill(4)
        if study in paths and plane in PLANES:
            if plane in paths[study]:
                raise ValueError(f"Duplicate MRNet {study}/{plane}")
            paths[study][plane] = path
            matched += 1
        if (index + 1) % 500 == 0:
            log(f"verified MRNet layout {index + 1}/{len(arrays)}")
    bad = {study: sorted(value) for study, value in paths.items() if set(value) != set(PLANES)}
    if bad or matched != 3390:
        raise ValueError(
            f"MRNet manifest mapping failed: matched={matched}, incomplete={len(bad)}, "
            f"examples={list(bad.items())[:3]}"
        )
    y = np.column_stack([[labels[target].loc[study] for study in studies]
                         for target in MRNET_TARGETS]).astype(np.float32)
    return paths, np.asarray(studies), y, {
        "root": str(root),
        "studies": len(studies),
        "mounted_arrays": len(arrays),
        "manifest_volumes": matched,
        "manifest_sha256": sha256(manifest_path),
        "positive_counts": dict(zip(MRNET_TARGETS, y.sum(axis=0).astype(int).tolist())),
    }


def extract_mrnet_features(paths: dict[str, dict[str, Path]], studies: np.ndarray,
                           encoder: nn.Module, device: torch.device) -> np.ndarray:
    features = np.empty((len(studies), len(PLANES), SLICES_PER_PLANE, FEATURE_DIM), dtype=np.float16)
    started = time.monotonic()
    for row, study in enumerate(studies):
        check_wall("MRNet feature extraction")
        chosen = []
        for plane in PLANES:
            volume = np.load(paths[str(study)][plane], mmap_mode="r")
            if volume.ndim != 3:
                raise ValueError(f"Bad MRNet volume {paths[str(study)][plane]}: {volume.shape}")
            chosen.append(np.asarray(volume[uniform_indices(len(volume))]))
        features[row] = encode_slices(encoder, np.concatenate(chosen), device).reshape(
            len(PLANES), SLICES_PER_PLANE, FEATURE_DIM
        )
        if (row + 1) % 100 == 0:
            rate = (row + 1) * len(PLANES) * SLICES_PER_PLANE / (time.monotonic() - started)
            log(f"MRNet features {row + 1}/{len(studies)}: {rate:.1f} slices/s")
    return features


class OrderedStudyHead(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = HEAD_DIM,
                 targets: int = len(MRNET_TARGETS)):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU())
        self.plane_embedding = nn.Parameter(torch.randn(len(PLANES), hidden) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(SLICES_PER_PLANE, hidden) * 0.02)
        self.depthwise = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.attention = nn.Linear(hidden, targets)
        self.classifier = nn.Parameter(torch.randn(targets, hidden * 2) * 0.02)
        self.bias = nn.Parameter(torch.zeros(targets))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.projection(features)
        x = x + self.plane_embedding[None, :, None, :] + self.position_embedding[None, None, :, :]
        local = []
        for plane in range(len(PLANES)):
            sequence = x[:, plane].transpose(1, 2)
            local.append((sequence + self.pointwise(torch.nn.functional.gelu(
                self.depthwise(sequence)
            ))).transpose(1, 2))
        x = torch.stack(local, dim=1).flatten(1, 2)
        attention = self.attention(x).softmax(dim=1)
        attended = torch.einsum("bst,bsf->btf", attention, x)
        maximum = x.amax(dim=1).unsqueeze(1).expand(-1, self.classifier.shape[0], -1)
        pooled = torch.cat((attended, maximum), dim=2)
        return (pooled * self.classifier).sum(dim=2) + self.bias


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(x), HEAD_BATCH_SIZE):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output.append(model(x[start:start + HEAD_BATCH_SIZE].to(device)).sigmoid().float().cpu())
    return torch.cat(output).numpy()


def aucs(y: np.ndarray, p: np.ndarray, names: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    report = {name: float(roc_auc_score(y[:, i], p[:, i])) for i, name in enumerate(names)}
    return float(np.mean(list(report.values()))), report


def train_head(features: np.ndarray, labels: np.ndarray, device: torch.device) -> tuple[nn.Module, dict]:
    strata = labels[:, 0].astype(int) * 4 + labels[:, 1].astype(int) * 2 + labels[:, 2].astype(int)
    indices = np.arange(len(labels))
    train_rows, val_rows = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=strata
    )
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels)
    loader = DataLoader(TensorDataset(x[train_rows], y[train_rows]), batch_size=HEAD_BATCH_SIZE,
                        shuffle=True, num_workers=0, drop_last=False)
    model = OrderedStudyHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    positive = labels[train_rows].sum(axis=0)
    negative = len(train_rows) - positive
    pos_weight = torch.from_numpy(np.clip(negative / np.maximum(positive, 1), 1, 8)).float().to(device)
    best_auc, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(HEAD_EPOCHS):
        check_wall("MRNet head training")
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch_x.to(device))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), batch_y.to(device), pos_weight=pos_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        val_predictions = predict(model, x[val_rows], device)
        val_macro, val_per_target = aucs(labels[val_rows], val_predictions, MRNET_TARGETS)
        history.append({"epoch": epoch + 1, "loss": total / len(loader),
                        "val_macro_auc": val_macro, "val_per_target_auc": val_per_target})
        log(f"head epoch {epoch + 1}: loss={total / len(loader):.5f} val={val_macro:.5f}")
        if val_macro > best_auc + 1e-4:
            best_auc = val_macro
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 5 and stale >= HEAD_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("MRNet head did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_predictions = predict(model, x[val_rows], device)
    val_macro, val_per_target = aucs(labels[val_rows], val_predictions, MRNET_TARGETS)
    return model, {
        "split": "fixed 80/20 stratified by the three-label bit pattern",
        "train_studies": len(train_rows), "held_studies": len(val_rows),
        "macro_auc": val_macro, "per_target_auc": val_per_target,
        "history": history,
    }


def extract_rsna_gold(encoder: nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    raw_root = find_competition_root()
    train = pd.read_csv(raw_root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train.loc[train[list(RSNA_TARGETS)].notna().all(axis=1),
                     ["StudyInstanceUID", *RSNA_TARGETS]].copy()
    if len(train) != 4407 or len(gold) != 58:
        raise ValueError(f"Competition cohort drift: train={len(train)}, gold={len(gold)}")
    volumes, ids = load_corpus()
    row_of = {uid: row for row, uid in enumerate(ids)}
    if not set(gold.StudyInstanceUID).issubset(row_of):
        raise ValueError("Gold studies missing from physical corpus")
    output = np.empty((len(gold), len(PLANES), SLICES_PER_PLANE, FEATURE_DIM), dtype=np.float16)
    side_counts = {"L": 0, "R": 0, "unknown": 0}
    for gold_row, uid in enumerate(gold.StudyInstanceUID):
        check_wall("RSNA gold feature extraction")
        volume = np.asarray(volumes[row_of[uid]])
        side = read_study_side(raw_root, uid)
        side_counts[side or "unknown"] += 1
        if side == "R":
            volume = canonicalize_right(volume)
        present = volume.reshape(44, -1).any(axis=1)
        groups = ((0, 22), (22, 36), (36, 44))  # sagittal, coronal, axial in corpus order
        selected = {}
        for name, (start, stop) in zip(("sagittal", "coronal", "axial"), groups):
            rows = np.flatnonzero(present[start:stop]) + start
            if not len(rows):
                raise ValueError(f"Gold {uid} has no {name} slices")
            selected[name] = volume[rows[uniform_indices(len(rows))]]
        ordered = np.concatenate([selected[plane] for plane in PLANES])
        output[gold_row] = encode_slices(encoder, ordered, device).reshape(
            len(PLANES), SLICES_PER_PLANE, FEATURE_DIM
        )
    truth = gold[list(RSNA_TARGETS)].to_numpy(np.float32)
    return output, truth, gold, {"studies": len(gold), "laterality": side_counts}


def self_test() -> None:
    assert np.array_equal(uniform_indices(3), np.array([0, 0, 1, 1, 1, 1, 2, 2]))
    sample = np.arange(44 * 3 * 4).reshape(44, 3, 4)
    flipped = canonicalize_right(sample)
    assert np.array_equal(flipped[0], sample[11])
    assert np.array_equal(flipped[22, :, 0], sample[22, :, -1])
    model = OrderedStudyHead(input_dim=32, hidden=16)
    logits = model(torch.randn(2, 3, 8, 32))
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    logits.sum().backward()
    print("PATH39_SELF_TEST_OK")


def main() -> None:
    seed_everything()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Path39 probe requires a GPU")
    paths, studies, labels, mrnet_audit = discover_mrnet()
    log(f"MRNet ready: {mrnet_audit}")
    encoder, encoder_audit = load_encoder(device)
    features = extract_mrnet_features(paths, studies, encoder, device)
    head, heldout = train_head(features, labels, device)
    log(f"MRNet held-out macro={heldout['macro_auc']:.5f}")
    gold_features, gold_truth, gold, gold_audit = extract_rsna_gold(encoder, device)
    gold_predictions = predict(head, torch.from_numpy(gold_features.astype(np.float32)), device)

    task_index = {name: i for i, name in enumerate(MRNET_TARGETS)}
    target_index = {name: i for i, name in enumerate(RSNA_TARGETS)}
    mapped_scores = {
        "ACL": gold_predictions[:, task_index["acl"]],
        "Medial Meniscus": gold_predictions[:, task_index["meniscus"]],
        "Lateral Meniscus": gold_predictions[:, task_index["meniscus"]],
    }
    mapped_auc = {
        target: float(roc_auc_score(gold_truth[:, target_index[target]], scores))
        for target, scores in mapped_scores.items()
    }
    mapped_macro = float(np.mean(list(mapped_auc.values())))
    abnormal_auc = {
        target: float(roc_auc_score(gold_truth[:, index], gold_predictions[:, task_index["abnormal"]]))
        for index, target in enumerate(RSNA_TARGETS)
    }
    deltas = {target: mapped_auc[target] - PATH17_REFERENCE[target] for target in mapped_auc}
    qualified = [target for target, delta in deltas.items() if delta >= 0.01]
    passed = bool(heldout["macro_auc"] >= 0.85 and mapped_macro >= 0.89 and qualified)

    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "architecture": "frozen OrthoFoundation plus physical ordered local-convolution MRNet head",
        "encoder": encoder_audit,
        "planes": PLANES,
        "slices_per_plane": SLICES_PER_PLANE,
        "targets": MRNET_TARGETS,
        "gold_usage": "none in training; 58 gold studies used once for transfer evaluation",
    }
    torch.save(checkpoint, "/kaggle/working/path39_mrnet_ortho_probe.pt")
    prediction_frame = pd.DataFrame({"StudyInstanceUID": gold.StudyInstanceUID})
    for target, scores in mapped_scores.items():
        prediction_frame[target] = scores
    prediction_frame["MRNet abnormal"] = gold_predictions[:, task_index["abnormal"]]
    prediction_frame.to_csv("/kaggle/working/path39_gold_predictions.csv", index=False)
    audit = {
        "status": "PATH39_MRNET_ORTHO_PROBE_COMPLETE",
        "elapsed_seconds": elapsed(),
        "wall_seconds": WALL_SECONDS,
        "hypothesis": "real external knee-MRI injury labels reveal transferable physical signal hidden by report-label noise",
        "independence": "MRNet diagnosis labels plus OrthoFoundation; no public RSNA predictions or saved submission",
        "mrnet": mrnet_audit,
        "encoder": encoder_audit,
        "physical_contract": {
            "mrnet": "three planes, eight uniformly ordered slices per plane",
            "rsna": "fixed physical corpus, right knees canonicalized from DICOM patient coordinates",
            "aggregation": "per-plane local 1D convolution before attention and max pooling",
        },
        "mrnet_heldout": heldout,
        "rsna_gold": {
            **gold_audit,
            "usage": "single untouched transfer evaluation; no gold row entered training",
            "mapped_macro_auc": mapped_macro,
            "mapped_per_target_auc": mapped_auc,
            "path17_reference_auc": PATH17_REFERENCE,
            "delta_vs_path17": deltas,
            "qualified_targets_plus_0.01": qualified,
            "abnormal_score_per_target_auc": abnormal_auc,
        },
        "candidate_gate": {
            "mrnet_heldout_macro_min": 0.85,
            "rsna_mapped_macro_min": 0.89,
            "requires_target_delta_vs_path17": 0.01,
            "passed_for_deeper_adaptation": passed,
            "submission_created": False,
        },
    }
    Path("/kaggle/working/path39_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    log(f"RSNA mapped macro={mapped_macro:.5f}; deltas={deltas}; qualified={qualified}")
    log(f"PATH39_MRNET_ORTHO_PROBE_COMPLETE passed={passed} elapsed={elapsed()/60:.1f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    args, _ = parser.parse_known_args()
    if args.self_test:
        self_test()
    else:
        try:
            main()
        except Exception as error:
            Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
            Path("/kaggle/working/path39_audit.json").write_text(json.dumps({
                "status": "PATH39_REJECTED_ERROR",
                "elapsed_seconds": elapsed(),
                "wall_seconds": WALL_SECONDS,
                "error_type": type(error).__name__,
                "error": str(error),
                "submission_created": False,
            }, indent=2) + "\n")
            raise
