#!/usr/bin/env python3
"""Path37: knee-domain foundation features with physical target specialists.

The frozen OrthoFoundation encoder supplies independent knee-MRI slice features.
A small target-routed all-slice head learns only on competition labels. Gold images
enter only fold-clean adaptation and the final full-gold deployment fit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset


SEED = 20260901
IMAGE_SIZE = 224
FEATURE_DIM = 2048
HEAD_DIM = 256
SCOUT_EPOCHS = 30
SCOUT_PATIENCE = 5
BATCH_SIZE = 128
FEATURE_BATCH_PER_GPU = 32
GOLD_REPEAT = 20
GOLD_FINETUNE_EPOCHS = 1
ORTHO_SHA256 = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
# The published CC0 corpus is a 140 mm in-plane crop with these five fixed slots.
SLOT_LENGTHS = (12, 10, 8, 6, 8)
SLOT_BOUNDS = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
SLOT_IDS = torch.tensor(sum(([slot] * count for slot, count in enumerate(SLOT_LENGTHS)), []))
# Sagittal-FS, sagittal-nonFS, coronal-FS, coronal-nonFS, axial.
ROUTES = torch.tensor([
    [1, 1, 1, 0, 0],  # ACL
    [0, 0, 1, 1, 1],  # MCL
    [1, 1, 1, 1, 0],  # medial meniscus
    [1, 1, 1, 1, 0],  # lateral meniscus
    [1, 1, 1, 1, 0],  # medial OA
    [1, 1, 1, 1, 0],  # lateral OA
    [1, 0, 1, 0, 1],  # PF OA
    [1, 0, 1, 0, 1],  # effusion
    [1, 0, 1, 0, 1],  # synovitis
    [1, 0, 1, 0, 1],  # Baker's cyst
    [1, 0, 1, 0, 1],  # contusion
    [1, 0, 1, 0, 1],  # fracture
], dtype=torch.bool)

GOLD_FOLDS_SHA256 = "b2a29a83558dbad9ff490ac43731e4377ab297523ed8a2e66ede02050f0a7a12"
GOLD_FOLDS_CSV = """StudyInstanceUID,gold_fold
1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149,3
1.2.826.0.1.3680043.8.498.10170898615867673028696505248839028269,2
1.2.826.0.1.3680043.8.498.10306159113324811538703788080836752052,1
1.2.826.0.1.3680043.8.498.11287937729196958426538087439102017580,0
1.2.826.0.1.3680043.8.498.11382021393803389951964005983002209238,4
1.2.826.0.1.3680043.8.498.11548045715264151632153040089882701935,4
1.2.826.0.1.3680043.8.498.11557620559191469069130827959098335840,1
1.2.826.0.1.3680043.8.498.11771393824519892797114773408583976756,0
1.2.826.0.1.3680043.8.498.11851412923016044948101698015974810604,0
1.2.826.0.1.3680043.8.498.11915937982684988073644209606907169581,3
1.2.826.0.1.3680043.8.498.12448079646359892252441208258836556945,4
1.2.826.0.1.3680043.8.498.12505035424093604269515328931488770819,2
1.2.826.0.1.3680043.8.498.12606657226568558340797193167488111973,3
1.2.826.0.1.3680043.8.498.12801308844398614687904447633432197492,2
1.2.826.0.1.3680043.8.498.12978510157202852202776899910529174803,1
1.2.826.0.1.3680043.8.498.13267780356245120052411517053322874891,4
1.2.826.0.1.3680043.8.498.13335129881737731410081002627729200903,0
1.2.826.0.1.3680043.8.498.15593638897292057356864060466120253309,4
1.2.826.0.1.3680043.8.498.16060119389060497136231217718921482192,1
1.2.826.0.1.3680043.8.498.17844546765907321649997094867791102711,3
1.2.826.0.1.3680043.8.498.18392509497170616983977319528036573378,3
1.2.826.0.1.3680043.8.498.22109739962224309418874538994436903404,3
1.2.826.0.1.3680043.8.498.25695966186129395049687747156570466645,4
1.2.826.0.1.3680043.8.498.26790702379506190936834447203448882465,1
1.2.826.0.1.3680043.8.498.27437263843446879932281897446100134924,2
1.2.826.0.1.3680043.8.498.28925345859498351203477642741908452608,2
1.2.826.0.1.3680043.8.498.29764868091238287072166823522853419550,0
1.2.826.0.1.3680043.8.498.30246079718471552972130572444383079911,1
1.2.826.0.1.3680043.8.498.32321830776739689645700555055955725945,2
1.2.826.0.1.3680043.8.498.37040910196459054543310335064167212742,0
1.2.826.0.1.3680043.8.498.37392127307867238524251864751713143683,4
1.2.826.0.1.3680043.8.498.39500553372517290823815606829518305500,0
1.2.826.0.1.3680043.8.498.41600498783384864111179175518614837125,3
1.2.826.0.1.3680043.8.498.47921753480592595198052407850568677187,4
1.2.826.0.1.3680043.8.498.48348615349418615469875801439348424274,1
1.2.826.0.1.3680043.8.498.48946580946665031852355005294734101132,4
1.2.826.0.1.3680043.8.498.54977581323931277817074741182670080450,3
1.2.826.0.1.3680043.8.498.56512564464301544960446419143135447363,3
1.2.826.0.1.3680043.8.498.59483999067816153759785789654766826710,2
1.2.826.0.1.3680043.8.498.62465595376489211274225312453216559395,1
1.2.826.0.1.3680043.8.498.62549354677638403845904556149868367236,4
1.2.826.0.1.3680043.8.498.64408609256163127278435484217683272910,0
1.2.826.0.1.3680043.8.498.64703506772167798469048460791472465039,3
1.2.826.0.1.3680043.8.498.65708905118633339771181857781063784327,2
1.2.826.0.1.3680043.8.498.67188121544063723837669983597453941774,4
1.2.826.0.1.3680043.8.498.69392348385274125385290015404144639002,1
1.2.826.0.1.3680043.8.498.72853333220043794904856138561095171921,0
1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220,1
1.2.826.0.1.3680043.8.498.73926443729786165628848843707532839995,3
1.2.826.0.1.3680043.8.498.75187434248356774277526985329346125190,4
1.2.826.0.1.3680043.8.498.77362718298550276679350855963451855003,2
1.2.826.0.1.3680043.8.498.78512215519177850923279235346968676828,0
1.2.826.0.1.3680043.8.498.82166943552764439138333504456139890254,1
1.2.826.0.1.3680043.8.498.86968600239724310678905311244945464037,2
1.2.826.0.1.3680043.8.498.88077418639301174409926781329613570435,0
1.2.826.0.1.3680043.8.498.90283565381042081768587894596970552767,1
1.2.826.0.1.3680043.8.498.94433753471890306108089557761231739312,1
1.2.826.0.1.3680043.8.498.97274720257634584071500649275217521662,3
"""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_file(name: str) -> Path:
    root = Path("/kaggle/input")
    matches = []
    for current, directories, files in os.walk(root):
        if Path(current) == root:
            directories[:] = [directory for directory in directories
                              if directory not in ("competitions", "rsna-knee-abnormality-detection")]
        else:
            directories[:] = [directory for directory in directories
                              if directory not in ("train_series", "test_series", "train_images", "test_images")]
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


def load_corpus() -> tuple[TwoPartArray, np.ndarray, np.ndarray]:
    first = np.load(find_file("all_vols.npy"), mmap_mode="r")
    second = np.load(find_file("extra_vols.npy"), mmap_mode="r")
    ids = np.concatenate([
        np.load(find_file("all_ids.npy"), allow_pickle=True).astype(str),
        np.load(find_file("extra_ids.npy"), allow_pickle=True).astype(str),
    ])
    masks = np.concatenate([
        np.load(find_file("all_masks.npy")), np.load(find_file("extra_masks.npy"))
    ]).astype(np.uint8)
    volumes = TwoPartArray(first, second)
    if volumes.shape != (4407, 44, 336, 336) or masks.shape != (4407, 44):
        raise ValueError(f"Unexpected corpus contract: {volumes.shape}, {masks.shape}")
    if len(set(ids)) != 4407:
        raise ValueError("Corpus IDs are not unique and complete")
    return volumes, masks, ids


def read_study_metadata(raw_root: Path, study: str) -> tuple[str | None, str]:
    import pydicom

    files = sorted((raw_root / "train_series" / study).glob("*/*.dcm"))
    if not files:
        return None, "unknown|unknown|unknown"
    tags = [
        "Laterality", "ImageLaterality", "ImagePositionPatient", "ImageOrientationPatient",
        "PixelSpacing", "Rows", "Columns", "Manufacturer", "ManufacturerModelName",
        "MagneticFieldStrength",
    ]
    try:
        ds = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True,
                             force=True, specific_tags=tags)
    except Exception:
        return None, "unknown|unknown|unknown"
    explicit = next((str(getattr(ds, tag, "")).strip().upper()[:1]
                     for tag in ("Laterality", "ImageLaterality")
                     if str(getattr(ds, tag, "")).strip().upper()[:1] in ("L", "R")), None)
    side = explicit
    if side is None:
        try:
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            spacing = np.asarray(ds.PixelSpacing, dtype=float)
            centre = (ipp + iop[:3] * spacing[1] * float(ds.Columns) / 2
                      + iop[3:] * spacing[0] * float(ds.Rows) / 2)
            side = None if abs(centre[0]) < 20 else ("R" if centre[0] < 0 else "L")
        except Exception:
            side = None
    scanner = "|".join(str(getattr(ds, key, "unknown") or "unknown").strip().lower()
                       for key in ("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"))
    return side, scanner


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
        # DataParallel autocast is thread-local, so each replica enters it here.
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=images.is_cuda):
            output = self.backbone.forward_features(images)
        return torch.cat([output["x_norm_clstoken"], output["x_norm_patchtokens"].amax(1)], dim=1)


def load_ortho_encoder(device: torch.device) -> tuple[nn.Module, dict]:
    weights_path = find_file("OrthoFoudation-L.pth")
    digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ORTHO_SHA256:
        raise RuntimeError(f"OrthoFoundation checkpoint hash drift: {digest.hexdigest()}")

    source_files = list(weights_path.parent.glob("**/dinov3/hub/backbones.py"))
    if len(source_files) != 1:
        raise FileNotFoundError(f"Expected one mounted DINOv3 source tree, found {source_files}")
    source_root = source_files[0].parents[2]
    sys.path.insert(0, str(source_root))
    from dinov3.hub.backbones import dinov3_vitl16

    backbone = dinov3_vitl16(pretrained=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("backbone."): value for key, value in state.items()}
    backbone.load_state_dict(state, strict=True)
    del state
    encoder: nn.Module = OrthoEncoder(backbone).eval().to(device)
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
    info = {
        "architecture": "OrthoFoundation DINOv3 ViT-L/16",
        "sha256": ORTHO_SHA256,
        "parameters": sum(parameter.numel() for parameter in backbone.parameters()),
        "feature": "normalized CLS concatenated with channelwise max over normalized patch tokens",
    }
    return encoder, info


@torch.inference_mode()
def build_feature_cache(volumes: TwoPartArray, masks: np.ndarray, ids: np.ndarray,
                        raw_root: Path, device: torch.device) -> tuple[Path, list[str], dict]:
    final = Path("/kaggle/working/path37_orthofoundation_features.npy")
    tmp = final.with_suffix(".tmp.npy")
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                      shape=(len(ids), 44, FEATURE_DIM))
    encoder, encoder_info = load_ortho_encoder(device)
    batch_size = FEATURE_BATCH_PER_GPU * max(torch.cuda.device_count(), 1)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    scanners, counts, started, encoded = [], Counter(), time.time(), 0
    for index, study in enumerate(ids):
        side, scanner = read_study_metadata(raw_root, study)
        scanners.append(scanner)
        counts[side or "unknown"] += 1
        volume = np.asarray(volumes[index])
        if side == "R":
            volume = canonicalize_right(volume)
        present = volume.reshape(44, -1).any(axis=1)
        masks[index] = present
        cache[index] = 0
        slice_indices = np.flatnonzero(present)
        slices = torch.from_numpy(np.array(volume[slice_indices], copy=True)).float().unsqueeze(1)
        for start in range(0, len(slices), batch_size):
            images = torch.nn.functional.interpolate(
                slices[start:start + batch_size].to(device, non_blocking=True),
                size=(IMAGE_SIZE, IMAGE_SIZE), mode="area",
            ).repeat(1, 3, 1, 1).div_(255)
            images = images.sub_(mean).div_(std)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                features = encoder(images)
            rows = slice_indices[start:start + len(features)]
            cache[index, rows] = features.float().cpu().numpy().astype(np.float16)
            encoded += len(features)
        if (index + 1) % 100 == 0:
            elapsed = time.time() - started
            log(f"features {index + 1}/{len(ids)}: {encoded / elapsed:.1f} slices/s")
    cache.flush()
    del cache, encoder
    torch.cuda.empty_cache()
    os.replace(tmp, final)
    log(f"feature cache complete: {final.stat().st_size / 1024**3:.2f} GiB; laterality={dict(counts)}")
    return final, scanners, encoder_info


def load_label_sources(ids: np.ndarray) -> dict[str, pd.DataFrame]:
    files = {
        "steven": "llm_labels_full.csv",
        "pilkwang": "report_labels_v2.csv",
        "lixin": "labels_llm_gpt56sol.csv",
    }
    sources = {}
    for name, filename in files.items():
        frame = pd.read_csv(find_file(filename), dtype={"StudyInstanceUID": str})
        if not set(TARGETS).issubset(frame.columns):
            raise ValueError(f"{filename} has the wrong label schema")
        frame = frame.set_index("StudyInstanceUID")[TARGETS].apply(pd.to_numeric, errors="coerce")
        sources[name] = frame.reindex(ids)
    # Pilkwang lacks one study; a deterministic Steven fallback preserves full coverage.
    for name in ("pilkwang", "lixin"):
        sources[name] = sources[name].fillna(sources["steven"])
    sources["steven_synovitis"] = sources["steven"].copy()
    undecided = sources["steven_synovitis"]["Synovitis"].eq(0.5)
    sources["steven_synovitis"].loc[undecided, "Synovitis"] = (
        0.25 + 0.5 * sources["steven_synovitis"].loc[undecided, "Effusion"]
    )
    if any(not np.isfinite(frame.to_numpy()).all() for frame in sources.values()):
        raise ValueError("Non-finite report labels remain after alignment")
    return sources


LABEL_CANDIDATES = {
    "steven": {"steven": 1.0},
    "pilkwang": {"pilkwang": 1.0},
    "lixin": {"lixin": 1.0},
    "steven_synovitis": {"steven_synovitis": 1.0},
    "steven_pilkwang": {"steven": 0.5, "pilkwang": 0.5},
    "steven_lixin": {"steven": 0.5, "lixin": 0.5},
    "three_way": {"steven": 1 / 3, "pilkwang": 1 / 3, "lixin": 1 / 3},
}


def candidate_values(sources: dict[str, pd.DataFrame], candidate: str, target: str) -> pd.Series:
    return sum(weight * sources[source][target]
               for source, weight in LABEL_CANDIDATES[candidate].items())


def choose_label_policy(sources: dict[str, pd.DataFrame], gold: pd.DataFrame) -> tuple[dict, np.ndarray]:
    digest = hashlib.sha256(GOLD_FOLDS_CSV.encode()).hexdigest()
    if digest != GOLD_FOLDS_SHA256:
        raise RuntimeError(f"Embedded gold-fold hash drift: {digest}")
    folds = pd.read_csv(io.StringIO(GOLD_FOLDS_CSV), dtype={"StudyInstanceUID": str})
    table = gold.merge(folds, on="StudyInstanceUID", validate="one_to_one")
    if len(table) != 58 or set(table.gold_fold) != set(range(5)):
        raise ValueError("Immutable gold-fold coverage failed")
    uid_to_row = {uid: row for row, uid in enumerate(sources["steven"].index)}
    crossfit_predictions = np.zeros((58, len(TARGETS)), dtype=np.float32)
    deployment = {}
    fold_winners = {}
    per_target = {}
    for target_index, target in enumerate(TARGETS):
        target_fold_winners = {}
        for fold in range(5):
            train = table.gold_fold != fold
            held = table.gold_fold == fold
            scores = {}
            for name in LABEL_CANDIDATES:
                values = candidate_values(sources, name, target)
                scores[name] = roc_auc_score(
                    table.loc[train, target], [values.iloc[uid_to_row[uid]] for uid in table.loc[train, "StudyInstanceUID"]]
                )
            winner = max(scores, key=scores.get)
            # 58 studies cannot resolve tiny gains: require +0.02 over the strongest single key.
            if scores[winner] < scores["steven"] + 0.02:
                winner = "steven"
            target_fold_winners[str(fold)] = winner
            values = candidate_values(sources, winner, target)
            for row in np.flatnonzero(held.to_numpy()):
                crossfit_predictions[row, target_index] = values.iloc[uid_to_row[table.iloc[row].StudyInstanceUID]]
        full_scores = {
            name: roc_auc_score(table[target], [candidate_values(sources, name, target).iloc[uid_to_row[uid]]
                                                for uid in table.StudyInstanceUID])
            for name in LABEL_CANDIDATES
        }
        winner = max(full_scores, key=full_scores.get)
        if full_scores[winner] < full_scores["steven"] + 0.02:
            winner = "steven"
        deployment[target] = winner
        fold_winners[target] = target_fold_winners
        per_target[target] = float(roc_auc_score(table[target], crossfit_predictions[:, target_index]))
    audit = {
        "selection_evidence": "five-fold crossfit on immutable gold_folds.csv",
        "deployment_artifact": "policy refit on all 58 gold after the candidate grid and 0.02 margin were frozen",
        "gold_fold_sha256": GOLD_FOLDS_SHA256,
        "candidates": LABEL_CANDIDATES,
        "fold_winners": fold_winners,
        "deployment_winners": deployment,
        "crossfit_per_target_auc": per_target,
        "crossfit_macro_auc": float(np.mean(list(per_target.values()))),
    }
    labels = np.column_stack([
        candidate_values(sources, deployment[target], target).to_numpy(np.float32)
        for target in TARGETS
    ])
    return audit, labels


def materialize_policy(sources: dict[str, pd.DataFrame], policy: dict[str, str]) -> np.ndarray:
    return np.column_stack([
        candidate_values(sources, policy[target], target).to_numpy(np.float32)
        for target in TARGETS
    ])


class FeatureDataset(Dataset):
    def __init__(self, cache_path: Path, masks: np.ndarray, labels: np.ndarray,
                 weights: np.ndarray, indices: np.ndarray, augment: bool):
        self.cache_path, self.masks = cache_path, masks
        self.labels, self.weights = labels, weights
        self.indices, self.augment, self.cache = indices, augment, None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        if self.cache is None:
            self.cache = np.load(self.cache_path, mmap_mode="r")
        row = int(self.indices[item])
        x = torch.from_numpy(np.array(self.cache[row], copy=True)).float()
        return x, torch.from_numpy(self.masks[row]), torch.from_numpy(self.labels[row]), torch.from_numpy(self.weights[row])


class FoundationSpecialist(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, feature_dim: int = HEAD_DIM):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, feature_dim), nn.GELU(), nn.Dropout(0.1)
        )
        self.slot_embedding = nn.Parameter(torch.randn(5, feature_dim) * 0.02)
        self.attention = nn.Linear(feature_dim, len(TARGETS))
        self.classifier = nn.Parameter(torch.randn(len(TARGETS), feature_dim * 2) * 0.02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))
        self.register_buffer("slot_ids", SLOT_IDS.clone())
        self.register_buffer("routes", ROUTES.clone())

    def forward(self, features: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        features = self.projection(features) + self.slot_embedding[self.slot_ids]
        allowed = self.routes[:, self.slot_ids].T.unsqueeze(0)
        valid = present.bool().unsqueeze(2) & allowed
        if not valid.any(dim=1).all():
            raise RuntimeError("A target has no physically routed slice")
        attention = self.attention(features).masked_fill(~valid, -1e4).softmax(dim=1)
        attended = torch.einsum("bst,bsf->btf", attention, features)
        expanded = features.unsqueeze(2).expand(-1, -1, len(TARGETS), -1)
        maximum = expanded.masked_fill(~valid.unsqueeze(3), -1e4).amax(dim=1)
        pooled = torch.cat([attended, maximum], dim=2)
        return (pooled * self.classifier).sum(dim=2) + self.bias


def auc_report(truth: np.ndarray, predictions: np.ndarray) -> tuple[float, dict[str, float]]:
    report = {}
    for index, target in enumerate(TARGETS):
        binary = (truth[:, index] > 0.5).astype(int)
        if len(np.unique(binary)) == 2:
            report[target] = float(roc_auc_score(binary, predictions[:, index]))
    return float(np.mean(list(report.values()))), report


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outputs, truths = [], []
    for volume, present, truth, _ in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(volume.to(device, non_blocking=True), present.to(device, non_blocking=True))
        outputs.append(logits.sigmoid().float().cpu().numpy())
        truths.append(truth.numpy())
    return np.concatenate(outputs), np.concatenate(truths)


def fit_model(cache_path: Path, masks: np.ndarray, labels: np.ndarray, weights: np.ndarray,
              train_indices: np.ndarray, val_indices: np.ndarray | None, epochs: int,
              device: torch.device, seed: int, initial_state: dict | None = None,
              learning_rate: float = 2e-3) -> tuple[FoundationSpecialist, list[dict], int]:
    seed_everything(seed)
    workers = 2 if device.type == "cuda" else 0
    train_ds = FeatureDataset(cache_path, masks, labels, weights, train_indices, True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers,
                              pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                              drop_last=True)
    val_loader = None if val_indices is None else DataLoader(
        FeatureDataset(cache_path, masks, labels, weights, val_indices, False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    model = FoundationSpecialist().to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    positive = (weights[train_indices] * labels[train_indices]).sum(axis=0)
    negative = (weights[train_indices] * (1 - labels[train_indices])).sum(axis=0)
    pos_weight = torch.from_numpy(np.clip(negative / np.maximum(positive, 1e-6), 1, 8).astype(np.float32)).to(device)
    history, best_auc, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for features, present, truth, confidence in train_loader:
            features, present = features.to(device, non_blocking=True), present.to(device, non_blocking=True)
            truth, confidence = truth.to(device, non_blocking=True), confidence.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(features, present)
                cell_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), truth, reduction="none", pos_weight=pos_weight
                )
                loss = (cell_loss * confidence).sum() / confidence.sum().clamp_min(1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": total / len(train_loader)}
        if val_loader is not None:
            predictions, truth = predict(model, val_loader, device)
            val_auc, per_target = auc_report(truth, predictions)
            row.update({"scanner_val_macro_auc": val_auc, "scanner_val_per_target": per_target})
            if val_auc > best_auc + 1e-4:
                best_auc, best_epoch, stale = val_auc, epoch + 1, 0
            else:
                stale += 1
        history.append(row)
        log(f"epoch {epoch + 1}/{epochs}: {row}")
        if val_loader is not None and epoch >= 3 and stale >= SCOUT_PATIENCE:
            break
    return model, history, max(best_epoch, 4)


def self_test() -> None:
    assert tuple(SLOT_IDS.shape) == (44,) and ROUTES.shape == (12, 5)
    sample = np.arange(44 * 3 * 4, dtype=np.int32).reshape(44, 3, 4)
    flipped = canonicalize_right(sample)
    assert np.array_equal(flipped[0], sample[11])
    assert np.array_equal(flipped[22, :, 0], sample[22, :, -1])
    model = FoundationSpecialist(input_dim=32, feature_dim=16)
    features = torch.randn(2, 44, 32)
    present = torch.ones(2, 44, dtype=torch.uint8)
    present[1, 12:22] = 0
    logits = model(features, present)
    assert logits.shape == (2, 12) and torch.isfinite(logits).all()
    logits.sum().backward()
    print("PATH37_SELF_TEST_OK")


def main() -> None:
    seed_everything()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Path37 training requires a GPU")
    raw_root = find_competition_root()
    train = pd.read_csv(raw_root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train.loc[train[TARGETS].notna().all(axis=1), ["StudyInstanceUID"] + TARGETS].copy()
    if len(train) != 4407 or len(gold) != 58:
        raise ValueError(f"Competition cohort drift: train={len(train)}, gold={len(gold)}")
    volumes, masks, ids = load_corpus()
    if set(ids) != set(train.StudyInstanceUID):
        raise ValueError("Physical corpus and competition cohort differ")
    sources = load_label_sources(ids)
    label_audit, deployment_labels = choose_label_policy(sources, gold)
    log(f"label crossfit macro AUC={label_audit['crossfit_macro_auc']:.5f}; deployment={label_audit['deployment_winners']}")
    gold_map = gold.set_index("StudyInstanceUID")[TARGETS]
    gold_indices = np.asarray([i for i, uid in enumerate(ids) if uid in gold_map.index])
    silver_indices = np.asarray([i for i, uid in enumerate(ids) if uid not in gold_map.index])
    gold_truth = gold_map.loc[ids[gold_indices]].to_numpy(np.float32)
    base_labels = sources["steven"].to_numpy(np.float32)
    base_labels[gold_indices] = gold_truth
    base_weights = np.clip(2 * np.abs(base_labels - 0.5), 0, 1).astype(np.float32)
    base_weights[gold_indices] = 1.0
    cache_path, scanner_list, encoder_info = build_feature_cache(volumes, masks, ids, raw_root, device)
    scanners = np.asarray(scanner_list)
    split = next(GroupKFold(5).split(silver_indices, groups=scanners[silver_indices]))
    scout_train, scout_val = silver_indices[split[0]], silver_indices[split[1]]
    log(f"scanner scout: train={len(scout_train)}, val={len(scout_val)}, groups={len(set(scanners))}")
    scout, scout_history, deploy_epochs = fit_model(
        cache_path, masks, base_labels, base_weights, scout_train, scout_val,
        SCOUT_EPOCHS, device, SEED,
    )
    del scout
    torch.cuda.empty_cache()
    log(f"gold-free base: all {len(silver_indices)} silver studies for {deploy_epochs} epochs")
    base_model, deploy_history, _ = fit_model(
        cache_path, masks, base_labels, base_weights, silver_indices, None,
        deploy_epochs, device, SEED + 100,
    )
    base_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
    del base_model
    torch.cuda.empty_cache()

    gold_folds = pd.read_csv(io.StringIO(GOLD_FOLDS_CSV), dtype={"StudyInstanceUID": str})
    fold_map = dict(zip(gold_folds.StudyInstanceUID, gold_folds.gold_fold))
    gold_fold_of = np.asarray([fold_map.get(uid, -1) for uid in ids])
    if set(gold_fold_of[gold_indices]) != set(range(5)) or np.any(gold_fold_of[silver_indices] != -1):
        raise RuntimeError("Gold-fold assignment drift before model crossfit")
    crossfit_predictions = np.full((len(ids), len(TARGETS)), np.nan, dtype=np.float32)
    crossfit_histories = []
    gold_training_counts = {}
    for fold in range(5):
        fold_policy = {target: label_audit["fold_winners"][target][str(fold)] for target in TARGETS}
        fold_labels = materialize_policy(sources, fold_policy)
        fold_labels[gold_indices] = gold_truth
        fold_weights = np.clip(2 * np.abs(fold_labels - 0.5), 0, 1).astype(np.float32)
        fold_weights[gold_indices] = 1.0
        trained_gold = gold_indices[gold_fold_of[gold_indices] != fold]
        held_gold = gold_indices[gold_fold_of[gold_indices] == fold]
        tune_indices = np.concatenate([silver_indices, np.repeat(trained_gold, GOLD_REPEAT)])
        if np.intersect1d(tune_indices, held_gold).size:
            raise RuntimeError(f"Held gold entered fold {fold} adaptation")
        log(f"gold crossfit fold {fold}: train_gold={len(trained_gold)}, held_gold={len(held_gold)}")
        fold_model, fold_history, _ = fit_model(
            cache_path, masks, fold_labels, fold_weights, tune_indices, None,
            GOLD_FINETUNE_EPOCHS, device, SEED + 200 + fold,
            initial_state=base_state, learning_rate=2e-4,
        )
        held_loader = DataLoader(
            FeatureDataset(cache_path, masks, fold_labels, fold_weights, held_gold, False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
        )
        crossfit_predictions[held_gold] = predict(fold_model, held_loader, device)[0]
        crossfit_histories.append({"fold": fold, "policy": fold_policy, "history": fold_history})
        gold_training_counts[str(fold)] = int(len(trained_gold))
        del fold_model
        torch.cuda.empty_cache()
    if not np.isfinite(crossfit_predictions[gold_indices]).all():
        raise RuntimeError("Cross-fitted gold predictions are incomplete")
    gold_macro, gold_per_target = auc_report(gold_truth, crossfit_predictions[gold_indices])

    deployment_labels[gold_indices] = gold_truth
    deployment_weights = np.clip(2 * np.abs(deployment_labels - 0.5), 0, 1).astype(np.float32)
    deployment_weights[gold_indices] = 1.0
    deployment_indices = np.concatenate([silver_indices, np.repeat(gold_indices, GOLD_REPEAT)])
    log(f"full-gold deployment adaptation: gold={len(gold_indices)} x{GOLD_REPEAT}")
    model, deployment_adaptation, _ = fit_model(
        cache_path, masks, deployment_labels, deployment_weights, deployment_indices, None,
        GOLD_FINETUNE_EPOCHS, device, SEED + 300,
        initial_state=base_state, learning_rate=2e-4,
    )
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": "frozen OrthoFoundation ViT-L/16 plus physical target-routed all-slice head",
        "encoder": encoder_info,
        "targets": TARGETS,
        "slot_lengths": SLOT_LENGTHS,
        "image_size": IMAGE_SIZE,
        "crop_mm": 140.0,
        "gold_training_count": 58,
        "gold_usage": "crossfit evidence; full-gold deployment adaptation",
        "label_policy": label_audit["deployment_winners"],
        "deploy_epochs": deploy_epochs,
        "gold_repeat": GOLD_REPEAT,
    }
    torch.save(checkpoint, "/kaggle/working/path37_orthofoundation_specialist.pt")
    prediction_frame = pd.DataFrame(crossfit_predictions[gold_indices], columns=TARGETS)
    prediction_frame.insert(0, "StudyInstanceUID", ids[gold_indices])
    prediction_frame["gold_fold"] = gold_fold_of[gold_indices]
    prediction_frame.to_csv("/kaggle/working/path37_gold_predictions.csv", index=False)
    audit = {
        "status": "PATH37_ORTHOFOUNDATION_SPECIALIST_TRAIN_COMPLETE",
        "hypothesis": "knee-domain self-supervised features plus physical all-slice target specialists overcome the report-label ceiling",
        "independence": "no public competition prediction, Raptor weight, or Claude path used",
        "architecture_selection_note": (
            "OrthoFoundation was selected from primary web research before its gold result; all competition gold "
            "predictions remain fold-clean, while final deployment uses all 58 gold studies"
        ),
        "encoder": encoder_info,
        "physical_contract": {
            "source": "CC0 fixed-140mm five-slot corpus",
            "slice_processing": "individual 224px all-slice encoding; no cross-slot 2.5D windows",
            "laterality": "right knees canonicalized from DICOM patient coordinates/tags",
            "target_routing": "anatomy-specific allowed MRI sequence slots before attention-plus-max pooling",
            "slot_lengths": SLOT_LENGTHS,
            "image_size": IMAGE_SIZE,
        },
        "label_audit": label_audit,
        "scanner_grouped_scout": scout_history,
        "gold_free_base_history": deploy_history,
        "gold_crossfit_histories": crossfit_histories,
        "gold_crossfit_evaluation": {
            "usage": "each gold study predicted only by a model trained on the other immutable folds",
            "studies": len(gold_indices),
            "macro_auc": gold_macro,
            "per_target_auc": gold_per_target,
            "gold_training_count_per_fold": gold_training_counts,
        },
        "deployment_adaptation": deployment_adaptation,
        "gold_training_count": 58,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "candidate_gate": {
            "required_macro_auc": 0.89,
            "passed": bool(gold_macro >= 0.89),
            "submission_created": False,
        },
    }
    Path("/kaggle/working/path37_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    cache_path.unlink()
    log(f"Path37 crossfit gold macro AUC={gold_macro:.5f}; per-target={gold_per_target}")
    log("PATH37_ORTHOFOUNDATION_SPECIALIST_TRAIN_COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    args, _ = parser.parse_known_args()
    self_test() if args.self_test else main()
