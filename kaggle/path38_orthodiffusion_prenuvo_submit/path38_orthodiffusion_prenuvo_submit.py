#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PLANE_WEIGHTS = {
    0: "sagittal_model.pt",
    1: "coronal_model.pt",
    2: "axial_model.pt",
}


def log(message: str) -> None:
    print(f"[path38] {message}", flush=True)


def find_bundle() -> Path:
    hits = []
    for root, dirs, _files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in {"train_images", "test_images", "train_series", "test_series"}]
        path = Path(root)
        if (
            (path / "code" / "infer_lib.py").is_file()
            and (path / "heads" / "best_model_fold0.pth").is_file()
            and all((path / name).is_file() for name in PLANE_WEIGHTS.values())
        ):
            hits.append(path)
    if not hits:
        raise FileNotFoundError("OrthoDiffusion Prenuvo bundle not found under /kaggle/input")
    if len(hits) > 1:
        log(f"multiple bundles found; using shortest path: {[str(x) for x in hits]}")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def unwrap_state_dict(obj) -> dict:
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint is {type(obj).__name__}, expected dict/module")

    for key in ("state_dict", "model_state_dict", "model", "net", "head"):
        value = obj.get(key)
        if isinstance(value, torch.nn.Module):
            value = value.state_dict()
        if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
            obj = value
            break

    prefixes = ("module.", "model.", "net.")
    clean = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        clean[new_key] = value
    if not clean:
        raise ValueError("no tensor weights found in checkpoint")
    return clean


def load_head(infer_lib, path: Path, device: torch.device):
    head = infer_lib.PlaneFusionHead().to(device)
    checkpoint = torch.load(path, map_location=device)
    state = unwrap_state_dict(checkpoint)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{path.name} incompatible: missing={missing}, unexpected={unexpected}")
    head.eval()
    return head


def encode_volume(infer_lib, diffusion, vol_f16: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    vol = vol_f16.astype(np.float32, copy=False)
    x = infer_lib.to_ortho_input(vol).to(device)
    t = torch.tensor([int(cfg["t"])], device=device, dtype=torch.long)
    try:
        feat = diffusion.get_feature(x, t, name=cfg["block"], use_amp=(device.type == "cuda"))
    except TypeError:
        feat = diffusion.get_feature(x, t, name=cfg["block"])
    return feat.float().mean(dim=(2, 3, 4))[0].detach().cpu().numpy().astype(np.float32, copy=False)


def rank_predictions(preds: np.ndarray) -> np.ndarray:
    ranked = pd.DataFrame(preds, columns=TARGETS).rank(method="average", pct=True).to_numpy(np.float32)
    return np.clip(ranked, 1e-6, 1 - 1e-6)


def main() -> None:
    started = time.time()
    out_dir = Path("/kaggle/working")
    bundle = find_bundle()
    code_dir = bundle / "code"
    sys.path.insert(0, str(code_dir))
    import infer_lib  # noqa: PLC0415

    if list(infer_lib.LABEL_COLS) != TARGETS:
        raise RuntimeError(f"Path38 label contract drift: helper={infer_lib.LABEL_COLS}, kernel={TARGETS}")

    comp_root = infer_lib.find_comp_root()
    image_root = infer_lib.find_image_root(comp_root)
    test_df = pd.read_csv(comp_root / "test.csv")
    series_df = pd.read_csv(comp_root / "test_series.csv")
    study_ids = test_df["StudyInstanceUID"].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"bundle={bundle}")
    log(f"comp_root={comp_root}")
    log(f"image_root={image_root}")
    log(f"studies={len(study_ids)} device={device}")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for Path38 OrthoDiffusion inference")

    volume_cache = []
    plane_counts = {0: 0, 1: 0, 2: 0}
    for idx, study_id in enumerate(study_ids, 1):
        vols = infer_lib.pick_plane_vols(study_id, series_df, image_root)
        row = []
        for plane in range(3):
            vol = vols.get(plane)
            row.append(None if vol is None else vol.astype(np.float16, copy=False))
            if vol is not None:
                plane_counts[plane] += 1
        if not any(v is not None for v in row):
            raise RuntimeError(f"no usable series for study {study_id}")
        volume_cache.append(row)
        if idx == 1 or idx % 25 == 0 or idx == len(study_ids):
            log(f"cached selected plane volumes {idx}/{len(study_ids)}")

    features = np.zeros((len(study_ids), 3, infer_lib.FEAT_DIM), dtype=np.float32)
    masks = np.zeros((len(study_ids), 3), dtype=np.float32)
    for plane, weight_name in PLANE_WEIGHTS.items():
        if plane_counts[plane] == 0:
            continue
        log(f"loading plane={plane} weight={weight_name} count={plane_counts[plane]}")
        diffusion = infer_lib.load_diffusion(bundle / weight_name, device, code_dir)
        cfg = infer_lib.PLANE_CFG[plane]
        for idx, row in enumerate(volume_cache):
            vol = row[plane]
            if vol is None:
                continue
            features[idx, plane] = encode_volume(infer_lib, diffusion, vol, cfg, device)
            masks[idx, plane] = 1.0
            if (idx + 1) == 1 or (idx + 1) % 25 == 0 or (idx + 1) == len(study_ids):
                log(f"encoded plane={plane} {idx + 1}/{len(study_ids)}")
        del diffusion
        torch.cuda.empty_cache()
        gc.collect()

    del volume_cache
    gc.collect()
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite OrthoDiffusion features")
    if (masks.sum(axis=1) == 0).any():
        bad = [study_ids[i] for i in np.where(masks.sum(axis=1) == 0)[0][:5]]
        raise RuntimeError(f"studies without encoded planes: {bad}")

    fold_preds = []
    feat_t = torch.from_numpy(features).to(device)
    mask_t = torch.from_numpy(masks).to(device)
    for head_path in sorted((bundle / "heads").glob("best_model_fold*.pth")):
        log(f"loading head={head_path.name}")
        head = load_head(infer_lib, head_path, device)
        preds = []
        with torch.no_grad():
            for start in range(0, len(study_ids), 64):
                logits = head(feat_t[start : start + 64], mask_t[start : start + 64])
                preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        fold_preds.append(np.concatenate(preds, axis=0))
        del head
        torch.cuda.empty_cache()

    if not fold_preds:
        raise RuntimeError("no Path38 head checkpoints found")
    preds = rank_predictions(np.mean(fold_preds, axis=0))
    if not np.isfinite(preds).all():
        raise RuntimeError("non-finite Path38 predictions")

    sub = pd.DataFrame({"StudyInstanceUID": study_ids})
    for j, label in enumerate(TARGETS):
        sub[label] = preds[:, j]
    sub.to_csv(out_dir / "submission.csv", index=False)

    audit = {
        "path": "path38_orthodiffusion_prenuvo",
        "gold_usage": "none",
        "family": "OrthoDiffusion/Prenuvo direct offline inference",
        "bundle": str(bundle),
        "study_count": len(study_ids),
        "plane_counts": plane_counts,
        "heads": [p.name for p in sorted((bundle / "heads").glob("best_model_fold*.pth"))],
        "elapsed_sec": round(time.time() - started, 3),
        "submission_shape": list(sub.shape),
    }
    (out_dir / "path38_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log("wrote submission.csv and path38_audit.json")


def self_check() -> None:
    expected = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]
    assert TARGETS == expected, TARGETS
    ranked = rank_predictions(np.array([[0.1] * 12, [0.3] * 12, [0.2] * 12], dtype=np.float32))
    assert ranked.shape == (3, 12)
    assert np.isfinite(ranked).all()
    fake = {
        "state_dict": {
            "module.proj.weight": torch.zeros(256, 256),
            "module.proj.bias": torch.zeros(256),
            "module.head.0.weight": torch.zeros(512, 256),
            "module.head.0.bias": torch.zeros(512),
            "module.head.3.weight": torch.zeros(12, 512),
            "module.head.3.bias": torch.zeros(12),
        }
    }
    assert sorted(unwrap_state_dict(fake)) == [
        "head.0.bias",
        "head.0.weight",
        "head.3.bias",
        "head.3.weight",
        "proj.bias",
        "proj.weight",
    ]
    log("self_check_ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
        raise SystemExit(0)
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
