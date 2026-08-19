from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "routed_dino_train" / "routed_dino_train.py"
OUTPUT = Path(__file__).resolve().parent / "path4_knowledge_transformer_train.py"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return source.replace(old, new, 1)


def main() -> None:
    source = SOURCE.read_text()

    doc_end = source.index('"""', 3) + 3
    source = (
        '"""Path 4 knowledge-first DINOv2 volume transformer for RSNA Knee MRI.\n\n'
        "This clean experiment keeps the 58 expert-labelled studies monitor-only. "
        "It routes six DICOM-derived sequence families, decodes three ordered contiguous "
        "2.5D windows per sequence, and reasons jointly over them with target-specific "
        "anatomical routing and a small Transformer. DINOv2-S is warm-started from the "
        "clean localized family; only its last block and the new head are trainable. "
        "Training and model selection use scanner-isolated non-gold folds and masked "
        "report supervision. The run emits checkpoints and OOF evidence only.\n"
        '"""'
        + source[doc_end:]
    )

    constants = {
        "SEED = 20260813": "SEED = 20260817",
        "EPOCHS = 5": "EPOCHS = 3",
        "IMG_SIZE = 280": "IMG_SIZE = 224",
        "BATCH_STUDIES = 4": "BATCH_STUDIES = 2",
        "EVAL_BATCH = 6": "EVAL_BATCH = 2",
        "UNFREEZE_LAST = 4": "UNFREEZE_LAST = 1",
        "TIME_LIMIT_S = 7.8 * 3600": "TIME_LIMIT_S = 10.5 * 3600",
        "MIN_OOF_GAIN = 0.002": "MIN_OOF_GAIN = 0.003",
        "MIN_TARGET_GAIN = 0.001": "MIN_TARGET_GAIN = 0.0015",
        "MAX_FOLD_REGRESSION = 0.015": "MAX_FOLD_REGRESSION = 0.010",
        "MAX_MACRO_FOLD_REGRESSION = 0.002": "MAX_MACRO_FOLD_REGRESSION = 0.003",
        "BLEND_WEIGHTS = np.linspace(0.0, 1.0, 5)": (
            "BLEND_WEIGHTS = np.asarray([0.0, 0.10, 0.20, 0.35, 0.50, 0.70], dtype=np.float32)"
        ),
    }
    for old, new in constants.items():
        source = replace_once(source, old, new, f"constant {old}")

    old_targets = '''SPECIALIST_TARGETS = {"ACL", "Contusion", "Fracture"}

# DICOM-derived slots. Missing acquisitions remain masked and are never substituted
'''
    new_targets = '''SPECIALIST_TARGETS = set(TARGETS)

# Clinical routing is an inductive bias, not supervision. A target can attend only to
# the sequence families ordinarily used to assess that structure. If all routed slots
# are missing for one study, the head falls back to every acquired slot rather than
# fabricating a sequence.
TARGET_ROUTES = {
    "ACL": ("SAG_FLUID_FS", "COR_FLUID_FS", "SAG_FLUID_NOFS", "SAG_T1"),
    "MCL": ("COR_FLUID_FS", "SAG_FLUID_FS", "COR_T1"),
    "Medial Meniscus": ("SAG_FLUID_FS", "COR_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"),
    "Lateral Meniscus": ("SAG_FLUID_FS", "COR_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"),
    "Medial OA": ("COR_T1", "SAG_T1", "COR_FLUID_FS", "AX_FLUID_FS"),
    "Lateral OA": ("COR_T1", "SAG_T1", "COR_FLUID_FS", "AX_FLUID_FS"),
    "PF OA": ("AX_FLUID_FS", "SAG_FLUID_FS", "SAG_T1"),
    "Effusion": ("AX_FLUID_FS", "SAG_FLUID_FS", "COR_FLUID_FS"),
    "Synovitis": ("AX_FLUID_FS", "SAG_FLUID_FS", "COR_FLUID_FS"),
    "Baker's": ("SAG_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS"),
    "Contusion": ("SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "COR_T1", "SAG_T1"),
    "Fracture": ("SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "COR_T1", "SAG_T1"),
}

# DICOM-derived slots. Missing acquisitions remain masked and are never substituted
'''
    source = replace_once(source, old_targets, new_targets, "target routes")

    window_pool_start = source.index("# Per-target anatomical-window aggregation.")
    window_pool_end = source.index("\n\nFATSAT_OPTIONS", window_pool_start)
    source = (
        source[:window_pool_start]
        + "# All three physical windows are consumed jointly by the volume Transformer."
        + source[window_pool_end:]
    )

    class_start = source.index("class LocalizedSlotHead")
    class_end = source.index("\ndef build_model", class_start)
    new_classes = r'''class TargetFamilyTransformerHead(nn.Module):
    """Jointly aggregate ordered windows and routed sequence families per target."""

    def __init__(self, dim: int, n_slots: int, n_outputs: int, hidden: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.10)
        )
        self.slot_embedding = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.window_embedding = nn.Parameter(torch.randn(N_GROUPS, hidden) * 0.02)
        self.target_embedding = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 3,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.query = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.output_weight = nn.Parameter(torch.randn(n_outputs, hidden) * 0.02)
        self.output_bias = nn.Parameter(torch.zeros(n_outputs))
        slot_names = [slot[0] for slot in SLOTS]
        route = torch.zeros(n_outputs, n_slots, dtype=torch.bool)
        for target_index, target in enumerate(TARGETS):
            unknown = set(TARGET_ROUTES[target]) - set(slot_names)
            if unknown:
                raise ValueError(f"Unknown routed slots for {target}: {sorted(unknown)}")
            for name in TARGET_ROUTES[target]:
                route[target_index, slot_names.index(name)] = True
        self.register_buffer("route_mask", route, persistent=True)
        self.hidden = hidden

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: [batch, slot, ordered-window, target, feature]
        batch, slots, windows, outputs = features.shape[:4]
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[None, :, None, None, :]
        hidden = hidden + self.window_embedding[None, None, :, None, :]
        hidden = hidden + self.target_embedding[None, None, None, :, :]
        hidden = hidden.permute(0, 3, 1, 2, 4).reshape(
            batch * outputs, slots * windows, self.hidden
        )

        present = mask[:, None, :, None].bool().expand(-1, outputs, -1, windows)
        routed = self.route_mask[None, :, :, None].expand(batch, -1, -1, windows)
        allowed = present & routed
        empty = ~allowed.flatten(2).any(-1)
        if empty.any():
            allowed = torch.where(empty[:, :, None, None], present, allowed)
        key_padding = ~allowed.reshape(batch * outputs, slots * windows)

        encoded = self.transformer(hidden, src_key_padding_mask=key_padding)
        query = self.query[None].expand(batch, -1, -1).reshape(
            batch * outputs, self.hidden
        )
        attention = torch.einsum("blh,bh->bl", encoded, query) / self.hidden**0.5
        attention = attention.masked_fill(key_padding, -1e4).softmax(-1)
        context = torch.einsum("bl,blh->bh", attention, encoded)
        weight = self.output_weight[None].expand(batch, -1, -1).reshape(
            batch * outputs, self.hidden
        )
        bias = self.output_bias[None].expand(batch, -1).reshape(batch * outputs)
        return ((context * weight).sum(-1) + bias).reshape(batch, outputs)


class KneeDINO(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        dim = backbone.config.hidden_size
        self.patch_query = nn.Parameter(torch.randn(len(TARGETS), dim) * 0.02)
        self.head = TargetFamilyTransformerHead(dim * 2, len(SLOTS), len(TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slots, windows = images.shape[:3]
        pixels = images.reshape(batch * slots * windows, *images.shape[3:]).float().div_(255.0)
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
        features = features.reshape(batch, slots, windows, len(TARGETS), -1)
        return self.head(features, mask)

'''
    source = source[:class_start] + new_classes + source[class_end:]

    init_start = source.index("def initialize_from_localized")
    init_end = source.index("\ndef macro_auc", init_start)
    new_init = r'''def initialize_from_localized(model: KneeDINO, checkpoint_path: Path, fold: int) -> None:
    """Warm-start only the compatible clean DINO and target patch-query weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "localized_target_patch_attention_v1":
        raise ValueError(f"Fold {fold} localized checkpoint has unexpected architecture")
    if checkpoint.get("fold") != fold or checkpoint.get("gold_training_count") != 0:
        raise ValueError(f"Fold {fold} localized checkpoint failed provenance validation")
    state = checkpoint["state_dict"]
    compatible = {
        key: value for key, value in state.items()
        if key.startswith("backbone.") or key == "patch_query"
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected or not compatible or "patch_query" not in compatible:
        raise ValueError(f"Fold {fold} warm-start compatibility failure")
    if any(key.startswith("backbone.") for key in missing):
        raise ValueError(f"Fold {fold} warm start omitted backbone parameters")
    log(
        f"fold {fold}: warm-started DINO and patch queries from "
        f"{checkpoint_path.name}; new volume Transformer initialised"
    )

'''
    source = source[:init_start] + new_init + source[init_end:]

    take_start = source.index("def take_group")
    take_end = source.index("\ndef make_clean_split", take_start)
    new_take = r'''def take_all_groups(cache_rows: np.ndarray) -> np.ndarray:
    """Expose all three ordered contiguous windows to the model jointly."""
    batch, slots, slices, height, width = cache_rows.shape
    if slices != CACHE_SLICES or CACHE_SLICES != GROUP_SIZE * N_GROUPS:
        raise ValueError("Path4 cache/window contract drift")
    return cache_rows.reshape(batch, slots, N_GROUPS, GROUP_SIZE, height, width)


@torch.no_grad()
def predict(model, cache, mask, indices, device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(indices), EVAL_BATCH):
        selected = indices[start:start + EVAL_BATCH]
        images = torch.from_numpy(take_all_groups(cache[selected])).to(
            device, non_blocking=True
        )
        present = torch.from_numpy(mask[selected]).to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images, present)
        outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outputs)


def affine_augment(images: torch.Tensor) -> torch.Tensor:
    original_shape = images.shape
    channels, height, width = original_shape[-3:]
    flat = images.reshape(-1, channels, height, width).float()
    count = flat.shape[0]
    angles = (torch.rand(count, device=flat.device) - 0.5) * (10 * math.pi / 180)
    scales = 0.95 + torch.rand(count, device=flat.device) * 0.10
    tx = (torch.rand(count, device=flat.device) - 0.5) * 0.06
    ty = (torch.rand(count, device=flat.device) - 0.5) * 0.06
    theta = torch.zeros(count, 2, 3, device=flat.device)
    theta[:, 0, 0] = scales * torch.cos(angles)
    theta[:, 0, 1] = -scales * torch.sin(angles)
    theta[:, 1, 0] = scales * torch.sin(angles)
    theta[:, 1, 1] = scales * torch.cos(angles)
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    gain = 0.92 + torch.rand(count, 1, 1, 1, device=flat.device) * 0.16
    bias = (torch.rand(count, 1, 1, 1, device=flat.device) - 0.5) * 12
    return (flat * gain + bias).clamp(0, 255).reshape(original_shape)

'''
    source = source[:take_start] + new_take + source[take_end:]

    old_train_images = '''            group = int(np.random.randint(N_GROUPS))
            images = torch.from_numpy(take_group(cache[selected], group)).to(
                device, non_blocking=True
            )
'''
    new_train_images = '''            images = torch.from_numpy(take_all_groups(cache[selected])).to(
                device, non_blocking=True
            )
'''
    source = replace_once(source, old_train_images, new_train_images, "joint window training")

    load_helper_anchor = "\ndef choose_target_blend(\n"
    load_helper_pos = source.index(load_helper_anchor)
    load_frontier = r'''

def load_frontier_oof(
    oof_path: Path,
    eligible_studies: np.ndarray,
) -> np.ndarray:
    """Load the clean frontier parent's full non-gold OOF matrix by study identity."""
    with np.load(oof_path, allow_pickle=False) as bundle:
        expected = {"ids", "pred", "y_derived", "gold_mask", "targets"}
        if set(bundle.files) != expected:
            raise ValueError(f"Unexpected frontier OOF members: {bundle.files}")
        if bundle["targets"].astype(str).tolist() != TARGETS:
            raise ValueError("Frontier OOF target order drift")
        ids = bundle["ids"].astype(str)
        predictions = bundle["pred"].astype(np.float32)
    if len(ids) != len(set(ids)):
        raise ValueError("Frontier OOF contains duplicate study IDs")
    index = {study: position for position, study in enumerate(ids)}
    missing = [study for study in eligible_studies if study not in index]
    if missing:
        raise ValueError(f"Frontier OOF misses {len(missing)} Path4 studies")
    aligned = predictions[[index[study] for study in eligible_studies]]
    if not np.isfinite(aligned).all():
        raise ValueError("Frontier OOF contains non-finite predictions")
    return rank_predictions(aligned)
'''
    source = source[:load_helper_pos] + load_frontier + source[load_helper_pos:]

    source = replace_once(
        source,
        '''    previous_oof_path = find_unique_artifact(
        "localized_oof_predictions.csv", "localized-dinov2"
    )
    localized_checkpoints = find_localized_checkpoints()
''',
        '''    previous_oof_path = find_unique_artifact(
        "localized_oof_predictions.csv", "localized-dinov2"
    )
    frontier_oof_path = find_unique_artifact("oof.npz", "rsna-knee-weights")
    localized_checkpoints = find_localized_checkpoints()
''',
        "frontier OOF path",
    )
    source = replace_once(
        source,
        '''    previous_oof, baseline_oof = load_previous_candidate(
        previous_oof_path, eligible_studies
    )
''',
        '''    previous_oof, _ = load_previous_candidate(
        previous_oof_path, eligible_studies
    )
    baseline_oof = load_frontier_oof(frontier_oof_path, eligible_studies)
''',
        "frontier OOF baseline",
    )

    source = source.replace("routed_dino_fold", "path4_dino_fold")
    source = source.replace("routed_oof_predictions.csv", "path4_oof_predictions.csv")
    source = source.replace(
        '"routed_specialist_transfer_v2"',
        '"path4_target_family_slice_transformer_v1"',
    )
    source = source.replace(
        '"warm_start": "localized_target_patch_attention_v1",',
        '"warm_start": "localized DINO backbone and patch queries only",',
    )
    source = source.replace(
        '"window_pool": WINDOW_POOL,\n',
        '"target_routes": TARGET_ROUTES,\n',
    )
    source = source.replace(
        '"specialist_targets": sorted(SPECIALIST_TARGETS),',
        '"specialist_targets": sorted(SPECIALIST_TARGETS),\n'
        '            "volume_aggregation": "two-layer target-routed Transformer over 18 ordered sequence-window tokens",',
        1,
    )
    source = source.replace(
        "target_gate = len(selected_specialists) >= 2",
        "target_gate = len(selected_specialists) >= 4",
    )
    source = source.replace(
        "routed family gold monitor",
        "Path4 family gold monitor",
    )
    source = source.replace(
        "training output contains checkpoints and OOF evidence only; hidden inference is separate",
        "Path4 training output contains checkpoints and OOF evidence only; hidden inference is separate",
    )
    source = source.replace(
        "Routed OOF gates did not pass",
        "Path4 OOF gates did not pass",
    )
    source = source.replace(
        "Routed output is missing guarded-blend OOF evidence",
        "Path4 output is missing guarded-blend OOF evidence",
    )
    source = replace_once(
        source,
        '''    Path("metrics.json").write_text(json.dumps({
        "folds": fold_histories,
''',
        '''    Path("metrics.json").write_text(json.dumps({
        "path": "Path4",
        "experiment": "knowledge-first target-family volume Transformer",
        "gold_policy": "58 official studies excluded from training, checkpoint selection, and blend selection; monitor only",
        "gold_training_or_selection_used": False,
        "cv_grouping": "scanner/site DICOM fingerprint with folds locked to prior exact OOF",
        "frontier_oof_sha256": hashlib.sha256(frontier_oof_path.read_bytes()).hexdigest(),
        "folds": fold_histories,
''',
        "Path4 metric provenance",
    )
    source = replace_once(
        source,
        '''    Path("weights_manifest.json").write_text(json.dumps({
        "architecture": "path4_target_family_slice_transformer_v1",
''',
        '''    Path("weights_manifest.json").write_text(json.dumps({
        "architecture": "path4_target_family_slice_transformer_v1",
        "gold_training_or_selection_used": False,
        "cv_grouping": "scanner/site DICOM fingerprint",
        "frontier_oof_sha256": hashlib.sha256(frontier_oof_path.read_bytes()).hexdigest(),
''',
        "Path4 manifest provenance",
    )

    OUTPUT.write_text(source)

    metadata = {
        "id": "aakashkavuru/rsna-knee-path4-knowledge-transformer-train",
        "title": "RSNA Knee Path4 Knowledge Transformer Train",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "stevenleehans/rsna-knee-llm-report-labels",
            "pilkwang/rsna-knee-weights",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": ["aakashkavuru/rsna-knee-localized-dinov2"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    }
    (OUTPUT.parent / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"wrote {OUTPUT} ({source.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
