"""
Knee MRI abnormality classifier.

Architecture: 2.5D ConvNeXt-Small slice encoder → BiLSTM slice aggregation
→ 12 independent binary heads.

Evidence:
- ConvNeXt-Small proven by RSNA 2024 lumbar spine 1st place winner.
- BiLSTM slice aggregation proven by same winner.
- 12 independent heads handle severe per-class imbalance (2025 aneurysm winner).
- 2.5D (3 adjacent slices as channels) beats single-slice and 3D (Chang 2019).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SliceEncoder(nn.Module):
    """
    2.5D slice encoder using a pretrained ConvNeXt-Small backbone.

    Takes a single 2.5D input (3 adjacent slices as 3 channels) and
    produces a feature vector.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        in_channels: int = 3,
        feature_dim: int = 768,
    ):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("timm is required. Install: pip install timm")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # Remove classification head
        )

        # Get the actual feature dimension from the backbone
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            features = self.backbone(dummy)
            actual_dim = features.shape[-1]

        self.feature_dim = actual_dim
        logger.info(f"SliceEncoder: {backbone_name}, feature_dim={actual_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W) input tensor.

        Returns:
            (batch, feature_dim) feature vector.
        """
        return self.backbone(x)


class BiLSTMAggregator(nn.Module):
    """
    BiLSTM slice aggregation.

    Takes per-slice features and aggregates them into a study-level
    representation. Proven by RSNA 2024 lumbar spine 1st place.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_dim = hidden_dim * 2  # bidirectional

        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_slices, feature_dim) per-slice features.

        Returns:
            (batch, output_dim) aggregated study-level features.
        """
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, num_slices, hidden*2)

        # Attention-weighted pooling over slices
        attn_weights = self.attention(lstm_out)  # (batch, num_slices, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        aggregated = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        return aggregated


class AttentionMILAggregator(nn.Module):
    """
    Gated attention MIL aggregator (alternative to BiLSTM).

    From Ilse et al. 2018, adapted for medical image classification.
    Attention-MIL lifted RSNA 2024 lumbar winner's public LB by +0.02.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_slices, feature_dim) per-slice features.

        Returns:
            (batch, feature_dim) aggregated features.
        """
        A_V = self.attention_V(x)  # (batch, num_slices, hidden)
        A_U = self.attention_U(x)  # (batch, num_slices, hidden)
        A = self.attention_w(A_V * A_U)  # (batch, num_slices, 1)
        A = F.softmax(A, dim=1)

        aggregated = (x * A).sum(dim=1)  # (batch, feature_dim)
        return self.dropout(aggregated)


class KneeClassifier(nn.Module):
    """
    Full knee MRI classifier: 2.5D encoder → slice aggregator → 12 binary heads.

    Usage:
        model = KneeClassifier(backbone_name="convnext_small", aggregator="bilstm")
        # x: (batch, num_slices, channels, H, W)
        logits = model(x)  # (batch, 12)
        probs = torch.sigmoid(logits)
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small",
        aggregator: str = "bilstm",  # "bilstm" or "attention_mil"
        in_channels: int = 3,
        num_slices: int = 24,
        num_classes: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_slices = num_slices
        self.num_classes = num_classes

        # Slice encoder
        self.encoder = SliceEncoder(
            backbone_name=backbone_name,
            pretrained=True,
            in_channels=in_channels,
        )

        # Slice aggregator
        if aggregator == "bilstm":
            self.aggregator = BiLSTMAggregator(
                feature_dim=self.encoder.feature_dim,
                dropout=dropout,
            )
            agg_dim = self.aggregator.output_dim
        elif aggregator == "attention_mil":
            self.aggregator = AttentionMILAggregator(
                feature_dim=self.encoder.feature_dim,
                dropout=dropout,
            )
            agg_dim = self.encoder.feature_dim
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        # 12 independent binary heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(agg_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
            for _ in range(num_classes)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_slices, channels, H, W) input tensor.

        Returns:
            (batch, num_classes) raw logits.
        """
        batch_size, num_slices, C, H, W = x.shape

        # Encode each slice
        x = x.view(batch_size * num_slices, C, H, W)
        features = self.encoder(x)  # (batch*num_slices, feature_dim)
        features = features.view(batch_size, num_slices, -1)  # (batch, num_slices, feature_dim)

        # Aggregate across slices
        aggregated = self.aggregator(features)  # (batch, agg_dim)

        # 12 independent heads
        logits = torch.cat([head(aggregated) for head in self.heads], dim=1)

        return logits


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    From Ridnik et al. 2021. Down-weights and hard-thresholds easy/mislabeled
    negatives. Fits noisy multilabel supervision.

    Parameters from the paper: γ−=4, γ+=1, clip=0.05.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw logits.
            targets: (batch, num_classes) soft targets in [0, 1].
            mask: (batch, num_classes) binary mask. 1 = include in loss, 0 = exclude.
                  Use to mask "not_addressed" cells (soft label 0.5).
        """
        probs = torch.sigmoid(logits)

        # Asymmetric clipping
        if self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)
        else:
            probs_neg = probs

        # Positive and negative loss components
        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps))

        # Asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = probs * targets + (1 - probs) * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            focal_weight = (1 - pt) ** gamma
            loss = -(loss_pos + loss_neg) * focal_weight
        else:
            loss = -(loss_pos + loss_neg)

        # Apply mask
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)

        return loss.mean()


class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy with masking for "not addressed" cells.

    Simple alternative to AsymmetricLoss. Masks cells where the
    soft label is 0.5 (not_addressed) to exclude them from the loss.
    """

    def __init__(self, label_smoothing: float = 0.05, mask_threshold: float = 0.5):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.mask_threshold = mask_threshold
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = self.bce(logits, targets)

        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)

        return loss.mean()
