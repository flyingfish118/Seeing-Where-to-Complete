"""Visual--geometric prototype predictors used by DINO-VGP and C-VGP."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _conv_bn_act(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class DepthwiseBlock(nn.Module):
    """Mobile-style image block so all three SCC views share one encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PointSetEncoder(nn.Module):
    """PointNet encoder; max/mean pooling makes the input order irrelevant."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(inplace=True),
            nn.Conv1d(width, width * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(width * 2, width * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.SiLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.net(points.transpose(1, 2).contiguous())
        return torch.cat([features.amax(dim=-1), features.mean(dim=-1)], dim=1)


class SharedViewEncoder(nn.Module):
    """Encode three SCC views with shared lightweight convolutional weights."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_act(3, width, stride=2),
            DepthwiseBlock(width, width * 2, stride=2),
            DepthwiseBlock(width * 2, width * 3, stride=2),
            DepthwiseBlock(width * 3, width * 4, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward_tokens(self, views: torch.Tensor) -> torch.Tensor:
        """Return one lightweight feature token for each SCC view."""
        batch_size, n_views, channels, height, width = views.shape
        encoded = self.net(views.reshape(batch_size * n_views, channels, height, width))
        return encoded.flatten(1).reshape(batch_size, n_views, -1)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        encoded = self.forward_tokens(views)
        return torch.cat([encoded.amax(dim=1), encoded.mean(dim=1)], dim=1)


class FrozenDINOv3ViewEncoder(nn.Module):
    """Frozen DINOv3 visual features from three SCC views.

    The backbone sees only SCC renderings. Geometry remains in the separate
    point-set encoder, avoiding a coordinate-as-text serialization entirely.
    """

    def __init__(self, model_name: str, checkpoint: str, image_size: int) -> None:
        super().__init__()
        import timm
        from safetensors.torch import load_file

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint_path}")
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=image_size)
        self.model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = int(self.model.num_features)

    def train(self, mode: bool = True) -> "FrozenDINOv3ViewEncoder":
        # Keep frozen BatchNorm/DropPath behavior stable even while the student trains.
        super().train(False)
        return self

    def forward_tokens(self, views: torch.Tensor) -> torch.Tensor:
        """Return one frozen global DINO token per SCC camera view."""
        batch_size, n_views, channels, height, width = views.shape
        # No gradients or activation cache are retained for the frozen foundation model.
        with torch.no_grad():
            encoded = self.model(views.reshape(batch_size * n_views, channels, height, width))
        return encoded.reshape(batch_size, n_views, -1)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        encoded = self.forward_tokens(views)
        return torch.cat([encoded.amax(dim=1), encoded.mean(dim=1)], dim=1)


class SVDViewAwareFusion(nn.Module):
    """SVDFormer-style fusion of point features with positioned SCC tokens.

    SVDFormer concatenates each view feature with a repeated global point
    feature, adds a camera-position embedding, aligns the three tokens with
    self-attention, and max-pools the aligned view evidence.  Here the SCC
    cameras are the repository's actual ``-y``, ``y``, and ``z`` directions.
    """

    def __init__(self, point_dim: int, view_dim: int, hidden_dim: int = 256, view_distance: float = 1.0) -> None:
        super().__init__()
        if hidden_dim % 8 != 0:
            raise ValueError("SVD view-fusion hidden_dim must be divisible by 8")
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.view_proj = nn.Linear(view_dim, hidden_dim)
        self.token_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        distance = float(view_distance)
        # View order is fixed in litevcp.data.VIEW_SUFFIXES: -y, y, z.
        self.register_buffer(
            "view_positions",
            torch.tensor([[0.0, -distance, 0.0], [0.0, distance, 0.0], [0.0, 0.0, distance]]),
            persistent=False,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, point_features: torch.Tensor, view_tokens: torch.Tensor) -> torch.Tensor:
        if view_tokens.ndim != 3 or view_tokens.shape[1] != 3:
            raise ValueError("SVD view fusion expects exactly three SCC view tokens")
        point = self.point_proj(point_features)
        view = self.view_proj(view_tokens)
        tokens = self.token_proj(torch.cat([view, point.unsqueeze(1).expand(-1, 3, -1)], dim=-1))
        tokens = tokens + self.position_mlp(self.view_positions).unsqueeze(0)
        query = self.norm_q(tokens)
        tokens = tokens + self.attn(query, query, query, need_weights=False)[0]
        tokens = self.norm_attn(tokens)
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        view_global = tokens.amax(dim=1)
        return torch.cat([point, view_global], dim=1)


class LiteVCPSetKD(nn.Module):
    """Predict an unordered K-point defect prototype from points and SCC views.

    The decoder emits point coordinates, but training is entirely set-based via
    Chamfer and D2P-field losses. It therefore has no coordinate-token order
    dependence and needs no language-model inference at deployment.
    """

    def __init__(self, num_prototype_points: int = 30, width: int = 64) -> None:
        super().__init__()
        self.num_prototype_points = num_prototype_points
        self.point_encoder = PointSetEncoder(width)
        self.view_encoder = SharedViewEncoder(width)

        point_dim = width * 8
        view_dim = width * 8
        hidden_dim = width * 8
        self.fusion = nn.Sequential(
            nn.Linear(point_dim + view_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.prototype_head = nn.Linear(hidden_dim, num_prototype_points * 3)

    def forward(self, points: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        point_features = self.point_encoder(points)
        view_features = self.view_encoder(views)
        fused = self.fusion(torch.cat([point_features, view_features], dim=1))
        return self.prototype_head(fused).view(-1, self.num_prototype_points, 3)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LiteDINOPrototypeStudent(nn.Module):
    """C-VGP compatibility class with frozen-DINO feature alignment.

    DINO features are cached offline during training.  This module therefore
    contains no foundation-model dependency at deployment: its shared CNN
    emits one token per SCC view, projects it into the reference token space,
    and fuses pooled visual evidence with the partial-cloud feature.
    """

    def __init__(
        self,
        num_prototype_points: int = 30,
        width: int = 96,
        teacher_feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.num_prototype_points = int(num_prototype_points)
        self.point_encoder = PointSetEncoder(width)
        self.view_encoder = SharedViewEncoder(width)
        view_token_dim = width * 4
        self.view_projection = nn.Sequential(
            nn.LayerNorm(view_token_dim),
            nn.Linear(view_token_dim, int(teacher_feature_dim), bias=False),
        )
        point_dim = width * 8
        visual_dim = int(teacher_feature_dim) * 2
        hidden_dim = width * 8
        self.fusion = nn.Sequential(
            nn.Linear(point_dim + visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.prototype_head = nn.Linear(hidden_dim, self.num_prototype_points * 3)

    def forward_with_view_tokens(self, points: torch.Tensor, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        point_features = self.point_encoder(points)
        view_tokens = self.view_projection(self.view_encoder.forward_tokens(views))
        visual_features = torch.cat([view_tokens.amax(dim=1), view_tokens.mean(dim=1)], dim=1)
        fused = self.fusion(torch.cat([point_features, visual_features], dim=1))
        prototype = self.prototype_head(fused).view(-1, self.num_prototype_points, 3)
        return prototype, view_tokens

    def forward(self, points: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        return self.forward_with_view_tokens(points, views)[0]

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class DINOv3SetKD(nn.Module):
    """DINO-VGP compatibility class and its single-modality ablations.

    ``use_view_encoder`` and ``use_point_encoder`` implement the visual-only,
    point-only, and visual-geometric ablations with one shared decoder.
    """

    def __init__(
        self,
        num_prototype_points: int = 30,
        width: int = 96,
        dino_model_name: str = "vit_large_patch16_dinov3",
        dino_checkpoint: str = "",
        image_size: int = 256,
        use_point_encoder: bool = True,
        use_view_encoder: bool = True,
        fusion_type: str = "concat",
        svd_hidden_dim: int = 256,
        svd_view_distance: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_prototype_points = num_prototype_points
        self.use_point_encoder = bool(use_point_encoder)
        self.use_view_encoder = bool(use_view_encoder)
        self.fusion_type = str(fusion_type).lower()
        if not self.use_point_encoder and not self.use_view_encoder:
            raise ValueError("At least one of point or view encoders must be enabled")
        self.point_encoder = PointSetEncoder(width) if self.use_point_encoder else None
        self.view_encoder = (
            FrozenDINOv3ViewEncoder(dino_model_name, dino_checkpoint, image_size)
            if self.use_view_encoder else None
        )
        point_dim = width * 8
        view_dim = self.view_encoder.feature_dim * 2 if self.view_encoder is not None else 0
        hidden_dim = width * 8
        direct_fusion_input_dim = (point_dim if self.use_point_encoder else 0) + view_dim
        self.svd_view_fusion = None
        self.svd_residual = None
        self.svd_gate = None
        if self.fusion_type not in {"concat", "svdformer"}:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        # Retain the direct-concatenation path in the SVD ablation. Its
        # zero-initialized residual makes epoch zero exactly the established
        # concat student, while view-aware interaction can be learned only if
        # it improves validation performance.
        fusion_input_dim = direct_fusion_input_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.prototype_head = nn.Linear(hidden_dim, num_prototype_points * 3)
        if self.fusion_type == "svdformer":
            if self.point_encoder is None or self.view_encoder is None:
                raise ValueError("SVDFormer-style fusion requires both point and SCC view encoders")
            self.svd_view_fusion = SVDViewAwareFusion(
                point_dim=point_dim,
                view_dim=self.view_encoder.feature_dim,
                hidden_dim=int(svd_hidden_dim),
                view_distance=float(svd_view_distance),
            )
            self.svd_residual = nn.Sequential(
                nn.LayerNorm(self.svd_view_fusion.output_dim),
                nn.Linear(self.svd_view_fusion.output_dim, hidden_dim, bias=False),
            )
            self.svd_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward_with_view_tokens(self, points: torch.Tensor, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a prototype and expose per-view DINO tokens when available."""
        if self.svd_view_fusion is not None:
            assert self.point_encoder is not None and self.view_encoder is not None
            point_features = self.point_encoder(points)
            view_tokens = self.view_encoder.forward_tokens(views)
            view_features = torch.cat([view_tokens.amax(dim=1), view_tokens.mean(dim=1)], dim=1)
            direct_features = torch.cat([point_features, view_features], dim=1)
            fused = self.fusion(direct_features)
            assert self.svd_residual is not None and self.svd_gate is not None
            residual = self.svd_residual(self.svd_view_fusion(point_features, view_tokens))
            fused = fused + torch.tanh(self.svd_gate) * residual
            return self.prototype_head(fused).view(-1, self.num_prototype_points, 3), view_tokens
        features = []
        view_tokens = None
        if self.point_encoder is not None:
            features.append(self.point_encoder(points))
        if self.view_encoder is not None:
            view_tokens = self.view_encoder.forward_tokens(views)
            features.append(torch.cat([view_tokens.amax(dim=1), view_tokens.mean(dim=1)], dim=1))
        fused_input = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        fused = self.fusion(fused_input)
        if view_tokens is None:
            raise ValueError("This predictor has no SCC view encoder or DINO tokens")
        return self.prototype_head(fused).view(-1, self.num_prototype_points, 3), view_tokens

    def forward(self, points: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        # Point-only ablation has no visual token to return, but it still
        # uses the identical PointNet and prototype decoder path.
        if not self.use_view_encoder:
            assert self.point_encoder is not None
            return self.prototype_head(self.fusion(self.point_encoder(points))).view(
                -1, self.num_prototype_points, 3
            )
        return self.forward_with_view_tokens(points, views)[0]

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
