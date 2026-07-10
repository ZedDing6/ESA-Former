"""ESA-Former for multispectral image classification."""

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch_dct

Tensor = torch.Tensor


def drop_path(
    x: Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> Tensor:
    """Apply stochastic depth to a residual branch."""
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device,
    )
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic depth applied per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """Feed-forward network used in each Transformer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attention_drop_rate: float = 0.0,
        projection_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop_rate)
        self.projection = nn.Linear(dim, dim)
        self.projection_drop = nn.Dropout(projection_drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, channels = x.shape

        qkv = self.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_drop(attention.softmax(dim=-1))

        x = (attention @ value).transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            channels,
        )
        x = self.projection(x)
        return self.projection_drop(x)


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attention_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attention = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention_drop_rate=attention_drop_rate,
            projection_drop_rate=drop_rate,
        )
        self.drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop_rate=drop_rate,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SpectralEncoder(nn.Module):
    """Encode the ordered spectral sequence at each pooled spatial location."""

    def __init__(
        self,
        out_channels: int,
        spatial_size: int = 24,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()

        self.spatial_pool = nn.AdaptiveAvgPool2d(
            (spatial_size, spatial_size)
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.spatial_pool(x)
        batch_size, channels, height, width = x.shape

        x = x.permute(0, 2, 3, 1).reshape(
            batch_size * height * width,
            1,
            channels,
        )
        x = self.encoder(x).mean(dim=-1)

        return x.reshape(
            batch_size,
            height,
            width,
            -1,
        ).permute(0, 3, 1, 2)


class CoordinateFrequencyResponse(nn.Module):
    """Generate channel-specific frequency responses from coordinates and complexity."""

    def __init__(
        self,
        height: int,
        width: int,
        descriptor_dim: int = 1,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()

        coordinate = self._build_frequency_coordinate(height, width)
        self.register_buffer(
            "coordinate",
            coordinate,
            persistent=False,
        )

        self.position_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.descriptor_embedding = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.response_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _build_frequency_coordinate(
        height: int,
        width: int,
    ) -> Tensor:
        u = torch.arange(height, dtype=torch.float32).view(height, 1)
        v = torch.arange(width, dtype=torch.float32).view(1, width)

        u = u.expand(height, width) / max(height - 1, 1)
        v = v.expand(height, width) / max(width - 1, 1)
        radius = torch.sqrt(u.square() + v.square())
        return radius.unsqueeze(-1)

    def forward(self, descriptor: Tensor) -> Tensor:
        if descriptor.shape[-1] != 1:
            raise ValueError(
                "The channel descriptor must have shape [B, C, 1]."
            )

        position_feature = self.position_embedding(self.coordinate)
        descriptor_feature = self.descriptor_embedding(descriptor)

        fused_feature = (
            position_feature.unsqueeze(0).unsqueeze(0)
            + descriptor_feature.unsqueeze(2).unsqueeze(3)
        )
        response = self.response_head(fused_feature).squeeze(-1)
        return 0.5 + torch.sigmoid(response)


class FSSCA(nn.Module):
    """Frequency-guided spectral-spatial collaborative attention."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps

        self.channel_projection = nn.Linear(channels, channels)
        self.frequency_response = CoordinateFrequencyResponse(
            height=height,
            width=width,
            descriptor_dim=1,
            hidden_dim=16,
        )
        self.spatial_smoothing = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self._initialize_smoothing_kernel()

    def _initialize_smoothing_kernel(self) -> None:
        with torch.no_grad():
            self.spatial_smoothing.weight.fill_(1.0 / 9.0)

    @staticmethod
    def _dct_2d(x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size * channels, height, width)
        x = torch_dct.dct_2d(x, norm="ortho")
        return x.reshape(batch_size, channels, height, width)

    @staticmethod
    def _idct_2d(x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size * channels, height, width)
        x = torch_dct.idct_2d(x, norm="ortho")
        return x.reshape(batch_size, channels, height, width)

    def _channel_complexity(self, x_frequency: Tensor) -> Tensor:
        _, _, height, width = x_frequency.shape

        high_frequency = x_frequency[
            :,
            :,
            3 * height // 4 :,
            3 * width // 4 :,
        ].abs().sum(dim=(2, 3))
        total_frequency = x_frequency.abs().sum(dim=(2, 3))

        return high_frequency / (total_frequency + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        def frequency_block(x_inner: Tensor) -> Tensor:
            x_frequency = self._dct_2d(x_inner)
            complexity = self._channel_complexity(x_frequency)

            channel_attention = torch.sigmoid(
                self.channel_projection(complexity)
            )
            response = self.frequency_response(
                complexity.unsqueeze(-1)
            )

            x_frequency = x_frequency * response
            x_spatial = self._idct_2d(x_frequency)
            x_spatial = x_spatial * channel_attention[:, :, None, None]

            spatial_attention = torch.sigmoid(
                self.spatial_smoothing(x_spatial)
            )
            return x_inner + x_inner * spatial_attention

        return checkpoint.checkpoint(
            frequency_block,
            x,
            use_reentrant=False,
        )


class SpectralSpatialTokenizer(nn.Module):
    """Generate aligned spectral and spatial tokens for ESA-Former."""

    def __init__(
        self,
        image_size: int = 200,
        in_channels: int = 5,
        embed_dim: int = 192,
        kernel_size: int = 9,
        stride: int = 8,
        padding: int = 0,
        token_size: int = 24,
    ) -> None:
        super().__init__()

        self.spectral_encoder = SpectralEncoder(
            out_channels=embed_dim,
            spatial_size=token_size,
        )
        self.horizontal_projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.vertical_projection = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, padding),
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        token_height = (
            image_size + 2 * padding - kernel_size
        ) // stride + 1
        token_width = token_height
        if token_height != token_size or token_width != token_size:
            raise ValueError(
                "The spatial tokenizer output must match token_size. "
                f"Got {token_height}x{token_width}, expected "
                f"{token_size}x{token_size}."
            )

        self.token_height = token_height
        self.token_width = token_width
        self.fssca = FSSCA(
            channels=embed_dim,
            height=token_height,
            width=token_width,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, token_height * token_width, embed_dim)
        )
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        spectral_feature = self.spectral_encoder(x)

        spatial_feature = checkpoint.checkpoint(
            self.horizontal_projection,
            x,
            use_reentrant=False,
        )
        spatial_feature = checkpoint.checkpoint(
            self.vertical_projection,
            spatial_feature,
            use_reentrant=False,
        )
        spatial_feature = checkpoint.checkpoint(
            self.fssca,
            spatial_feature,
            use_reentrant=False,
        )

        fused_feature = spatial_feature + spectral_feature
        batch_size, channels, height, width = fused_feature.shape

        tokens = fused_feature.permute(0, 2, 3, 1).reshape(
            batch_size,
            height * width,
            channels,
        )
        tokens = tokens + self.position_embedding

        class_token = self.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        return tokens, height, width


class TokenMerge(nn.Module):
    """Similarity-guided local token recalibration and projection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.channel_scale = nn.Parameter(
            torch.zeros(1, in_dim, 1, 1)
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )
        self.class_projection = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity()
        )

    def _similarity_calibration(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        pad_height = height % 2
        pad_width = width % 2
        padded_height = height + pad_height
        padded_width = width + pad_width

        if pad_height or pad_width:
            x = F.pad(
                x,
                (0, pad_width, 0, pad_height),
                mode="reflect",
            )

        mask = x.new_ones(
            batch_size,
            1,
            padded_height,
            padded_width,
        )
        if pad_height:
            mask[:, :, height:, :] = 0
        if pad_width:
            mask[:, :, :, width:] = 0
        x = x * mask

        groups = x.reshape(
            batch_size,
            channels,
            padded_height // 2,
            2,
            padded_width // 2,
            2,
        )
        groups = groups.permute(0, 2, 4, 3, 5, 1).reshape(
            batch_size,
            -1,
            4,
            channels,
        )

        normalized_groups = F.normalize(groups, dim=-1)
        attention = normalized_groups @ normalized_groups.transpose(-1, -2)
        attention = attention.softmax(dim=-1)
        calibrated_groups = attention @ groups

        calibrated = calibrated_groups.reshape(
            batch_size,
            padded_height // 2,
            padded_width // 2,
            2,
            2,
            channels,
        )
        calibrated = calibrated.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            channels,
            padded_height,
            padded_width,
        )

        if pad_height or pad_width:
            calibrated = calibrated[:, :, :height, :width]

        return self.channel_scale * calibrated

    def forward(
        self,
        x: Tensor,
        height: int,
        width: int,
    ) -> Tuple[Tensor, int, int]:
        batch_size, _, channels = x.shape
        class_token = x[:, :1]
        spatial_tokens = x[:, 1:]

        feature_map = spatial_tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        feature_map = feature_map + self._similarity_calibration(
            feature_map
        )

        merged = checkpoint.checkpoint(
            self.projection,
            feature_map,
            use_reentrant=False,
        )
        new_height, new_width = merged.shape[-2:]
        merged_tokens = merged.flatten(2).transpose(1, 2)

        class_token = self.class_projection(class_token)
        output = torch.cat((class_token, merged_tokens), dim=1)
        return output, new_height, new_width


class ESAFormer(nn.Module):
    """Three-stage ESA-Former for multispectral image classification."""

    def __init__(
        self,
        image_size: int = 200,
        in_channels: int = 5,
        num_classes: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        representation_size: Optional[int] = None,
        drop_rate: float = 0.0,
        attention_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.tokenizer = SpectralSpatialTokenizer(
            image_size=image_size,
            in_channels=in_channels,
            embed_dim=192,
            kernel_size=9,
            stride=8,
            padding=0,
            token_size=24,
        )

        self.stage1 = TransformerBlock(
            dim=192,
            num_heads=4,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attention_drop_rate=attention_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.merge1 = TokenMerge(
            in_dim=192,
            out_dim=384,
            kernel_size=2,
            stride=2,
        )

        self.stage2 = TransformerBlock(
            dim=384,
            num_heads=8,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attention_drop_rate=attention_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.merge2 = TokenMerge(
            in_dim=384,
            out_dim=768,
            kernel_size=2,
            stride=1,
        )

        self.stage3 = TransformerBlock(
            dim=768,
            num_heads=12,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attention_drop_rate=attention_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.norm = norm_layer(768)

        if representation_size is not None:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "fc",
                            nn.Linear(768, representation_size),
                        ),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.num_features = 768
            self.pre_logits = nn.Identity()

        self.classifier = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(_initialize_weights)

    def forward_features(self, x: Tensor) -> Tensor:
        x, height, width = self.tokenizer(x)

        x = self.stage1(x)
        x, height, width = self.merge1(x, height, width)

        x = self.stage2(x)
        x, height, width = self.merge2(x, height, width)

        x = self.stage3(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def esa_former(
    num_classes: int = 4,
    in_channels: int = 5,
    representation_size: Optional[int] = None,
    **kwargs,
) -> ESAFormer:
    """Build the ESA-Former configuration used in the main experiments."""
    return ESAFormer(
        image_size=200,
        in_channels=in_channels,
        num_classes=num_classes,
        representation_size=representation_size,
        **kwargs,
    )


__all__ = [
    "ESAFormer",
    "FSSCA",
    "SpectralSpatialTokenizer",
    "TokenMerge",
    "esa_former",
]
