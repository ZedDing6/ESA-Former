from functools import partial
from collections import OrderedDict

import torch
import torch.utils.checkpoint as cp
import torch.nn as nn
import torch.nn.functional as F
import torch_dct


class SpectralConv1D(nn.Module):
    def __init__(self, in_bands=5, out_channels=32, spatial_size=24):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.spatial_pool(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * H * W, C)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.mean(dim=-1)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        return x


class CoordConditionedFrequencyResponse(nn.Module):
    def __init__(self, dct_h: int, dct_w: int, desc_dim: int = 1, hidden_dim: int = 16):
        super().__init__()
        coord = self.build_frequency_coord(dct_h, dct_w)
        self.register_buffer("coord", coord, persistent=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.desc_mlp = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def build_frequency_coord(H: int, W: int) -> torch.Tensor:
        u = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
        v = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
        u_norm = u / max(H - 1, 1)
        v_norm = v / max(W - 1, 1)
        r = torch.sqrt(u_norm ** 2 + v_norm ** 2)
        return r.unsqueeze(-1)

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        B, C, D = desc.shape
        assert D == 1, f"Expected desc_dim=1, but got {D}"
        pos_feat = self.pos_mlp(self.coord)
        desc_feat = self.desc_mlp(desc)
        fused = pos_feat.unsqueeze(0).unsqueeze(0) + desc_feat.unsqueeze(2).unsqueeze(3)
        response = self.head(fused).squeeze(-1)
        response = 0.5 + torch.sigmoid(response)
        return response


class FSSCA(nn.Module):
    def __init__(self, in_channels: int, dct_h: int = 200, dct_w: int = 200, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.channel_fc = nn.Linear(in_channels, in_channels)
        self.freq_response = CoordConditionedFrequencyResponse(
            dct_h=dct_h,
            dct_w=dct_w,
            desc_dim=1,
            hidden_dim=16
        )
        self.sa_norm = nn.GroupNorm(in_channels, in_channels)
        self.smooth = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self._init_smooth()

    def _init_smooth(self):
        with torch.no_grad():
            self.smooth.weight.fill_(1.0 / 9.0)

    @staticmethod
    def dct_2d(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.reshape(B * C, H, W)
        x = torch_dct.dct_2d(x, norm='ortho')
        return x.reshape(B, C, H, W)

    @staticmethod
    def idct_2d(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.reshape(B * C, H, W)
        x = torch_dct.idct_2d(x, norm='ortho')
        return x.reshape(B, C, H, W)

        def channel_complexity(self, x_freq: torch.Tensor) -> torch.Tensor:
        """
        高频能量 / 总能量
        DCT 中左上角为低频，右下角为高频

        x_freq: [B, C, H, W]
        return: [B, C]
        """
        B, C, H, W = x_freq.shape

        high = x_freq[:, :, 3 * H // 4:, 3 * W // 4:].abs().sum(dim=(2, 3))
        total = x_freq.abs().sum(dim=(2, 3)) + self.eps

        return high / total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _freq_block(x_inner: torch.Tensor) -> torch.Tensor:
            x_freq = self.dct_2d(x_inner)
            complexity = self.channel_complexity(x_freq)
            ca = torch.sigmoid(self.channel_fc(complexity))
            desc = complexity.unsqueeze(-1)
            response = self.freq_response(desc)
            x_freq_enh = x_freq * response
            x_spatial = self.idct_2d(x_freq_enh)
            x_spatial = self.sa_norm(x_spatial)
            x_spatial = x_spatial * ca.unsqueeze(-1).unsqueeze(-1)
            sa = torch.sigmoid(self.smooth(x_spatial))
            x_out = x_inner + x_inner * sa
            return x_out
        return cp.checkpoint(_freq_block, x, use_reentrant=False)


class SpecSpaTokenizer(nn.Module):
    def __init__(self, img_size=200, in_c=4, embed_dim=768):
        super().__init__()
        self.kernel_h = self.kernel_w = 9
        self.stride_h = self.stride_w = 8
        self.padding_h = self.padding_w = 0
        self.spectral_encoder = SpectralConv1D(
            in_bands=in_c,
            out_channels=embed_dim
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=(self.kernel_h, 1), stride=(self.stride_h, 1),
                      padding=(self.padding_h, 0)),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, self.kernel_w), stride=(1, self.stride_w),
                      padding=(0, self.padding_w)),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.token_h = (img_size + self.padding_h * self.padding_h - self.kernel_h) // self.stride_h + 1
        self.token_w = (img_size + self.padding_w * self.padding_w - self.kernel_w) // self.stride_w + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_h * self.token_w, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = FSSCA(embed_dim, dct_h=self.token_h, dct_w=self.token_w)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, Spectral_x):
        x_spectral = self.spectral_encoder(Spectral_x)
        x_spatial = cp.checkpoint(self.conv_h, x, use_reentrant=False)
        x_spatial = cp.checkpoint(self.conv_v, x_spatial, use_reentrant=False)
        x_spatial = cp.checkpoint(self.attn, x_spatial, use_reentrant=False)
        x_spatial = x_spatial + x_spectral
        B, C2, H2, W2 = x_spatial.shape
        x_spatial_flat = x_spatial.permute(0, 2, 3, 1).reshape(B, H2 * W2, C2)
        x_proj = x_spatial_flat + self.pos_embed
        cls_token = self.cls_token.expand(B, -1, -1)
        x_final = torch.cat([cls_token, x_proj], dim=1)
        return x_final, H2, W2


class TokenMerge(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.scale = nn.Parameter(torch.zeros(1))
        self.padding = padding
        self.merge = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )
        self.cls_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def sim_calib(self, x):
        B, C, H, W = x.shape
        pad_h = 0
        pad_w = 0
        if H % 2 != 0:
            pad_h = 1
            H_pad = H + 1
        else:
            H_pad = H
        if W % 2 != 0:
            pad_w = 1
            W_pad = W + 1
        else:
            W_pad = W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        mask = torch.ones((B, 1, H_pad, W_pad), device=x.device)
        if pad_h > 0:
            mask[:, :, H:, :] = 0
        if pad_w > 0:
            mask[:, :, :, W:] = 0
        x = x * mask
        x_group = x.reshape(B, C, H_pad // 2, 2, W_pad // 2, 2)
        x_group = x_group.permute(0, 2, 4, 3, 5, 1)
        x_group = x_group.reshape(B, -1, 4, C)
        x_norm = F.normalize(x_group, dim=-1)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))
        attn = torch.softmax(sim, dim=-1)
        x_sim = torch.matmul(attn, x_group)
        x_sim = x_sim.reshape(B, H_pad // 2, W_pad // 2, 2, 2, C)
        x_sim = x_sim.permute(0, 5, 1, 3, 2, 4)
        x_sim = x_sim.reshape(B, C, H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            x_sim = x_sim[:, :, :H, :W]
        return self.scale * x_sim

    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, tokens = x[:, 0:1, :], x[:, 1:, :]
        x_feat = tokens.transpose(1, 2).reshape(B, C, H, W)
        x_feat = x_feat + self.sim_calib(x_feat)
        x_merged = cp.checkpoint(self.merge, x_feat, use_reentrant=False)
        H_new, W_new = x_merged.shape[2:]
        x_merged_flat = x_merged.flatten(2).transpose(1, 2)
        cls_token = self.cls_proj(cls_token)
        x_out = torch.cat([cls_token, x_merged_flat], dim=1)
        return x_out, H_new, W_new


class ESAFormer(nn.Module):
    def __init__(self, img_size=200, in_c=5, num_classes=1000, mlp_ratio=4.0,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
        super(ESAFormer, self).__init__()
        self.num_classes = num_classes
        self.num_features = 768

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.patch_embed = SpecSpaTokenizer(
            img_size=img_size, in_c=in_c, embed_dim=192
        )
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.stage1_blocks = nn.Sequential(*[
            Block(dim=192, num_heads=4, mlp_ratio=mlp_ratio,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=drop_path_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(1)
        ])
        self.stage1_merge = TokenMerge(in_dim=192, out_dim=384, kernel_size=2, stride=2)

        self.stage2_blocks = nn.Sequential(*[
            Block(dim=384, num_heads=8, mlp_ratio=mlp_ratio,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=drop_path_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(1)
        ])
        self.stage2_merge = TokenMerge(in_dim=384, out_dim=768, kernel_size=2, stride=1)

        self.stage3_blocks = nn.Sequential(*[
            Block(dim=768, num_heads=12, mlp_ratio=mlp_ratio,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=drop_path_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(1)
        ])

        self.norm = norm_layer(768)
        self.head = nn.Linear(768, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(_init_weights)

    def forward_features(self, x, Spectral_x):
        x, H, W = self.patch_embed(x, Spectral_x)
        x = self.stage1_blocks(x)
        x, H, W = self.stage1_merge(x, H, W)
        x = self.stage2_blocks(x)
        x, H, W = self.stage2_merge(x, H, W)
        x = self.stage3_blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, Spectral_x):
        x = self.forward_features(x, Spectral_x)
        x = self.head(x)
        return x


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def esaformer(num_classes: int = 1000):
    model = ESAFormer(
        img_size=200,
        in_c=5,
        num_classes=num_classes,
        mlp_ratio=4.0
    )
    return model


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output
