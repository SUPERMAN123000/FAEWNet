# Copyright (c) Open-CD. All rights reserved.
from typing import Tuple, Optional

import einops
import torch
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import FFN

from mmpretrain.models import build_norm_layer
from mmpretrain.models.backbones.vit_sam import Attention, window_partition, window_unpartition
from torch.nn.functional import embedding

from opencd.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

import torch.nn.init as init
import math

def get_rel_pos(q_size: int, k_size: int,
                rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):
    def __init__(
            self,
            embed_dims: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = head_embed_dims ** -0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (input_size is not None), \
                'Input size must be provided if using relative position embed.'
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_embed_dims))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_embed_dims))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, H, W, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


@MODELS.register_module()
class SMSFTimeFusionTransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        if self.window_size > 0:  # TODO: Maybe it should be ``self.window_size == 0`` here.
            in_channels = embed_dims * 2
            self.down_channel = torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            self.down_channel.weight.data.fill_(1.0 / in_channels)
            self.soft_ffn = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
            )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):

        shortcut = x
        x = self.ln1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)
        # time phase fusion
        if self.window_size > 0:  # TODO: Maybe it should be ``self.window_size == 0`` here.
            x = einops.rearrange(x, 'b h w d -> b d h w')  # 2B, C, H, W
            x0 = x[:x.size(0) // 2]
            x1 = x[x.size(0) // 2:]  # B, C, H, W
            x0_1 = torch.cat([x0, x1], dim=1)
            activate_map = self.down_channel(x0_1)
            activate_map = torch.sigmoid(activate_map)
            x0 = x0 + self.soft_ffn(x1 * activate_map)
            x1 = x1 + self.soft_ffn(x0 * activate_map)

            x = torch.cat([x0, x1], dim=0)
            x = einops.rearrange(x, 'b d h w -> b h w d')
        return x
