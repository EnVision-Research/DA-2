# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import math
import torch
import torch.nn as nn
import contextlib
from functools import partial
from typing import Sequence
from .block import (
    Block
)
from .attention import (
    MemEffAttention
)
from .mlp import (
    Mlp
)
from .patch_embed import (
    PatchEmbed
)
from .swiglu_ffn import (
    SwiGLUFFNFused
)

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import fmha, memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class DINOViT(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=partial(Block, attn_class=MemEffAttention),
        ffn_layer="mlp",
        block_chunks=0,
        output_idx=[6, 12, 18, 24],
        num_register_tokens=0,
        interpolate_antialias=False,
        use_norm=True,
        frozen_stages=0,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        self.frozen_stages = frozen_stages
        self.patch_size = patch_size
        self.output_idx = output_idx
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(
            torch.zeros(1, max(1, num_register_tokens), embed_dim)
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f():
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = nn.LayerNorm(embed_dim)
        self.use_norm = use_norm
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def interpolate_pos_encoding(self, x, W, H):
        previous_dtype = x.dtype
        N = self.pos_embed.shape[1] - 1
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = W // self.patch_size
        h0 = H // self.patch_size

        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        kwargs["size"] = (w0, h0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def tokenize(self, x):
        _, _, W, H = x.shape
        with torch.no_grad() if self.frozen_stages > -1 else contextlib.nullcontext():
            x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        dino_pos_embed = self.interpolate_pos_encoding(x, W, H)
        x = x + dino_pos_embed
        return x

    def forward_features(self, x):
        shapes = [val // self.patch_size for val in x.shape[-2:]]
        batch_size = x.shape[0]
        features = []
        x = self.tokenize(x)
        for i, blk in enumerate(self.blocks):
            with (
                torch.no_grad() if i < self.frozen_stages else contextlib.nullcontext()
            ):
                x = blk(x)
            features.append(x)
        if self.use_norm:
            with (
                torch.no_grad()
                if self.frozen_stages >= len(self.blocks)
                else contextlib.nullcontext()
            ):
                features = [self.norm(out) for out in features]
        features = [out[:, self.num_register_tokens + 1 :] for out in features]
        features = [out.reshape(batch_size, *shapes, -1) for out in features]
        return features

    def forward(self, *args):
        features = self.forward_features(*args)
        return features
