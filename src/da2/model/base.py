import torch
import torch.nn as nn
from math import log2, pi
from typing import Tuple
import torch.nn.functional as F
from einops import rearrange
from functools import partial


def fourier_dimension_expansion(
    x: torch.Tensor,
    dim: int = 512,
    max_freq: int = 64,
    use_cos: bool = True,
    use_log: bool = True,
):
    device, dtype, input_dim = x.device, x.dtype, x.shape[-1]
    # input_dim: 2
    num_bands = dim // (2 * input_dim) if use_cos else dim // input_dim
    # num_bands = 512 // 2 = 256
    if use_log:
        scales = 2.0 ** torch.linspace(
            0.0, log2(max_freq), steps=num_bands, device=device, dtype=dtype
        )
    else:
        scales = torch.linspace(
            1.0, max_freq / 2, num_bands, device=device, dtype=dtype
        )
    x = x.unsqueeze(-1)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * pi
    x = torch.cat(
        (
            [x.sin(), x.cos()]
            if use_cos
            else [
                x.sin(),
            ]
        ),
        dim=-1,
    )
    x = x.flatten(-2)
    return x

def flatten(
    flat_tensor: torch.Tensor,
    old: Tuple[int, int],
    new: Tuple[int, int],
) -> torch.Tensor:
    if old[0] == new[0] and old[1] == new[1]:
        return flat_tensor
    tensor = flat_tensor.view(flat_tensor.shape[0], old[0], old[1], -1).permute(
        0, 3, 1, 2
    )  # b c h w
    tensor_interp = F.interpolate(
        tensor,
        size=(new[0], new[1]),
        mode='nearest',
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, new[0] * new[1]
    ).permute(
        0, 2, 1
    )  # b (h w) c
    return flat_tensor_interp.contiguous()


class DimensionAligner(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int):
        super().__init__()
        self.aligners = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        self.checkpoint = True
        for input_dim in input_dims:
            self.aligners.append(nn.Linear(input_dim, hidden_dim))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        outs = [self.aligners[i](x) for i, x in enumerate(xs)]
        return outs


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = default(output_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        detach_query: bool = False,
        residual_ls: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = dim if context_dim is None else context_dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=False)
        self.ls1_1 = (
            LayerScale(dim, layer_scale)
            if layer_scale > 0.0 and not residual_ls
            else nn.Identity()
        )
        self.ls1_2 = (
            LayerScale(dim, layer_scale)
            if layer_scale > 0.0 and residual_ls
            else nn.Identity()
        )
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.detach_query = detach_query

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.detach_query:
            x = x.detach()
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), 'b n (kv h d) -> b h n d kv', h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), 'b n (h d) -> b h n d', h=self.num_heads)

        if rope is not None:
            q = rope(q.permute(0, 2, 1, 3), input_pos=rope_pos).permute(0, 2, 1, 3)
            k = rope(k.permute(0, 2, 1, 3), input_pos=rope_pos).permute(0, 2, 1, 3)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, 'b n (h d) -> b h n d', h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, 'b n (h d) -> b h n d', h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.out(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = self.ls1_1(
            self.attn(
                x,
                rope=rope,
                rope_pos=rope_pos,
                attn_bias=attn_bias,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
            )
        ) + self.ls1_2(x)
        x = self.ls2(self.mlp(x)) + x
        return x


class AttentionSeq(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        detach_query: bool = False,
        residual_ls: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    cosine=cosine,
                    gated=gated,
                    layer_scale=layer_scale,
                    context_dim=context_dim,
                    detach_query=detach_query,
                    residual_ls=residual_ls,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
                attn_bias=attn_bias,
                rope=rope,
                rope_pos=rope_pos,
            )
        return x


class ResidualConvNet(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size: int = 3,
        padding_mode: str = 'zeros',
        dilation: int = 1,
        layer_scale: float = 1.0,
        use_norm: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.activation = nn.LeakyReLU()
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(1, dim, 1, 1))
            if layer_scale > 0.0
            else 1.0
        )
        self.norm1 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.gamma * out + x


class ResidualUpsampler(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim: int = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        layer_scale: float = 1.0,
        padding_mode: str = 'zeros',
        use_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else hidden_dim // 2
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                ResidualConvNet(
                    hidden_dim,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                )
            )
        self.up = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                padding_mode=padding_mode,
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x
