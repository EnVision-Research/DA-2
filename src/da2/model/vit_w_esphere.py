import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .base import (
    fourier_dimension_expansion,
    flatten,
    DimensionAligner,
    AttentionSeq,
    ResidualUpsampler
)


class _ViT_w_Esphere(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        num_layers_head: int | list[int] = 4,
        dropout: float = 0.0,
        kernel_size: int = 7,
        layer_scale: float = 1.0,
        out_dim: int = 1,
        num_prompt_blocks: int = 1,
        use_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.up_sampler = nn.ModuleList([])
        self.pred_head = nn.ModuleList([])
        self.process_features = nn.ModuleList([])
        self.prompt_camera = nn.ModuleList([])
        mult = 2
        self.to_latents = nn.Linear(hidden_dim, hidden_dim)

        for _ in range(4):
            self.prompt_camera.append(
                AttentionSeq(
                    num_blocks=num_prompt_blocks,
                    dim=hidden_dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    layer_scale=-1.0,
                    context_dim=hidden_dim,
                )
            )

        for i, depth in enumerate(num_layers_head):
            current_dim = min(hidden_dim, mult * hidden_dim // int(2**i))
            next_dim = mult * hidden_dim // int(2 ** (i + 1))
            output_dim = max(next_dim, out_dim)
            self.process_features.append(
                nn.ConvTranspose2d(
                    hidden_dim,
                    current_dim,
                    kernel_size=max(1, 2 * i),
                    stride=max(1, 2 * i),
                    padding=0,
                )
            )
            self.up_sampler.append(
                ResidualUpsampler(
                    current_dim,
                    output_dim=output_dim,
                    expansion=expansion,
                    layer_scale=layer_scale,
                    kernel_size=kernel_size,
                    num_layers=depth,
                    use_norm=use_norm,
                )
            )
            pred_head = (
                nn.Sequential(nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim))
                if i == len(num_layers_head) - 1
                else nn.Identity()
            )
            self.pred_head.append(pred_head)

        self.to_depth_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode='reflect',
        )
        self.to_confidence_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode='reflect',
        )
        self.to_depth_hr = nn.Sequential(
            nn.Conv2d(
                output_dim // 2, 32, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.to_confidence_hr = nn.Sequential(
            nn.Conv2d(
                output_dim // 2, 32, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def set_original_shapes(self, shapes: tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes

    def embed_sphere_dirs(self, sphere_dirs):
        sphere_embedding = flatten(
            sphere_dirs, old=self.original_shapes, new=self.shapes
        )
        # index 0 -> Y
        # index 1 -> Z
        # index 2 -> X
        r1, r2, r3 = sphere_embedding[..., 0], sphere_embedding[..., 1], sphere_embedding[..., 2]
        polar = torch.asin(r2)
        r3_clipped = r3.abs().clip(min=1e-5) * (2 * (r3 >= 0).int() - 1)
        azimuth = torch.atan2(r1, r3_clipped)
        # [polar, azimuth] is the angle field
        sphere_embedding = torch.stack([polar, azimuth], dim=-1)
        # expand the dimension of the angle field to image feature dimensions, via sine-cosine basis embedding
        sphere_embedding = fourier_dimension_expansion(
            sphere_embedding,
            dim=self.hidden_dim,
            max_freq=max(self.shapes) // 2,
            use_cos=False,
        )
        return sphere_embedding

    def condition(self, feat, sphere_embeddings):
        conditioned_features = [
            prompter(rearrange(feature, 'b h w c -> b (h w) c'), sphere_embeddings)
            for prompter, feature in zip(self.prompt_camera, feat)
        ]
        return conditioned_features

    def process(self, features_list, sphere_embeddings):
        conditioned_features = self.condition(features_list, sphere_embeddings)
        init_latents = self.to_latents(conditioned_features[0])
        init_latents = rearrange(
            init_latents, 'b (h w) c -> b c h w', h=self.shapes[0], w=self.shapes[1]
        ).contiguous()
        conditioned_features = [
            rearrange(
                x, 'b (h w) c -> b c h w', h=self.shapes[0], w=self.shapes[1]
            ).contiguous()
            for x in conditioned_features
        ]
        latents = init_latents

        out_features = []
        # Pyramid-like multi-layer convolutional feature extraction
        for i, up in enumerate(self.up_sampler):
            latents = latents + self.process_features[i](conditioned_features[i + 1])
            latents = up(latents)
            out_features.append(latents)
        return out_features

    def prediction_head(self, out_features):
        depths = []
        h_out, w_out = out_features[-1].shape[-2:]
        for i, (layer, features) in enumerate(zip(self.pred_head, out_features)):
            out_depth_features = layer(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if i < len(self.pred_head) - 1:
                continue
            depths.append(out_depth_features)
        out_depth_features = F.interpolate(
            out_depth_features, size=(h_out, w_out), mode='bilinear', align_corners=True
        )
        distance = self.to_depth_lr(out_depth_features)
        distance = F.interpolate(
            distance, size=self.original_shapes, mode='bilinear', align_corners=True
        )
        distance = self.to_depth_hr(distance)
        return distance

    def forward(
        self,
        features: list[torch.Tensor],
        sphere_dirs: torch.Tensor
    ) -> torch.Tensor:
        sphere_embeddings = self.embed_sphere_dirs(sphere_dirs)
        features = self.process(features, sphere_embeddings)
        distance = self.prediction_head(features)
        return distance


class ViT_w_Esphere(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_aligner = DimensionAligner(
            input_dims=config['input_dims'],
            hidden_dim=config['hidden_dim'],
        )
        self._vit_w_esphere = _ViT_w_Esphere(**config)

    def forward(self, images, features, sphere_dirs) -> torch.Tensor:
        _, _, H, W = images.shape
        sphere_dirs = sphere_dirs
        common_shape = features[0].shape[1:3]
        features = self.dim_aligner(features)
        sphere_dirs = rearrange(sphere_dirs, 'b c h w -> b (h w) c')

        self._vit_w_esphere.set_shapes(common_shape)
        self._vit_w_esphere.set_original_shapes((H, W))
        logdistance = self._vit_w_esphere(
            features=features,
            sphere_dirs=sphere_dirs,
        )

        distance = torch.exp(logdistance.clip(min=-8.0, max=8.0) + 2.0)
        # distance = distance / torch.quantile(distance, 0.98)
        return distance
