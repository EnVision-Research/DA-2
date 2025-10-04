import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from math import (
    ceil, 
    sqrt
)
from huggingface_hub import PyTorchModelHubMixin
import torchvision.transforms.v2.functional as TF
from .dinov2 import DINOViT
from .vit_w_esphere import ViT_w_Esphere
from .sphere import Sphere


IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

class SphereViT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dino = DINOViT()
        self.vit_w_esphere = ViT_w_Esphere(config['spherevit']['vit_w_esphere'])
        feature_slices = self.dino.output_idx
        self.feature_slices = list(
            zip([0, *feature_slices[:-1]], feature_slices)
        )
        self.device = None

    def to(self, *args):
        self.device = args[0]
        return super().to(*args)

    def forward(self, images):
        B, _, H, W = images.shape
        current_pixels = H * W
        target_pixels = min(self.config['inference']['max_pixels'], 
            max(self.config['inference']['min_pixels'], current_pixels))
        factor = sqrt(target_pixels / current_pixels)
        sphere_config = deepcopy(self.config['spherevit']['sphere'])
        sphere_config['width'] *= factor
        sphere_config['height'] *= factor
        sphere = Sphere(config=sphere_config, device=self.device)
        H_new = int(H * factor)
        W_new = int(W * factor)
        DINO_patch_size = 14 # please see the line 51 of `src/da2/model/dinov2/dinovit.py` (I know it's a little ugly to hardcode it here T_T)
        H_new = ceil(H_new / DINO_patch_size) * DINO_patch_size
        W_new = ceil(W_new / DINO_patch_size) * DINO_patch_size
        images = F.interpolate(images, size=(H_new, W_new), mode='bilinear', align_corners=False)
        images = TF.normalize(
            images.float(),
            mean=IMAGENET_DATASET_MEAN,
            std=IMAGENET_DATASET_STD,
        )

        sphere_dirs = sphere.get_directions(shape=(H_new, W_new))
        sphere_dirs = sphere_dirs.to(self.device)
        sphere_dirs = sphere_dirs.repeat(B, 1, 1, 1)

        features = self.dino(images)
        features = [
            features[i:j][-1].contiguous()
            for i, j in self.feature_slices
        ]
        distance = self.vit_w_esphere(images, features, sphere_dirs)
        distance = F.interpolate(distance, size=(H, W), mode='bilinear', align_corners=False)
        distance = distance.squeeze(dim=1) # (b, 1, h, w) -> (b, h, w)
        return distance
