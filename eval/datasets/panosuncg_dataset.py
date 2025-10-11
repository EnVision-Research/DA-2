# Author: Haodong Li
# Last modified: 2025-05-25

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
import cv2
import os

class PanoSUNCGDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=1e-3,
            max_depth=5,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        img_path = os.path.join(self.dataset_dir, rel_path)
        depth_in = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        depth_in = depth_in[..., 0]
        depth_decoded = depth_in / 20.0
        return depth_decoded
