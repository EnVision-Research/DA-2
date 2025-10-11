# Authors: Bingxin Ke, Haodong Li
# Last modified: 2025-05-25
# Note: Add PanoSUNCGDataset, Matterport3DDataset, Stanford2D3DSDataset for 360° depth (or distance) evaluation.

import os

from .base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode
from .stanford2d3ds_dataset import Stanford2D3DSDataset
from .matterport3d_dataset import Matterport3DDataset
from .panosuncg_dataset import PanoSUNCGDataset

dataset_name_class_dict = {
    "2d3ds": Stanford2D3DSDataset,
    "matterport3d": Matterport3DDataset,
    "panosuncg": PanoSUNCGDataset
}


def get_dataset(
    cfg_data_split, dataset_name, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseDepthDataset:
    if dataset_name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[dataset_name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split['filenames'],
            dataset_dir=os.path.join(base_data_dir, cfg_data_split['dir']),
            disp_name=dataset_name,
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
