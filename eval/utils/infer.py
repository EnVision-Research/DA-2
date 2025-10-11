# Authors: Jing He, Haodong Li
# Last modified: 2025-10-10

import os
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
from eval.datasets import (
    BaseDepthDataset,
    get_dataset,
    DatasetMode,
    get_pred_name
)
from eval.utils import (
    MetricTracker,
    metric,
    init_per_sample_csv,
    write_per_sample_csv,
    write_metrics_txt, 
    align_depth_least_square, 
    align_depth_median
)


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def run_evaluation(model, config, dataset_name, output_dir, device):
    eval_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    dataset_config = config['evaluation']['datasets'][dataset_name]
    model_dtype = config['spherevit']['dtype']

    dataset: BaseDepthDataset = get_dataset(dataset_config, dataset_name, 
        base_data_dir=config['evaluation']['datasets_dir'], mode=DatasetMode.EVAL)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=True)

    metric_funcs = [getattr(metric, _met) for _met in config['evaluation']['metric_names']]
    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()
    alignment = config['evaluation']['alignment']
    per_sample_csv = init_per_sample_csv(eval_dir, alignment, metric_funcs)

    for data in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        rgb_tmp = data["rgb_int"].squeeze()
        if rgb_tmp.min() == 0:
            rgb_tmp = rgb_tmp.float()
            rgb_tmp = torch.mean(rgb_tmp, dim=0).cpu().numpy()
            zero_mask = rgb_tmp != 0
            valid_mask = valid_mask & zero_mask

        depth_raw_ts = depth_raw_ts.to(device)
        valid_mask_ts = valid_mask_ts.to(device)

        # Get predictions
        rgb_basename = os.path.basename(rgb_name)
        pred_basename = get_pred_name(
            rgb_basename, dataset.name_mode, suffix=""
        )
        pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)
        # resize to processing_res
        input_size = data["rgb_int"].shape
        input_rgb = resize_max_res(
            data["rgb_int"],
            max_edge_resolution=1092,
        )

        input_rgb = input_rgb[0].to(device)
        input_rgb = input_rgb / 255.0
        input_rgb = input_rgb.to(model_dtype)

        depth_pred = model(input_rgb.unsqueeze(0))
        depth_pred = depth_pred.unsqueeze(0)
        depth_pred = resize(depth_pred, input_size[-2:], antialias=True)
        depth_pred = depth_pred.squeeze().cpu().numpy()

        if "least_square" == alignment:
            depth_pred = np.clip(
                depth_pred, a_min=-1e6, a_max=1e6
            )
            depth_pred, _, _ = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=dataset_config['alignment_max_res'],
            )
        elif "median" == alignment:
            depth_pred = np.clip(
                depth_pred, a_min=-1e6, a_max=1e6
            )
            depth_pred, _, _ = align_depth_median(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
            )
        elif "metric" == alignment:
            pass
        else:
            raise NotImplementedError

        # Clip to dataset min max
        depth_pred = np.clip(
            depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        # Evaluate (using CUDA if available)
        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        write_per_sample_csv(per_sample_csv, pred_name, sample_metric)

    # -------------------- Save metrics to file --------------------
    eval_text = tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )
    write_metrics_txt(eval_dir, alignment, eval_text)

    return metric_tracker.result()
