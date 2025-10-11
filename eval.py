import os
import torch
from contextlib import nullcontext
from tqdm import tqdm
from da2 import (
    prepare_to_run,
    load_model
)
from eval.utils import run_evaluation


def eval(model, config, accelerator, output_dir):
    model = model.eval()
    eval_datasets = config['evaluation']['datasets']
    if accelerator.is_main_process:
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)
        with autocast_ctx, torch.no_grad():
            for dataset_name in eval_datasets.keys():
                metrics = run_evaluation(model, config, dataset_name, output_dir, accelerator.device)
                for metric_name in config['evaluation']['metric_show']:
                    config['env']['logger'].info(f"\033[92mEVAL --> {dataset_name}: {config['evaluation']['metric_show'][metric_name]} = {metrics[metric_name]}.\033[0m")

if __name__ == '__main__':
    config, accelerator, output_dir = prepare_to_run()
    model = load_model(config, accelerator)
    eval(model, config, accelerator, output_dir)
