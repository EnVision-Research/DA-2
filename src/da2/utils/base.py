import json
import argparse
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    InitProcessGroupKwargs, 
    ProjectConfiguration, 
    set_seed
)
import logging
from datetime import (
    timedelta,
    datetime
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    return args

def prepare_to_run():
    args = arg_parser()
    logging.basicConfig(
        format='%(asctime)s --> %(message)s',
        datefmt='%m/%d %H:%M:%S',
        level=logging.INFO,
    )
    config = load_config(args.config_path)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=config['accelerator']['timeout']))
    version = os.path.basename(args.config_path)[:-5]
    output_dir = f'output/{version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    accu_steps = config['accelerator']['accumulation_nsteps']
    accelerator = Accelerator(
        gradient_accumulation_steps=accu_steps,
        mixed_precision=config['accelerator']['mixed_precision'],
        log_with=config['accelerator']['report_to'],
        project_config=ProjectConfiguration(project_dir=output_dir),
        kwargs_handlers=[kwargs]
    )
    logger = get_logger(__name__, log_level='INFO')
    config['env']['logger'] = logger
    set_seed(config['env']['seed'])
    if config['env']['verbose']: 
        logger.info(f'Version: {version} (from {args.config_path})')
        logger.info(f'Output dir: {output_dir}')
        logger.info(f'Using {accelerator.num_processes} GPU' + ('s' if accelerator.num_processes > 1 else ''))
    return config, accelerator, output_dir
