import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed
)
import torch
from contextlib import nullcontext
import trimesh
import gradio as gr
from gradio_imageslider import ImageSlider
from da2.utils.base import load_config
from da2.utils.model import load_model
from da2.utils.io import (
    read_cv2_image,
    torch_transform,
    tensorize
)
from da2.utils.vis import colorize_distance
from da2.utils.d2pc import distance2pointcloud
from datetime import (
    timedelta,
    datetime
)
import cv2
import numpy as np

last_glb_path = None

def prepare_to_run_demo():
    config = load_config('configs/infer.json')
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=config['accelerator']['timeout']))
    output_dir = f'output/infer'
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
    return config, accelerator

def read_mask_demo(mask_path, shape):
    if mask_path is None:
        return np.ones(shape[1:]) > 0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask

def load_infer_data_demo(image, mask, model_dtype, device):
    cv2_image = read_cv2_image(image)
    image = torch_transform(cv2_image)
    mask = read_mask_demo(mask, image.shape)
    image = tensorize(image, model_dtype, device)
    return image, cv2_image, mask

def ply2glb(ply_path, glb_path):
    pcd = trimesh.load(ply_path)
    points = np.asarray(pcd.vertices)
    colors = np.asarray(pcd.visual.vertex_colors)
    cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    cloud.export(glb_path)
    os.remove(ply_path)

def fn(image_path, mask_path):
    global last_glb_path
    config, accelerator = prepare_to_run_demo()
    model = load_model(config, accelerator)
    image, cv2_image, mask = load_infer_data_demo(image_path, mask_path, 
        model_dtype=config['spherevit']['dtype'], device=accelerator.device)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx, torch.no_grad():
        distance = model(image).cpu().numpy()[0]
        if last_glb_path is not None:
            os.remove(last_glb_path)
        distance_vis = colorize_distance(distance, mask)
        save_path = f'cache/tmp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.glb'
        last_glb_path = save_path
        normal_image = distance2pointcloud(distance, cv2_image, mask, save_path=save_path.replace('.glb', '.ply'), return_normal=True, save_distance=False)
        ply2glb(save_path.replace('.glb', '.ply'), save_path)
        return save_path, [distance_vis, normal_image]

inputs = [
    gr.Image(label="Input Image", type="filepath"),
    gr.Image(label="Input Mask", type="filepath"),
]
outputs = [
    gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Point Cloud"),
    gr.ImageSlider(
        label="Output Depth / Normal (transformed from the depth)",
        type="pil",
        slider_position=75,
    )
]

demo = gr.Interface(
    fn=fn,
    title="DA<sup>2</sup>: <u>D</u>epth <u>A</u>nything in <u>A</u>ny <u>D</u>irection",
    description="""
        <p align="center">
        <a title="Project Page" href="https://depth-any-in-any-dir.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white">
        </a>
        <a title="arXiv" href="http://arxiv.org/abs/2509.26618" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white">
        </a>
        <a title="Github" href="https://github.com/EnVision-Research/DA-2" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://img.shields.io/github/stars/EnVision-Research/DA-2?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
        </a>
        <a title="Social" href="https://x.com/_akhaliq/status/1973283687652606411" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
        </a>
        <a title="Social" href="https://x.com/haodongli00/status/1973287870317338747" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
        </a>
        <br>
        <strong>Please consider starring <span style="color: orange">&#9733;</span> our <a href="https://github.com/EnVision-Research/DA-2" target="_blank" rel="noopener noreferrer">GitHub Repo</a> if you find this demo useful!</strong>
        </p>
        <p><strong>Note: the "Input Mask" is optional, all pixels are assumed to be valid if mask is None.</strong></p>
    """,
    inputs=inputs,
    outputs=outputs,
    examples=[
        [os.path.join(os.path.dirname(__file__), "assets/demos/a1.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a2.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a3.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a4.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/b0.png"), 
         os.path.join(os.path.dirname(__file__), "assets/masks/b0.png")],
        [os.path.join(os.path.dirname(__file__), "assets/demos/b1.png"), 
         os.path.join(os.path.dirname(__file__), "assets/masks/b1.png")],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a5.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a6.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a7.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a8.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/b2.png"), 
         os.path.join(os.path.dirname(__file__), "assets/masks/b2.png")],
        [os.path.join(os.path.dirname(__file__), "assets/demos/b3.png"), 
         os.path.join(os.path.dirname(__file__), "assets/masks/b3.png")],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a9.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a10.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a11.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/a0.png"), None],
        [os.path.join(os.path.dirname(__file__), "assets/demos/b4.png"), 
         os.path.join(os.path.dirname(__file__), "assets/masks/b4.png")],
        [os.path.join(os.path.dirname(__file__), "assets/demos/b5.png"), 
         os.path.join(os.path.dirname(__file__), "assets/masks/b5.png")],
    ],
    examples_per_page=20
)

demo.launch(
        server_name="0.0.0.0",
        server_port=6381,
)
