import os
import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image


def torch_transform(image):
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image

def read_cv2_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_mask(mask_path, shape):
    if not os.path.exists(mask_path):
        return np.ones(shape[1:]) > 0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask

def tensorize(array, model_dtype, device):
    array = torch.from_numpy(array).to(device).to(model_dtype).unsqueeze(dim=0)
    return array

def load_infer_data(config, device):
    image_dir = config['inference']['images']
    mask_dir = config['inference']['masks']

    image_paths = glob(os.path.join(image_dir, '*.png'))
    image_paths = sorted(image_paths)
    filenames = [os.path.basename(image_path)[:-4] for image_path in image_paths]
    cv2_images = [read_cv2_image(image_path) 
        for image_path in image_paths]
    PIL_images = [Image.fromarray(cv2_image) for cv2_image in cv2_images]
    images = [torch_transform(cv2_image) for cv2_image in cv2_images]

    mask_paths = [image_path.replace(image_dir, mask_dir) 
        for image_path in image_paths]
    masks = [read_mask(mask_path, images[i].shape) 
        for (i, mask_path) in enumerate(mask_paths)]

    model_dtype = config['spherevit']['dtype']
    images = [tensorize(image, model_dtype, device) for image in images]

    infer_data = {
        'images': {
            'PIL': PIL_images,
            'cv2': cv2_images,
            'torch': images
        },
        'masks': masks,
        'filenames': filenames,
        'size': len(images)
    }
    if config['env']['verbose']:
        s = 's' if len(images) > 1 else ''
        config['env']['logger'].info(f'Loaded {len(images)} image{s} in {model_dtype}')
    return infer_data
