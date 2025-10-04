import torch
from PIL import Image
import numpy as np
import matplotlib
import cv2


def concatenate_images(*image_lists):
    max_width = 0
    total_height = 0
    row_widths = []
    row_heights = []

    for i, image_list in enumerate(image_lists):
        width = sum(img.width for img in image_list)
        max_width = max(max_width, width)
        row_widths.append(width)
        # Assuming all images in the list have the same height
        height = image_list[0].height
        total_height += height
        row_heights.append(height)

    new_image = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for i, image_list in enumerate(image_lists):
        x_offset = 0
        for img in image_list:
            new_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row_heights[i]
    return new_image

def colorize_distance(distance, mask, cmap='Spectral'):
    if distance.ndim >= 3: distance = distance.squeeze()
    cm = matplotlib.colormaps[cmap]
    valid_distance = distance[mask]
    max_distance = np.quantile(valid_distance, 0.98)
    min_distance = np.quantile(valid_distance, 0.02)
    distance[~mask] = max_distance
    distance = ((distance - min_distance) / (max_distance - min_distance))
    distance = np.clip(distance, 0, 1)
    img_colored_np = cm(distance, bytes=False)[:, :, 0:3]
    distance_colored = (img_colored_np * 255).astype(np.uint8) 
    return Image.fromarray(distance_colored)
