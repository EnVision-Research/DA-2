import os
import numpy as np
import utils3d
from plyfile import PlyData, PlyElement
from PIL import Image


def sphere_uv2dirs(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions

def save_3d_points(points: np.array, colors: np.array, mask: np.array, save_path: str):
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    mask = mask.reshape(-1, 1)

    vertex_data = np.empty(mask.sum(), dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
    ])
    vertex_data['x'] = [a for i, a in enumerate(points[:, 0]) if mask[i]]
    vertex_data['y'] = [a for i, a in enumerate(points[:, 1]) if mask[i]]
    vertex_data['z'] = [a for i, a in enumerate(points[:, 2]) if mask[i]]
    vertex_data['red'] = [a for i, a in enumerate(colors[:, 0]) if mask[i]]
    vertex_data['green'] = [a for i, a in enumerate(colors[:, 1]) if mask[i]]
    vertex_data['blue'] = [a for i, a in enumerate(colors[:, 2]) if mask[i]]

    vertex_element = PlyElement.describe(vertex_data, 'vertex', comments=['vertices with color'])
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    PlyData([vertex_element], text=True).write(save_path)

def colorize_normal(normal: np.ndarray, normal_mask: np.ndarray):
    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    normal_mask = np.repeat(np.expand_dims(normal_mask, axis=-1), 3, axis=-1)
    normal_mask = normal_mask.astype(np.uint8)
    normal_rgb = normal_rgb * normal_mask
    return normal_rgb

def normal_normalize(normal: np.ndarray):
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-6] = 1e-6
    return normal / normal_norm

def distance2pointcloud(
    distance: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    save_path: str = None,
    return_normal: bool = False,
    save_distance: bool = False
):
    if distance.ndim >= 3: distance = distance.squeeze()
    if save_distance:
        save_path_dis = save_path.replace('3dpc', 'depth').replace('.ply', '.npy')
        save_dir_dis = os.path.dirname(save_path_dis)
        if not os.path.exists(save_dir_dis): os.makedirs(save_dir_dis, exist_ok=True)
        np.save(save_path_dis, distance)
    height, width = distance.shape[:2]
    points = distance[:, :, None] * sphere_uv2dirs(utils3d.numpy.image_uv(width=width, height=height))
    save_3d_points(points, image, mask, save_path)
    if return_normal:
        normal, normal_mask = utils3d.numpy.points_to_normals(points, mask)
        normal = normal * np.array([-1, -1, 1])
        normal = normal_normalize(normal)
        normal_1 = normal[..., 0]
        normal_2 = normal[..., 1]
        normal_3 = normal[..., 2]
        normal = np.stack([normal_1, normal_3, normal_2], axis=-1)
        normal_img = colorize_normal(normal, normal_mask)
        return Image.fromarray(normal_img)
