import torch


def get_uv_gird(h, w, device):
    pixel_coords_x = torch.linspace(0.5, w - 0.5, w, device=device)
    pixel_coords_y = torch.linspace(0.5, h - 0.5, h, device=device)
    stacks = [pixel_coords_x.repeat(h, 1), pixel_coords_y.repeat(w, 1).t()]
    grid = torch.stack(stacks, dim=0).float() 
    grid = grid.to(device).unsqueeze(0)
    return grid


class Sphere():
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def get_directions(self, shape):
        h, w = shape
        uv = get_uv_gird(h, w, device=self.device)
        u, v = uv.unbind(dim=1)
        width, height = self.config['width'], self.config['height']
        hfov, vfov = self.config['hfov'], self.config['vfov']
        longitude = (u - width / 2) / width * hfov
        latitude = (v - height / 2) / height * vfov
        x = torch.cos(latitude) * torch.sin(longitude)
        z = torch.cos(latitude) * torch.cos(longitude)
        y = torch.sin(latitude)
        sphere_directions = torch.stack([x, y, z], dim=1)
        return sphere_directions
