import os
import torch
from contextlib import nullcontext
from tqdm import tqdm
from da2 import (
    prepare_to_run,
    load_model,
    load_infer_data,
    distance2pointcloud,
    colorize_distance,
    concatenate_images
)


def infer(model, config, accelerator, output_dir):
    model.eval()
    if accelerator.is_main_process:
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)
        with autocast_ctx, torch.no_grad():
            infer_data = load_infer_data(config, accelerator.device)
            pred_distances = []
            for i in tqdm(range(infer_data['size']), desc='Predicting 360° depth'):
                distances = model(infer_data['images']['torch'][i])
                pred_distances.append(distances.cpu().numpy())
            pred_distance_images = []
            pred_normal_images = []
            for i in tqdm(range(infer_data['size']), desc='Visualizing 360° depth'):
                pred_distance_images.append(colorize_distance(pred_distances[i], infer_data['masks'][i]))
            for i in tqdm(range(infer_data['size']), desc='Saving 3D points'):
                normal_image = distance2pointcloud(pred_distances[i], 
                    infer_data['images']['cv2'][i], infer_data['masks'][i], 
                    save_path=os.path.join(output_dir, f'3dpc/{infer_data['filenames'][i]}.ply'), return_normal=True, save_distance=True)
                pred_normal_images.append(normal_image)
            concatenate_images(infer_data['images']['PIL'], pred_distance_images, pred_normal_images).save(os.path.join(output_dir, 'vis_all.png'))

if __name__ == '__main__':
    config, accelerator, output_dir = prepare_to_run()
    model = load_model(config, accelerator)
    infer(model, config, accelerator, output_dir)
