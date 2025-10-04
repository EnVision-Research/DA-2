import torch
from ..model.spherevit import SphereViT


def load_model(config, accelerator):
    model = SphereViT.from_pretrained('haodongli/DA-2', config=config)
    model = model.to(accelerator.device)
    torch.cuda.empty_cache()
    model = accelerator.prepare(model)
    if accelerator.num_processes > 1:
        model = model.module
    if config['env']['verbose']:
        config['env']['logger'].info(f'Model\'s dtype: {next(model.parameters()).dtype}.')
    config['spherevit']['dtype'] = next(model.parameters()).dtype
    return model
