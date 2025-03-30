'''https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0?utm_source=owkin&utm_medium=referral&utm_campaign=h-bioptimus-o
import functools

import timm
import torch
from torchvision import transforms 


PATH_TO_CHECKPOINT = ""  # Path to the downloaded checkpoint.

params = {
    'patch_size': 14, 
    'embed_dim': 1536, 
    'depth': 40, 
    'num_heads': 24, 
    'init_values': 1e-05, 
    'mlp_ratio': 5.33334, 
    'mlp_layer': functools.partial(
        timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
    ), 
    'act_layer': torch.nn.modules.activation.SiLU, 
    'reg_tokens': 4, 
    'no_embed_class': True, 
    'img_size': 224, 
    'num_classes': 0, 
    'in_chans': 3
}

model = timm.models.VisionTransformer(**params)
model.load_state_dict(torch.load(PATH_TO_CHECKPOINT, map_location="cpu"))
model.eval()
model.to("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.707223, 0.578729, 0.703617), 
        std=(0.211883, 0.230117, 0.177517)
    ),
])

input = torch.rand(3, 224, 224)
input = transforms.ToPILImage()(input)

# We recommend using mixed precision for faster inference.
with torch.autocast(device_type="cuda", dtype=torch.float16):
    with torch.inference_mode():
        features = model(transform(input).unsqueeze(0).to("cuda"))

assert features.shape == (1, 1536)
'''
import functools
import timm
import torch

def get_model(arch):
    if arch == 'h-optimus-0':
        params = {
            'patch_size': 14, 
            'embed_dim': 1536, 
            'depth': 40, 
            'num_heads': 24, 
            'init_values': 1e-05, 
            'mlp_ratio': 5.33334, 
            'mlp_layer': functools.partial(
                timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
            ), 
            'act_layer': torch.nn.modules.activation.SiLU, 
            'reg_tokens': 4, 
            'no_embed_class': True, 
            'img_size': 224, 
            'num_classes': 0, 
            'in_chans': 3
        }
        model = timm.models.VisionTransformer(**params)
        model.load_state_dict(torch.load('path/to/checkpoint.pth', map_location="cpu"))
    
    return model
