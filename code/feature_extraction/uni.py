import os
import torch
import timm

def get_model():
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load('path/to/checkpoint.pth', map_location="cpu"), strict=True)
    return model
