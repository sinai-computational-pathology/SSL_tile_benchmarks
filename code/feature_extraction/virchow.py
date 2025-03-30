import os
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
from huggingface_hub import login, hf_hub_download

class virchow(nn.Module):
    def __init__(self):
        super(virchow, self).__init__()
        self.virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    
    def forward(self, x):
        output = self.virchow(x)  # size: 1 x 257 x 1280
        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280
        # concatenate class token and average pool of patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding
