import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel

class phikon_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    
    def forward(self, x):
        outputs = self.vit(x)
        features = outputs.last_hidden_state[:, 0, :]
        return features

def get_model():
    return phikon_encoder()
