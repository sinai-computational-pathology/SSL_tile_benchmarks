import os
import numpy as np
import pandas as pd
import openslide
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image
import resnet_custom
import pdb
import encoders
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--slide_data', type=str, default='', help='path to slide data')
parser.add_argument('--tile_data', type=str, default='', help='path to tile data')
parser.add_argument('--encoder', type=str, default='', choices=[
    'tres50_imagenet',
    'ctranspath',
    'phikon',
    'uni',
    'gigapath',
    'virchow',
    'h-optimus-0',
    'dinosmall',
    'dinobase',
], help='choice of encoder')
parser.add_argument('--tilesize', type=int, default=224, help='tile size')
parser.add_argument('--bsize', type=int, default=128, help='batchs size')
parser.add_argument('--workers', type=int, default=10, help='workers')

class slide_dataset(data.Dataset):
    def __init__(self, slide, df, trans, tilesize):
        self.slide = slide
        self.df = df
        self.tilesize = tilesize
        self.trans = trans
    def __getitem__(self, index):
        row = self.df.iloc[index]
        size = int(np.round(self.tilesize * row.mult))
        img = self.slide.read_region((int(row.x), int(row.y)), int(row.level), (size, size)).convert('RGB')
        if row.mult != 1:
            img = img.resize((self.tilesize, self.tilesize), Image.LANCZOS)
        img = self.trans(img)
        return img
    def __len__(self):
        return len(self.df)

def main():
    '''
    slide_data has columns:
    - slide_path: full path to slide
    - slide: unique slide identifier
    - tensor_root: full path to root for that data. Need to add the encoder type
    - tensor_name: name of tensor file without path
    tile_data has columns:
    - slide: unique slide identifier
    - x: x coord
    - y: y coord
    - level: pyramid level at which to extract data
    - mult: factor for tile resize
    '''
    global args
    args = parser.parse_args()
    
    # Set up encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, ndim = encoders.get_encoder(args.encoder)
    model.to(device)
    
    # Set up data
    master = pd.read_csv(args.slide_data)
    tiles = pd.read_csv(args.tile_data)

    # Output directory
    row = master.iloc[0]
    if not os.path.exists(row.tensor_root):
        os.mkdir(row.tensor_root)
    
    if not os.path.exists(os.path.join(row.tensor_root, args.encoder)):
        os.mkdir(os.path.join(row.tensor_root, args.encoder))
    
    # Iterate dataset
    with torch.no_grad():
        for i, row in master.iterrows():
            print(f'[{i+1}]/[{len(master)}]', end='\r')
            
            tensor_name = os.path.join(row.tensor_root, args.encoder, row.tensor_name)
            
            if not os.path.exists(tensor_name):
                # Set up slide
                slide = openslide.OpenSlide(row.slide_path)
                
                # Get coords
                grid = tiles[tiles.slide == row.slide].reset_index(drop=True)
                
                # Set up dataset and loader
                dset = slide_dataset(slide, grid, transform, args.tilesize)
                loader = torch.utils.data.DataLoader(dset, batch_size=args.bsize, shuffle=False, num_workers=args.workers)
                
                # Save tensor
                tensor = torch.zeros(len(grid), ndim).float()
                for j, img in enumerate(loader):
                    out = model(img.cuda())
                    tensor[j*args.bsize:j*args.bsize+img.size(0),:] = out.detach().clone()
                
                torch.save(tensor, tensor_name)
    
    print('')

if __name__ == '__main__':
    main()
