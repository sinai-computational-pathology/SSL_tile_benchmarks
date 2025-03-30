'''
Sample inference script to generate feature representations of tiles from a foundation model.
'''
import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import openslide
import PIL.Image as Image
import argparse

class slide_dataset(data.Dataset):
    '''
    This dataset class should be included without modifications
    Arguments:
    - slide: openslide object.
    - df: dataframe with coordinates. It has columns x, y. Coordinates are always for level 0.
    - mult: rescaling factor necessary if the right magnification is not available in the slide.
    - level: the level to extract pixel data from the slide.
    - trans: a PIL image transform.
    - tilesize: 224.
    '''
    def __init__(self, slide, df, mult=1., level=0, tilesize=224, transform=None):
        self.slide = slide
        self.df = df
        self.mult = mult
        self.level = level
        self.size = int(np.round(tilesize * mult))
        self.tilesize = tilesize
        self.transform = transform
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self.slide.read_region((int(row.x), int(row.y)), int(self.level), (self.size, self.size)).convert('RGB')
        if self.mult != 1:
            img = img.resize((self.tilesize, self.tilesize), Image.LANCZOS)
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.df)

##### Model Definition #####
def get_model():
    # Define the model
    # Load model weights
    # Here we use a ResNet50 as an example
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Identity()
    return model, 2048
############################

##### Transform Definition #####
def get_trasform():
    # Define image transform
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
################################

def main(args):
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up model
    model, ndim = get_model()
    model.eval()
    model.to(device)
    transform = get_transform()
    
    # Set up data
    slide_data = pd.read_csv(args.slide_data)# Slide dataframe contains columns: slide id* (slide), slide path (slide_path), rescaling factor (mult), slide level (level)
    tile_data = pd.read_csv(args.tile_data)# Tile dataframe contains columns: slide id* (slide), tile coordinate x (x), tile coordinate y (y)
    
    # Iterate through slides        
    for i, row in slide_data.iterrows():
        
        # Output file name
        tensor_name = os.path.join(args.output, f'{row.slide}.pth')
        
        # Set up data
        slide = openslide.OpenSlide(row.slide_path)
        dataset = slide_dataset(slide, tile_data[tile_data.slide==row.slide], mult=row.mult, level=row.level, transform=transform, tilesize=args.tilesize)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        # Extract features and save
        with torch.no_grad():
            tensor = torch.zeros((tile_data.slide==row.slide).sum(), ndim).float()
            for j, img in enumerate(loader):
                out = model(img.to(device))
                tensor[j*args.batch_size:j*args.batch_size+img.size(0),:] = out.detach().clone()
            
            torch.save(tensor, tensor_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_data', type=str, default='', help='path to slide data csv file. It should include the following columns: ')
    parser.add_argument('--tile_data', type=str, default='', help='path to tile datacsv file. It should include the following columns: ')
    parser.add_argument('--output', type=str, default='', help='path to the output directory where .pth files will be saved.')
    parser.add_argument('--batch_size', type=int, default=128, help='batchs size')
    parser.add_argument('--workers', type=int, default=10, help='workers')
    parser.add_argument('--tilesize', type=int, default=224, help='tilesize')
    args = parser.parse_args()
    main(args)
