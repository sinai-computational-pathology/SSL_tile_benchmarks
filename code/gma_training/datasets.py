import os
import numpy as np
import pandas as pd
import torch

def get_datasets(mccv=0, data='', encoder=''):
    # Load slide data
    df = pd.read_csv(os.path.join('root/data/directory', data, 'slide_data.csv'))
    df['tensor_path'] = [os.path.join(x.tensor_root, encoder, x.tensor_name) for _, x in df.iterrows()]
    # Select mccv and clean
    df = df.rename(columns={'mccv{}'.format(mccv):'mccvsplit'})[['slide','target','mccvsplit','tensor_path']]
    # Split into train and val
    df_train = df[df.mccvsplit=='train'].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_val = df[df.mccvsplit=='val'].reset_index(drop=True).drop(columns=['mccvsplit'])
    # Create my loader objects
    dset_train = slide_dataset_classification(df_train)
    dset_val = slide_dataset_classification(df_val)
    return dset_train, dset_val

class slide_dataset_classification(object):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        # number of slides
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get the feature matrix for that slide
        h = torch.load(row.tensor_path)
        # get the target
        return h, row.target
