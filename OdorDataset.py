#create a custom pytorch dataset for the odor dataset
import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
class OdorDataset(Dataset):
    def __init__(self,base_dir, csv_file, transform=None):
        self.base_dir = base_dir
        self.ds = pd.read_csv(base_dir+csv_file)
        self.ds = self.prepare_dataset(self.ds)
        self.labels = self.ds['y']
        self.data = self.ds['embeddings']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def prepare_dataset(self,ds):
        ds['y'] = ds['y'].apply(ast.literal_eval)
        ds['embeddings'] = ds['embeddings'].apply(ast.literal_eval)
        return ds

