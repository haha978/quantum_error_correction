import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class SyndromeDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        self.h5_path = h5_file_path
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.syndrome = self.h5_file['syndromes']
        self.labels = self.h5_file['labels']
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        data = self.syndrome[idx]
        label = self.labels[idx]
        return data, label

    def close(self):
        self.h5_file.close()