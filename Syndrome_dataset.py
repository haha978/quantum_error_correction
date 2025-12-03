# syndrome_dataset.py
import h5py, torch
from torch.utils.data import Dataset, DataLoader

class SyndromeDataset(Dataset):
    def __init__(self, h5_path, dtype=torch.float32):
        self.h5 = h5py.File(h5_path, "r")
        self.syndrome = self.h5["syndromes"]
        self.labels = self.h5["labels"]
        self.dtype = dtype
    def __len__(self):
        return self.syndrome.shape[0]
    def __getitem__(self, idx):
        x = torch.as_tensor(self.syndrome[idx], dtype=self.dtype)
        y = torch.as_tensor(self.labels[idx])
        return x, y
    def close(self):
        self.h5.close()
    def __del__(self):
        self.close()