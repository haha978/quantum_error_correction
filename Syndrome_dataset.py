# syndrome_dataset.py
import h5py, torch
from torch.utils.data import Dataset, DataLoader

class SyndromeDataset(Dataset):
    def __init__(self, h5_path, syndrome_length, dtype=torch.float32, preload=True):
        self.h5_path = h5_path
        self.syndrome_length = syndrome_length
        self.dtype = dtype
        self.preload = preload
        self.h5 = None
        if self.preload:
            with h5py.File(self.h5_path, "r") as f:
                self.syndrome = f["detectors"][...]
                self.labels = f["labels"][...]
        else:
            self._open_file()
    
    def _open_file(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
            self.syndrome = self.h5["detectors"]
            self.labels = self.h5["labels"]
    
    def __len__(self):
        return self.syndrome.shape[0]
    
    def __getitem__(self, idx):
        if not self.preload and self.h5 is None:
            self._open_file()
        x = self.syndrome[idx]
        x = x.reshape(-1, self.syndrome_length)
        y = self.labels[idx]
        return x, y
    
    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None
    
    def __del__(self):
        self.close()
