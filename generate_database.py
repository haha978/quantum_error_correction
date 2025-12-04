import numpy as np
import os
import glob
import numpy as np
from PIL import Image
import h5py
from utils import create_directory

def split_indices(n, train_frac=0.7, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx

def write_split(src_path, dst_path, indices):    
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for name in ["syndromes", "detectors", "labels", "samples"]:
            data = np.array(src[name])[indices]   # valid for h5py
            dst.create_dataset(name, data=data, compression="gzip")
        dst.create_dataset("circuit", data=src["circuit"][()])
        for k, v in src.attrs.items():
            dst.attrs[k] = v

def main():
    dir_name = "./Uniform_noise"
    out_dir_name = "../QEC_data/"
    train_frac = 0.80
    val_frac = 0.1
    out_dir_train = os.path.join(out_dir_name, "train")
    out_dir_val = os.path.join(out_dir_name, "validation")
    out_dir_test = os.path.join(out_dir_name, "test")
    
    create_directory(out_dir_train)
    create_directory(out_dir_val)
    create_directory(out_dir_test)
    files = [fn for fn in os.listdir(dir_name) if fn[-2:] == 'h5']
    for file in files:
        with h5py.File(os.path.join(dir_name, file), "r") as f:
            n = f["syndromes"].shape[0]
        train_idx, val_idx, test_idx = split_indices(n, train_frac, val_frac, 0)
        write_split(os.path.join(dir_name, file), os.path.join(out_dir_train, f"{file[:-3]}_train.h5"), train_idx)
        write_split(os.path.join(dir_name, file), os.path.join(out_dir_val, f"{file[:-3]}_val.h5"), val_idx)
        write_split(os.path.join(dir_name, file), os.path.join(out_dir_test, f"{file[:-3]}_test.h5"), test_idx)
        print(f"Done. Train/val/test saved {file}")
        
if __name__ == '__main__':
    main()