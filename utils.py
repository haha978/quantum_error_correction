import os
import h5py
import numpy as np

def create_directory(path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Created directory at: {path}")
        except Exception as e:
            print(f"Error creating directory: {e}")
    else:
        print(f"Directory already exists at: {path}")
        
def load_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        assert 'images' in f.keys() and 'labels' in f.keys(), "need h5 file that has images and labels as key"
        img_l = np.array(f['images'])
        label_l = np.array(f['labels'])
    return img_l, label_l