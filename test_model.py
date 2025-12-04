import numpy as np
import os
import numpy as np
from PIL import Image
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from Syndrome_dataset import SyndromeDataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import h5py
import argparse
from datetime import datetime
from train_model import LSTMClassifier, get_datasets

def get_args(parser):
    parser.add_argument('--checkpoint_path', type = str, help = "PATH to checkpoint. Stores number of epochs, optimizer and model states")
    parser.add_argument('--data_dir', type = str, help = "PATH to directory to database.")
    parser.add_argument('--log_dir', type = str, help = "PATH to directory to database.")
    args = parser.parse_args()
    return args

def test(loader, model, criterion):
    model.eval()
    test_loss = 0
    probs = []
    labels = []
    with torch.no_grad():
        for idx, (input, target) in enumerate(loader):
            input = input.float().cuda()
            target = target.view(-1).long().cuda()
            output = model(input)
            
            loss = criterion(output, target)
            # update validation loss
            test_loss += loss.item()*input.size(0)
            prob = F.softmax(output, dim=1)[:, 1].clone()
            probs.extend(prob)
            labels.extend(target)
    probs = [prob.item() for prob in probs]
    preds = np.array([1 if prob > 0.5 else 0 for prob in probs])
    labels = np.array([label.item() for label in labels])
    test_acc = np.sum(preds == labels)/len(labels)
    # Get Validation Loss
    test_loss = test_loss/len(loader.dataset)
    return test_acc, test_loss


def main():
    parser = argparse.ArgumentParser(description = "Test LSTM")
    args = get_args(parser)
    DATA_DIR = args.data_dir
    CHECK_PT_PATH = args.checkpoint_path
    num_workers = 4
    D = 5
    H = 96
    #define model. Final layer has two nodes to computer cross-entropy loss
    model = LSTMClassifier(input_size = D**2 - 1, hidden_size = H, num_layers = 2)
    train_datasets, _, test_datasets = get_datasets(DATA_DIR, D)
    num_rounds = [test_ds.syndrome.shape[1]/test_ds.syndrome_length for test_ds in test_datasets]
    
    labels = np.concatenate([ds.labels for ds in train_datasets])
    num_zero = (labels == 0).sum()
    num_one = labels.size - num_zero
    weights = torch.tensor([num_one, num_zero], dtype=torch.float32)
    if torch.cuda.is_available():
        weights = weights.cuda()
    
    # Define cross entropy loss: can change the weight if the two classes are imbalanced
    criterion = nn.CrossEntropyLoss(weights).cuda()
    #send model to GPU
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(CHECK_PT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    
    test_acc_l = []
    for test_ds in test_datasets:
        test_dataloader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=False)
        test_acc, test_loss = test(test_dataloader, model, criterion)
        print("This is test accuracy: ", test_acc.item())
        print("This is test loss: ", test_loss)
        test_acc_l.append(test_acc.item())
    
    pairs = sorted(zip(num_rounds, test_acc_l), key=lambda x: x[0])
    num_rounds, test_acc_l = zip(*pairs)
    print(num_rounds)
    print(test_acc_l)
    
    #NEED SAVING STAGE


if __name__ == '__main__':
    main()