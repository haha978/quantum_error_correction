import numpy as np
import os
import numpy as np
from PIL import Image
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import h5py
import argparse
from datetime import datetime
from Syndrome_dataset import SyndromeDataset
from utils import create_directory


def get_args(parser):
    parser.add_argument('--num_epochs', type = int, default = 100, help = "number of epochs (default: 100)")
    parser.add_argument('--batch_size', type = int, default = 1024, help = "batch size (default: 32)")
    parser.add_argument('--checkpoint_path', type = str, help = "PATH to checkpoint. Stores number of epochs, optimizer and model states")
    parser.add_argument('--log_path', type = str, help = "PATH to log directory. stores checkpoints and output data.")
    parser.add_argument('--data_dir', type = str, default = "/home/leom/code/QEC_data/", help = "PATH to data directory (train, test, validation)")
    parser.add_argument('--num_layers', type = int, default = 2, help = "PATH to data directory (train, test, validation)")
    args = parser.parse_args()
    return args

def round_robin_iters(loaders):
    """
    Yield batches from multiple dataloaders by alternating between them.
    Stops once any underlying dataloader is exhausted.
    """
    iterators = [iter(dl) for dl in loaders]
    while True:
        for idx, it_ in enumerate(iterators):
            try:
                yield idx, next(it_)
            except StopIteration:
                return


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers= num_layers ,
            batch_first=True
            )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)

def check_cuda():
    """
    Check whether cuda and device is available or not
    """
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)

def validate(loaders, model, criterion):
    model.eval()
    val_loss = 0
    probs = []
    labels = []
    total_samples = sum(len(dl.dataset) for dl in loaders)
    if total_samples == 0:
        return 0, 0
    with torch.no_grad():
        for _, (input, target) in round_robin_iters(loaders):
            input = input.float().cuda()
            target = target.view(-1).long().cuda()
            output = model(input)
            loss = criterion(output, target)
            # update validation loss
            val_loss += loss.item()*input.size(0)
            prob = F.softmax(output, dim=1)[:, 1].clone()
            probs.extend(prob)
            labels.extend(target)
    probs = [prob.item() for prob in probs]
    preds = np.array([1 if prob > 0.5 else 0 for prob in probs])
    labels = np.array([label.item() for label in labels])
    val_acc = np.sum(preds == labels)/len(labels)
    # Get Validation Loss
    val_loss = val_loss/total_samples if total_samples else 0
    return val_acc, val_loss
        
def train(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0
    total_samples = sum(len(dl.dataset) for dl in loader)
    for _, (input, target) in round_robin_iters(loader):
        input = input.float().cuda()
        target = target.view(-1).long().cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    
    return running_loss/total_samples if total_samples else 0

def get_datasets(data_dir_path, D):
    train_dir = os.path.join(data_dir_path, "train")
    val_dir = os.path.join(data_dir_path, "validation")
    test_dir = os.path.join(data_dir_path, "test")
    datasets_l = []
    for data_dir in [train_dir, val_dir, test_dir]:
        data_path_l = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if fn[-2:] == 'h5']
        datasets = [SyndromeDataset(data_path, D**2 - 1) for data_path in data_path_l]
        datasets_l.append(datasets)
    train_datasets, val_datasets, test_datasets = datasets_l[0], datasets_l[1], datasets_l[2]
    return train_datasets, val_datasets, test_datasets

def main():
    parser = argparse.ArgumentParser(description = "Train image model")
    args = get_args(parser)
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 4
    NUM_EPOCHS = args.num_epochs
    CHECK_PT_PATH = args.checkpoint_path
    LOG_PATH = args.log_path
    create_directory(LOG_PATH)
    # distance and hidden size
    D = 5
    H = 96
    check_cuda()
    DATA_DIR = args.data_dir
    model = LSTMClassifier(input_size = D**2 - 1, hidden_size = H, num_layers = args.num_layers)  
    #send model to GPU
    if torch.cuda.is_available():
        model.cuda()
    
    train_datasets, val_datasets, test_datasets = get_datasets(DATA_DIR, D)
    train_dataloader = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) for ds in train_datasets]
    val_dataloader = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) for ds in val_datasets]
    
    labels = np.concatenate([ds.labels for ds in train_datasets])
    num_zero = (labels == 0).sum()
    num_one = labels.size - num_zero
    
    weights = torch.tensor([num_one, num_zero], dtype=torch.float32)
    if torch.cuda.is_available():
        weights = weights.cuda()
    
    # Define cross entropy loss: can change the weight if the two classes are imbalanced
    criterion = nn.CrossEntropyLoss(weight = weights).cuda()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # get dataset and dataloader
    if CHECK_PT_PATH:
        checkpoint = torch.load(CHECK_PT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        BATCH_SIZE = checkpoint['batch_size']
        print("Batch size in checkpoint stored override the argpurse input")
    else:
        start_epoch = 0
        
    for epoch in range(start_epoch, NUM_EPOCHS):
        # obtain the loss (with gradient ON) on training data for back-propagation
        loss = train(train_dataloader, model, criterion, optimizer)
        print(f"For epoch: {epoch}, train loss: {loss}")
        fconv = open(os.path.join(LOG_PATH, 'Train.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch,loss))
        fconv.close()
        
        # every epoch obtain validation accuracy and loss
        val_acc, val_loss = validate(val_dataloader, model, criterion)
        print(f"For epoch: {epoch}, validation loss: {val_loss}, validation accuracy: {val_acc}")
        
        # Need to save this to file
        fconv = open(os.path.join(LOG_PATH, 'Validation.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch,val_loss))
        fconv.write('{},accuracy,{}\n'.format(epoch,val_acc))
        fconv.close()
        
        #saving checkpoint
        obj = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(), 'batch_size': BATCH_SIZE}
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        print("This is time now: ", time_now)
        #always save the model and parameter at the end of the iterations!
        if epoch % 2 == 0:
            #every forth iteration save
            torch.save(obj, os.path.join(LOG_PATH,'checkpoint_best_{}_{}.pth'.format(epoch, time_now)))
            
if __name__ == '__main__':
    main()
