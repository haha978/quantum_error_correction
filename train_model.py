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
# from utils import create_directory

def get_args(parser):
    parser.add_argument('--num_epochs', type = int, default = 100, help = "number of epochs (default: 100)")
    parser.add_argument('--batch_size', type = int, default = 32, help = "batch size (default: 32)")
    parser.add_argument('--checkpoint_path', type = str, help = "PATH to checkpoint. Stores number of epochs, optimizer and model states")
    parser.add_argument('--log_path', type = str, help = "PATH to log directory. stores checkpoints and output data.")
    args = parser.parse_args()
    return args

def check_cuda():
    """
    Check whether cuda and device is available or not
    """
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)

def validate(loader, model, criterion):
    model.eval()
    val_loss = 0
    probs = []
    labels = []
    with torch.no_grad():
        for idx, (input, target) in enumerate(loader):
            input = input.float().cuda()
            target = target.float().cuda()
            output = model(input)
            loss = criterion(output, target)
            # update validation loss
            val_loss += loss.item()*input.size(0)
            prob = F.softmax(output, dim=1)[:, 1].clone()
            probs.extend(prob)
            labels.extend(target[:, 1])
    probs = [prob.item() for prob in probs]
    preds = np.array([1 if prob > 0.5 else 0 for prob in probs])
    labels = np.array([label.item() for label in labels])
    val_acc = np.sum(preds == labels)/len(labels)
    # Get Validation Loss
    val_loss = val_loss/len(loader.dataset)
    return val_acc, val_loss
        
def train(loader, model, criterion, optimizer):
    model.train()    
    running_loss = 0
    for i, (input, target) in enumerate(loader):
        input = input.float().cuda()
        target = target.float().cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    
    return running_loss/len(loader.dataset)
    
def main():
    # parser = argparse.ArgumentParser(description = "Train image model")
    # args = get_args(parser)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    D = 5
    H = 128
    # NUM_EPOCHS = args.num_epochs
    # CHECK_PT_PATH = args.checkpoint_path
    # LOG_PATH = args.log_path
    # create_directory(LOG_PATH)
    
    #define model. Final layer has two nodes to computer cross-entropy loss
    model = nn.LSTM(input_size = D**2, hidden_size = H, num_layers = 2)
    breakpoint()
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    #send model to GPU
    if torch.cuda.is_available():
        model.cuda()
    # print(model)
    
    # Define cross entropy loss: can change the weight if the two classes are imbalanced
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
    normalize = transforms.Normalize(mean=[0.5],std=[0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    
    # get dataset and dataloader
    train_dset = ImageDataset("/home/leom/code/droplet_ML/droplet_data/binary_h5_files/training.h5", trans)
    val_dset = ImageDataset("/home/leom/code/droplet_ML/droplet_data/binary_h5_files/validation.h5", trans)
    train_dataloader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=False)
    val_dataloader = DataLoader(val_dset, batch_size=2, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    
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
        if epoch % 4 == 0:
            #every forth iteration save
            torch.save(obj, os.path.join(LOG_PATH,'checkpoint_best_{}_{}.pth'.format(epoch, time_now)))

    
    

if __name__ == '__main__':
    main()