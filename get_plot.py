"""
Jupyter notebook that plots validation and training loss/accuracy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    dir_path = "/home/leom/code/QEC_data/Uniform_6_layers"
    val_csv_path = os.path.join(dir_path, "Validation.csv")
    train_csv_path = os.path.join(dir_path, "Train.csv")

    # validation
    val_csv = pd.read_csv(val_csv_path, header = None)
    val_epochs = np.arange(0, 59.1, 1)
    val_acc = np.array(val_csv.iloc[:, 2][1::2])
    val_loss = np.array(val_csv.iloc[:, 2][::2])

    # train
    train_csv = pd.read_csv(train_csv_path, header = None)
    train_epochs = np.arange(0, 59.1, 1)
    train_loss = np.array(train_csv.iloc[:, 2])

    fig, axs = plt.subplots(2, 1, layout='constrained', figsize = (4, 7))

    axs[0].plot(train_epochs, train_loss, c='r', linewidth=0.5, marker='o', markersize=2, label="train loss")
    axs[0].plot(val_epochs, val_loss, c='b', linewidth=0.5, marker='s', markersize=2, label="validation loss")
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].legend(fontsize = 8)
    axs[1].plot(val_epochs, val_acc, c='b', linewidth=0.5, marker='o', markersize=2, label="validation accuracy")
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('validation accuracy')
    axs[1].legend(fontsize = 8)
    axs[0].set_title("LSTM Decoder \n" + f"For {os.path.basename(dir_path)}, best epoch: {np.argmin(val_loss)}\n"\
        + f"val loss: {min(val_loss)}")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "plot.png"))
    print( f"For {os.path.basename(dir_path)}, best epoch: {np.argmin(val_loss)}")

if __name__ == '__main__':
    main()
