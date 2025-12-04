"""
Jupyter notebook that plots validation and training loss/accuracy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    val_csv_path = "/home/leom/code/quantum_error_correction/test2/Validation.csv"
    train_csv_path = "/home/leom/code/quantum_error_correction/test2/Train.csv"

    # validation
    val_csv = pd.read_csv(val_csv_path, header = None)
    val_epochs = np.arange(0, 54.1, 1)
    val_acc = np.array(val_csv.iloc[:, 2][1::2])
    val_loss = np.array(val_csv.iloc[:, 2][::2])

    # train
    train_csv = pd.read_csv(train_csv_path, header = None)
    train_epochs = np.arange(0, 54.1, 1)
    train_loss = np.array(train_csv.iloc[:, 2])

    fig, axs = plt.subplots(2, 1, layout='constrained', figsize = (4, 7))

    axs[0].plot(train_epochs, train_loss, c= 'r', linewidth = 1, label = "train loss")
    axs[0].plot(val_epochs, val_loss, c = 'b', linewidth = 1, label = "validation loss")
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].legend(fontsize = 8)
    axs[1].plot(val_epochs, val_acc, c = 'b', linewidth = 1, label = "validation accuracy")
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('validation accuracy')
    axs[1].legend(fontsize = 8)
    axs[0].set_title("low concentration vs high concentration")
    plt.savefig("/home/leom/code/quantum_error_correction/test2/plot.png")

if __name__ == '__main__':
    main()