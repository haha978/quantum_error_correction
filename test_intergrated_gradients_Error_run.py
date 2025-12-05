import argparse
import os
from typing import Optional
import matplotlib.pyplot as plt
import torch
from Syndrome_dataset import SyndromeDataset
from train_model import LSTMClassifier
import numpy as np
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Run Integrated Gradients on a trained LSTM classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file (.pth) containing the trained model.")
    parser.add_argument("--train_dir", required=True, help="directory to train data")
    parser.add_argument("--steps", type=int, default=64, help="Number of interpolation steps for Integrated Gradients.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the heatmap image.")
    parser.add_argument("--bad_qubit", type=int, default=None, help="bad qubit number")
    return parser.parse_args()


def load_model(checkpoint_path, input_size, hidden_size, num_layers, device):
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    return model

def integrated_gradients(model, input_tensor, labels, criterion, target_class=None, steps=64, baseline=None):
    device = input_tensor.device
    if baseline is None:
        baseline = torch.zeros_like(input_tensor, device=device)
    delta = input_tensor - baseline
    if target_class is None:
        with torch.no_grad():
            pred = model(input_tensor)
            target_class = int(pred.argmax(dim=1).item())
    target_class = int(target_class)

    total_grad = torch.zeros_like(input_tensor)
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        scaled = baseline + alpha * delta
        scaled.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        output = model(scaled)
        loss = criterion(output, labels)
        loss.backward()
        total_grad += scaled.grad.detach()
    avg_grad = total_grad / steps
    attributions = delta * avg_grad
    return attributions


def plot_heatmap(attributions, save_path: Optional[str] = None, title: Optional[str] = None):
    data = attributions.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(data, cmap="bwr", aspect="auto")
    plt.colorbar(label="Integrated Gradient")
    plt.xlabel("Qubit index")
    plt.ylabel("Measurement round")
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_spatial_profile(attributions, save_path: Optional[str] = None):
    spatial = attributions.sum(dim=1).squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(8, 3))
    plt.bar(range(len(spatial)), spatial, color="tab:blue")
    plt.xlabel("Qubit index")
    plt.ylabel("Integrated Gradient (sum over time)")
    plt.tight_layout()
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved spatial profile to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    input_size = 5 ** 2 - 1
    train_dir = args.train_dir
    train_data_path_l = [os.path.join(train_dir, fn) for fn in os.listdir(train_dir) if fn[-2:] == 'h5']
    train_datasets = [SyndromeDataset(train_dp, input_size) for train_dp in train_data_path_l]
    labels = np.concatenate([ds.labels for ds in train_datasets])
    num_zero = (labels == 0).sum()
    num_one = labels.size - num_zero
    weights = torch.tensor([num_one, num_zero], dtype=torch.float32)
    if torch.cuda.is_available():
        weights = weights.cuda()
    
    # Define cross entropy loss: can change the weight if the two classes are imbalanced
    criterion = nn.CrossEntropyLoss(weight = weights).cuda()
    model = load_model(
        args.checkpoint,
        input_size=input_size,
        hidden_size=96,
        num_layers=6,
        device=device,
    )
    for round in list(range(2, 21, 2)):
        data_path = os.path.join(f"/home/leom/code/QEC_data/Bad_QEC_data_{args.bad_qubit}/test", f"d5_r{round}_b{args.bad_qubit}_f10.0_test.h5")
        dataset = SyndromeDataset(data_path, input_size)
        round_attributions = []  
        for idx in range(len(dataset)):
            sample, label = dataset[idx]
            sample = torch.from_numpy(sample).unsqueeze(0).float().to(device)
            label = torch.from_numpy(label).long().to(device)
            # sample, label = get_sample(, input_size, args.sample_index, device)
            baseline = torch.zeros_like(sample, device=device)
            attributions = integrated_gradients(
                model,
                sample,
                label,
                criterion,
                steps=args.steps,
                baseline=baseline,
            )
            round_attributions.append(attributions.detach().cpu())
            if idx % 100 == 0:
                print(f"index: {idx} is done")
        data = np.array(torch.cat(round_attributions, dim=0))
        print(f"Done with round {round}")
        np.save(os.path.join(args.output, f"att_final_r{round}.npy"), data)

if __name__ == "__main__":
    main()
