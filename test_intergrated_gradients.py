import argparse
import os
from typing import Optional
import matplotlib.pyplot as plt
import torch
from Syndrome_dataset import SyndromeDataset
from train_model import LSTMClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Run Integrated Gradients on a trained LSTM classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file (.pth) containing the trained model.")
    parser.add_argument("--data_path", required=True, help="Path to the .h5 dataset file to draw samples from.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of the sample inside the dataset to explain.")
    parser.add_argument("--steps", type=int, default=64, help="Number of interpolation steps for Integrated Gradients.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the heatmap image.")
    return parser.parse_args()


def load_model(checkpoint_path, input_size, hidden_size, num_layers, device):
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_sample(data_path, syndrome_length, sample_index, device):
    dataset = SyndromeDataset(data_path, syndrome_length)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"Sample index {sample_index} is out of range for dataset of size {len(dataset)}")
    sample, label = dataset[sample_index]
    sample_tensor = torch.from_numpy(sample).unsqueeze(0).float().to(device)
    label = int(label)
    return sample_tensor, label


def integrated_gradients(model, input_tensor, target_class=None, steps=64, baseline=None):
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
        score = output[:, target_class].sum()
        score.backward()
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
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    input_size = 5 ** 2 - 1
    model = load_model(
        args.checkpoint,
        input_size=input_size,
        hidden_size=96,
        num_layers=6,
        device=device,
    )

    sample, label = get_sample(args.data_path, input_size, args.sample_index, device)
    baseline = torch.zeros_like(sample, device=device)
    attributions = integrated_gradients(
        model,
        sample,
        target_class= label,
        steps=args.steps,
        baseline=baseline,
    )
    print(f"Label: {label}")
    print(f"Predicted class: {model(sample).argmax(dim=1).item()}")
    plot_title = f"Integrated Gradients - sample {args.sample_index}"
    plot_heatmap(attributions, save_path=args.output, title=plot_title)
    if args.output:
        base, ext = os.path.splitext(args.output)
        spatial_path = f"{base}_spatial{ext or '.png'}"
    else:
        spatial_path = None
    plot_spatial_profile(attributions, save_path=spatial_path)

if __name__ == "__main__":
    main()
