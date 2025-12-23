import os
import numpy as np
import h5py
import stim
import pymatching
import matplotlib.pyplot as plt


def decode_file(filename: str):
    """Load one HDF5 file, build Matching from its circuit, and compute MWPM performance."""
    with h5py.File(filename, "r") as hf:
        detectors = hf["detectors"][()]           # shape (shots, num_detectors)
        labels = hf["labels"][()]                # shape (shots, num_observables)
        circuit_bytes = hf["circuit"][()]        # stored as bytes
        distance = int(hf.attrs["distance"])
        num_round = int(hf.attrs["num_round"])
        num_shots = int(hf.attrs["num_shots"])

    # Decode circuit string
    if isinstance(circuit_bytes, bytes):
        circuit_str = circuit_bytes.decode()
    else:
        # if it's a 0-d array of bytes, handle that too
        circuit_str = circuit_bytes.astype(str)

    circuit = stim.Circuit(circuit_str)

    # Build detector error model and Matching
    dem = circuit.detector_error_model()
    matching = pymatching.Matching.from_detector_error_model(dem)

    # Run MWPM decoding on all shots
    # matching.decode_batch returns an array of size (shots,)
    predicted_logical_flips = matching.decode_batch(detectors).astype(np.uint8)

    # We assume 1 logical observable (Z) -> use column 0
    obs = labels[:, 0].astype(np.uint8)

    # Net logical error after correction is physical flip XOR correction flip
    logical_error_bits = (obs ^ predicted_logical_flips).astype(np.uint8)
    logical_error_rate = logical_error_bits.mean()

    # # Optional: conditional error rate given at least one detection event
    # has_any_detection = detectors.any(axis=1)
    # if has_any_detection.any():
    #     cond_logical_error_rate = logical_error_bits[has_any_detection].mean()
    # else:
    #     cond_logical_error_rate = np.nan

    stats = {
        "distance": distance,
        "num_round": num_round,
        "num_shots": num_shots,
        "logical_error_rate": logical_error_rate,
        # "cond_logical_error_rate": cond_logical_error_rate,
    }

    return stats
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main():
    distance = 5

    # ---------- Visual styling ----------
    colors = {
        "Uniform_test": "tab:blue",
        "Error_test": "tab:red",
        "Bad_qubit_19_test": "tab:green",
        "Bad_qubit_9_test": "tab:purple",
    }

    mwpm_markers = {
        "Uniform_test": "o",
        "Error_test": "s",
        "Bad_qubit_19_test": "^",
        "Bad_qubit_9_test": "D",
    }

    nn_markers = mwpm_markers  # same shape, but dashed line


    pretty_label = {
        "Uniform_test": "Uniform noise",
        "Error_test": "Bad qubit 13, f=5",
        "Bad_qubit_19_test": "Bad qubit 19, f=10",
        "Bad_qubit_9_test": "Bad qubit 9, f=10",
    }

    # -------- CHANGE YOUR 4 NN CHECKPOINT FILES HERE --------
    nn_file_map = {
        "Uniform_test":       r"C:/Research_Chaitali/phy191a/Uniform_test.npy",
        "Error_test":         r"C:/Research_Chaitali/phy191a/Bad_qubit_13_test.npy",
        "Bad_qubit_19_test":  r"C:/Research_Chaitali/phy191a/Bad_qubit_19_test.npy",
        "Bad_qubit_9_test":   r"C:/Research_Chaitali/phy191a/Bad_qubit_9_test.npy",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create inset (zoom window)
    axins = inset_axes(ax, width="45%", height="45%", loc="center right")

    all_nn_err = []

    # ---------------------------------------------------------
    # LOOP OVER NOISE MODELS
    # ---------------------------------------------------------
    for name in ["Uniform_test", "Error_test", "Bad_qubit_19_test", "Bad_qubit_9_test"]:
        data_dir = f"C:/Research_Chaitali/phy191a/{name}/test"
        round_values = list(range(2, 22, 2))

        # ----------- MWPM curves -----------
        round_list = []
        ler_list = []

        for num_round in round_values:
            if name == "Uniform_test":
                file = os.path.join(data_dir, f"d{distance}_r{num_round}_test.h5")
            elif name == "Error_test":
                file = os.path.join(data_dir, f"d{distance}_r{num_round}_b13_f5.0_test.h5")
            elif name == "Bad_qubit_19_test":
                file = os.path.join(data_dir, f"d{distance}_r{num_round}_b19_f10.0_test.h5")
            elif name == "Bad_qubit_9_test":
                file = os.path.join(data_dir, f"d{distance}_r{num_round}_b9_f10.0_test.h5")

            if not os.path.isfile(file):
                continue

            stats = decode_file(file)
            round_list.append(stats["num_round"])
            ler_list.append(stats["logical_error_rate"])

        # Plot MWPM
        if len(round_list) > 0:
            ax.plot(
                round_list, ler_list,
                linestyle="--",
                marker='o',
                color=colors[name],
                label=f"{pretty_label[name]} (MWPM)"
            )

    # ---------------------------------------------------------
    # NN CURVES — in main plot AND inset plot
    # ---------------------------------------------------------
    for name in ["Uniform_test", "Error_test", "Bad_qubit_19_test", "Bad_qubit_9_test"]:
        nn_file = nn_file_map[name]
        if not os.path.isfile(nn_file):
            continue

        arr = np.load(nn_file, allow_pickle=True)
        obj = arr.item()

        nr = np.array(obj["num_rounds"], dtype=float)
        fidelity = np.array(obj["test_acc_l"], dtype=float)
        err = 1 - fidelity

        all_nn_err.extend(err)

        # Plot in main figure
        ax.plot(
            nr, err,
            linestyle="--",
            marker='s',
            color=colors[name],
            label=f"{pretty_label[name]} (NN)",
        )

        # Plot in inset (NN only)
        axins.plot(
            nr, err,
            linestyle="--",
            marker="s",
            color=colors[name],
        )

    # ---------------------------------------------------------
    # Style main axis
    # ---------------------------------------------------------
    ax.set_xlabel("Number of rounds", fontsize = 16)
    ax.set_ylabel("Logical Error Rate", fontsize = 16)
    #ax.set_title("MWPM vs NN decoder performance")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=12)
    ax.set_xticks(np.arange(2, 22, 2))

    # ---------------------------------------------------------
    # Style inset (auto-zoom)
    # ---------------------------------------------------------
    ymin = min(all_nn_err)
    ymax = max(all_nn_err)
    margin = 0.1 * (ymax - ymin)

    axins.set_ylim(ymin - margin, ymax + margin)
    axins.set_xlim(min(nr) - 0.5, max(nr) + 0.5)
    axins.set_yticks([0, 0.025, 0.05]) 
    axins.set_xticks([4, 8, 12, 16, 20])
    axins.grid(True, ls="--", alpha=0.4)
    axins.set_title("NN Decoder", fontsize=10)

    # Legend OUTSIDE on the far right
    ax.legend(
        fontsize=12,
        bbox_to_anchor=(1.01, 0.7),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.subplots_adjust(right=0.85)
    print("Saving figure to:", os.path.abspath("decoder_performance.png"))
    plt.savefig("decoder_performance.svg", bbox_inches="tight")
    plt.savefig("decoder_performance.png", dpi=300, bbox_inches="tight")
    plt.show()

    
if __name__ == "__main__":
    main()
