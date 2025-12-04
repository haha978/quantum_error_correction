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

    # Optional: conditional error rate given at least one detection event
    has_any_detection = detectors.any(axis=1)
    if has_any_detection.any():
        cond_logical_error_rate = logical_error_bits[has_any_detection].mean()
    else:
        cond_logical_error_rate = np.nan

    stats = {
        "distance": distance,
        "num_round": num_round,
        "num_shots": num_shots,
        "logical_error_rate": logical_error_rate,
        "cond_logical_error_rate": cond_logical_error_rate,
    }

    return stats


def main():
    distance = 5
    data_dir = "Bad_qubit"
    round_values = list(range(2, 22, 2))
    bad_qubit = 16
    noise_factor = 5.0 

    round_list = []
    ler_list = []
    ler_cond_list = []

    for num_round in round_values:
        filename = os.path.join(data_dir, f"d{distance}_r{num_round}_b{bad_qubit}_f{noise_factor}.h5")
        if not os.path.isfile(filename):
            print(f"WARNING: file not found: {filename} – skipping")
            continue

        stats = decode_file(filename)
        round_list.append(stats["num_round"])
        ler_list.append(stats["logical_error_rate"])
        ler_cond_list.append(stats["cond_logical_error_rate"])

        print(
            f"r = {stats['num_round']:2d} | "
            f"LER = {stats['logical_error_rate']:.3e} | "
            f"LER | syndrome!=0 = {stats['cond_logical_error_rate']:.3e}"
        )

    round_arr = np.array(round_list)
    ler_arr = np.array(ler_list)
    ler_cond_arr = np.array(ler_cond_list)

    # Plot logical error rate vs rounds
    plt.figure()
    plt.plot(round_arr, ler_arr, marker="o", label="Logical Z error rate")
    plt.plot(round_arr, ler_cond_arr, marker="s", linestyle="--",
             label="Logical Z error | at least one detection")
    plt.xlabel("Number of rounds")
    plt.ylabel("Error rate")
    #plt.yscale("log")
    plt.title(f"MWPM performance vs rounds (d={distance})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()