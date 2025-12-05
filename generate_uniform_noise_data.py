import os
import numpy as np
import stim
import h5py


def add_gate_depolarizing(base_circuit: stim.Circuit, p1: float, p2: float) -> stim.Circuit:
    # 1-qubit Clifford gates that appear in surface_code:rotated_memory_z
    one_qubit_gates = {
        "H",
        "S", "S_DAG",
        "SQRT_X", "SQRT_X_DAG",
        "SQRT_Y", "SQRT_Y_DAG",
        # Add more if needed
    }

    # 2-qubit Clifford gates used
    two_qubit_gates = {
        "CX",
        "CZ",
        # Add others if needed
    }

    out = stim.Circuit()

    for inst in base_circuit:
        # Case 1: it's a REPEAT block -> recurse into the body
        if isinstance(inst, stim.CircuitRepeatBlock):
            # Get a copy of the body
            body = inst.body_copy()

            # Add depolarizing noise *inside* the body
            noisy_body = add_gate_depolarizing(body, p1, p2)

            # Re-wrap it in a repeat block with the same repeat_count
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
            continue

        # Case 2: it's a normal instruction (CircuitInstruction)
        name = inst.name

        # Always copy the original instruction
        out.append(inst)

        # Targets for this instruction (qubits for H/CX/etc.)
        qubits = [t.value for t in inst.targets_copy()]

        # Then insert depolarizing depending on gate type
        if name in one_qubit_gates and qubits:
            out.append("DEPOLARIZE1", qubits, p1)

        elif name in two_qubit_gates and qubits:
            # DEPOLARIZE2(p) q0 q1 q2 q3 ... applies to (q0,q1), (q2,q3), ...
            out.append("DEPOLARIZE2", qubits, p2)

    return out


def main():
    # Code distance and shots
    distance = 5
    num_shots = 100000

    # Noise parameters
    f = 5  # noise factor
    p1 = 0.0005*f   # single qubit gate depolarizing
    p2 = 0.004*f    # two qubit gate depolarizing
    pRM = 0.00195*f # reset/measurement flip probability


    #Enter the round values you want to generate data for
    round_values = list(range(2, 22, 2))

    output_dir = "C:/Research_Chaitali/phy191a/Uniform_noise/test5"
    os.makedirs(output_dir, exist_ok=True)

    for num_round in round_values:
        print(f"Generating data for num_round = {num_round}")

        # 1) Generate base surface-code memory circuit (no Clifford gate noise yet)
        base = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=num_round,
            distance=distance,
            after_clifford_depolarization=0.0,
            after_reset_flip_probability=pRM,
            before_measure_flip_probability=pRM,
            before_round_data_depolarization=0.0,
        )

        # 2) Add gate depolarization (including inside REPEAT blocks)
        circuit = add_gate_depolarizing(base, p1=p1, p2=p2)

        # 3) Compile samplers
        detector_sampler = circuit.compile_detector_sampler()
        detectors, observables = detector_sampler.sample(
            num_shots,
            separate_observables=True
        )

        sampler = circuit.compile_sampler()
        samples = sampler.sample(num_shots)

        # You had some "final_data" extraction from samples.
        # Keep a similar structure, but note this is NOT the same as detectors.
        # For the NN later, you'll probably want "detectors".
        final_data = samples[:, :(distance**2 - 1) * num_round].astype(np.uint8)

        # Labels = logical observables
        labels = observables.astype(np.uint8)

        # Save to HDF5
        filename = os.path.join(output_dir, f"d{distance}_r{num_round}.h5")
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("syndromes", data=final_data, compression="gzip")
            hf.create_dataset("detectors", data=detectors.astype(np.uint8), compression="gzip")
            hf.create_dataset("labels", data=labels.astype(np.uint8), compression="gzip")
            hf.create_dataset("samples", data=samples.astype(np.uint8), compression="gzip")

            # Store the circuit as a string
            hf.create_dataset("circuit", data=np.array(str(circuit), dtype="S"))

            # Metadata
            hf.attrs["distance"] = distance
            hf.attrs["num_round"] = num_round
            hf.attrs["num_shots"] = num_shots
            hf.attrs["p1"] = p1
            hf.attrs["p2"] = p2
            hf.attrs["pRM"] = pRM

        print(f"  Saved {filename}")


if __name__ == "__main__":
    main()
