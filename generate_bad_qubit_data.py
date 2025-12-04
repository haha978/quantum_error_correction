import stim
from stim import Circuit
import numpy as np
import pymatching
import matplotlib.pyplot as plt
import h5py
import os

def add_gate_depolarizing_with_bad_qubit(
    base: stim.Circuit,
    p1: float,
    p2: float,
    bad_qubit: int | None = None,
    factor: float = 1.0,
) -> stim.Circuit:
    """
    Take a base circuit with no Clifford gate noise and return a new circuit where:
      - Every 1q Clifford gate gets DEPOLARIZE1(p1)
      - Every 2q Clifford gate gets DEPOLARIZE2(p2)
      - If bad_qubit participates in a gate, its noise is multiplied by `factor`.

    Handles REPEAT blocks by recursing into their bodies.
    Only touches 1q/2q Clifford gates (e.g. H, CX). All other instructions are copied as-is.
    """

    noisy = stim.Circuit()

    ONE_QUBIT = {"H"}       # extend if needed
    TWO_QUBIT = {"CX"}      # extend if needed

    for inst in base:
        # --- Case 1: REPEAT block -> recurse into its body ---
        if isinstance(inst, stim.CircuitRepeatBlock):
            inner_noisy = add_gate_depolarizing_with_bad_qubit(
                inst.body_copy(),
                p1=p1,
                p2=p2,
                bad_qubit=bad_qubit,
                factor=factor,
            )
            noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, inner_noisy))
            continue

        # --- Case 2: normal instruction ---
        name = inst.name
        targets = list(inst.targets_copy())      # GateTarget objects
        gate_args = inst.gate_args_copy()

        # 1) copy original instruction exactly
        noisy.append_operation(name, targets, gate_args)

        # 2) figure out which targets are qubits
        qubit_targets = [t.value for t in targets if t.is_qubit_target]

        # ----- 1-qubit Clifford gates -----
        if name in ONE_QUBIT and qubit_targets:
            normal = []
            bad = []
            for q in qubit_targets:
                if bad_qubit is not None and q == bad_qubit:
                    bad.append(q)
                else:
                    normal.append(q)

            if normal:
                noisy.append_operation("DEPOLARIZE1", normal, [p1])
            if bad:
                noisy.append_operation("DEPOLARIZE1", bad, [p1 * factor])

        # ----- 2-qubit Clifford gates -----
        elif name in TWO_QUBIT and qubit_targets:
            assert len(qubit_targets) % 2 == 0, "2q gate must have even # of qubit targets"

            normal_pairs = []
            bad_pairs = []

            for i in range(0, len(qubit_targets), 2):
                a, b = qubit_targets[i], qubit_targets[i + 1]
                if bad_qubit is not None and (a == bad_qubit or b == bad_qubit):
                    bad_pairs.extend([a, b])
                else:
                    normal_pairs.extend([a, b])

            if normal_pairs:
                noisy.append_operation("DEPOLARIZE2", normal_pairs, [p2])
            if bad_pairs:
                noisy.append_operation("DEPOLARIZE2", bad_pairs, [p2 * factor])

    return noisy


def main():
    # Circuit parameters
    distance = 5
    num_shots = 100  # increase later if you want better statistics

    # Uniform noise in circuit 
    p1 = 0.0005     # single qubit gate noise
    p2 = 0.004      # two qubit gate noise
    pRM = 0.00195   # reset and measurement noise

    bad_qubit_num = 16   # bad data qubit index
    noise_factor = 5.0   # factor by which to increase noise on bad qubit


    round_values = list(range(2,22,2))

    output_dir = "Bad_qubit"
    os.makedirs(output_dir, exist_ok=True)

    for num_round in round_values:
        print(f"Simulating num_round = {num_round}")

        base = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=num_round,
            distance=distance,
            after_clifford_depolarization=0.0,
            after_reset_flip_probability=pRM,
            before_measure_flip_probability=pRM,
            before_round_data_depolarization=0.0,
        )

        circuit = add_gate_depolarizing_with_bad_qubit(
            base,
            p1=p1,
            p2=p2,
            bad_qubit=bad_qubit_num,
            factor=noise_factor,
        )

        # 2) Compile samplers
        detector_sampler = circuit.compile_detector_sampler()
        detectors, observables = detector_sampler.sample(
            num_shots,
            separate_observables=True
        )

        sampler = circuit.compile_sampler()
        samples = sampler.sample(num_shots)

        coords = circuit.get_final_qubit_coordinates()  # dict indexed by qubit id
        data_qubits = [qid for qid, xy in coords.items() if xy[0] % 2 == 1 and xy[1] % 2 == 1]
        ancilla_qubits = [qid for qid, xy in coords.items() if qid not in data_qubits]
        logical_z_qubits = [qid for qid, c in coords.items() if c[0] == 1]

        # Final 25 measurements are the data-qubit readout (your original slice)
        final_data = samples[:, :(distance**2 - 1) * num_round].astype(np.uint8)
        labels = observables.astype(np.uint8)

        # ---- Save to HDF5: d{distance}_r{num_round}_b{bad_qubit}_f{factor}.h5 ----
        filename = os.path.join(
            output_dir,
            f"d{distance}_r{num_round}_b{bad_qubit_num}_f{noise_factor}.h5"
        )

        with h5py.File(filename, "w") as hf:
            hf.create_dataset("syndromes", data=final_data, compression="gzip")
            hf.create_dataset("detectors", data=detectors.astype(np.uint8), compression="gzip")
            hf.create_dataset("labels", data=labels, compression="gzip")
            hf.create_dataset("samples", data=samples.astype(np.uint8), compression="gzip")
            hf.create_dataset("circuit", data=np.array(str(circuit), dtype="S"))

            # Metadata
            hf.attrs["distance"] = distance
            hf.attrs["num_round"] = num_round
            hf.attrs["num_shots"] = num_shots
            hf.attrs["p1"] = p1
            hf.attrs["p2"] = p2
            hf.attrs["pRM"] = pRM
            hf.attrs["bad_qubit_num"] = bad_qubit_num
            hf.attrs["noise_factor"] = noise_factor

        print(f"  Saved dataset to {filename}")


if __name__ == "__main__":
    main()