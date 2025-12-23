import stim
from stim import Circuit
import numpy as np
import pymatching
import matplotlib.pyplot as plt
import h5py
import os
from typing import Dict, Optional, Any


def add_gate_depolarizing_with_bad_qubits(
    base: stim.Circuit,
    p1: float,
    p2: float,
    bad_qubits: Optional[Dict[int, float]] = None,
) -> stim.Circuit:
    """
    Take a base circuit and return a new circuit where:
      - Every 1q Clifford gate (in ONE_QUBIT) gets DEPOLARIZE1(p1 * factor(q))
      - Every 2q Clifford gate (in TWO_QUBIT) gets DEPOLARIZE2(p2 * factor(a)*factor(b))
        (PRODUCT rule for 2-qubit gates)

    bad_qubits: dict {qubit_id: factor}. If a qubit isn't in the dict, factor=1.0.

    Handles REPEAT blocks by recursing into their bodies.
    Only touches 1q/2q Clifford gates in the sets below. Everything else is copied as-is.
    """

    if bad_qubits is None:
        bad_qubits = {}

    noisy = stim.Circuit()

    # Extend these sets if your generated circuit uses more gates
    ONE_QUBIT = {"H"}       # e.g., add "S", "SQRT_X", ...
    TWO_QUBIT = {"CX"}      # e.g., add "CZ" if present

    def q_factor(q: int) -> float:
        return float(bad_qubits.get(q, 1.0))

    for inst in base:
        # --- Case 1: REPEAT block -> recurse into its body ---
        if isinstance(inst, stim.CircuitRepeatBlock):
            inner_noisy = add_gate_depolarizing_with_bad_qubits(
                inst.body_copy(),
                p1=p1,
                p2=p2,
                bad_qubits=bad_qubits,
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
        if not qubit_targets:
            continue

        # ----- 1-qubit Clifford gates -----
        if name in ONE_QUBIT:
            # group by factor so we can append fewer DEPOLARIZE1 instructions
            buckets: Dict[float, list[int]] = {}
            for q in qubit_targets:
                f = q_factor(q)
                buckets.setdefault(f, []).append(q)

            for f, qs in buckets.items():
                noisy.append_operation("DEPOLARIZE1", qs, [p1 * f])

        # ----- 2-qubit Clifford gates -----
        elif name in TWO_QUBIT:
            assert len(qubit_targets) % 2 == 0, "2q gate must have even # of qubit targets"

            # PRODUCT factor rule: f_pair = f(a) * f(b)
            buckets2: Dict[float, list[int]] = {}
            for i in range(0, len(qubit_targets), 2):
                a, b = qubit_targets[i], qubit_targets[i + 1]
                f_pair = q_factor(a) * q_factor(b)
                buckets2.setdefault(f_pair, []).extend([a, b])

            for f_pair, pairs_flat in buckets2.items():
                noisy.append_operation("DEPOLARIZE2", pairs_flat, [p2 * f_pair])

    return noisy


def main():
    # Circuit parameters
    distance = 5
    num_shots = 100000  # increase later if you want better statistics

    # Uniform noise in circuit
    p1 = 0.0005     # single qubit gate noise
    p2 = 0.004      # two qubit gate noise
    pRM = 0.00195   # reset and measurement noise

    # Multiple bad qubits: dict qubit_id -> factor
    bad_qubits = {
        16: 10.0,
        19: 5.0,
        # add more if you want
        # 9: 3.0,
    }

    round_values = list(range(2, 22, 2))

    output_dir = "Multiple_Bad_Qubits"
    os.makedirs(output_dir, exist_ok=True)

    # A stable tag for filenames/metadata
    bad_tag = "_".join(f"q{q}x{bad_qubits[q]:g}" for q in sorted(bad_qubits)) if bad_qubits else "none"

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

        circuit = add_gate_depolarizing_with_bad_qubits(
            base,
            p1=p1,
            p2=p2,
            bad_qubits=bad_qubits,
        )

        # Compile samplers
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

        final_data = samples[:, :(distance**2 - 1) * num_round].astype(np.uint8)
        labels = observables.astype(np.uint8)

        filename = os.path.join(
            output_dir,
            f"d{distance}_r{num_round}_{bad_tag}.h5"
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
            hf.attrs["bad_qubits"] = str(bad_qubits)  # easy to read back later
            hf.attrs["bad_tag"] = bad_tag

            # Optional: store structured bad qubits too
            hf.create_dataset("bad_qubit_ids", data=np.array(sorted(bad_qubits.keys()), dtype=np.int32))
            hf.create_dataset("bad_qubit_factors", data=np.array([bad_qubits[q] for q in sorted(bad_qubits)], dtype=np.float64))

        print(f"  Saved dataset to {filename}")


if __name__ == "__main__":
    main()
