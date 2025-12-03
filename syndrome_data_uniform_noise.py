import stim
from stim import Circuit
import numpy as np
import pymatching
import matplotlib.pyplot as plt
import h5py

def add_gate_depolarizing(base_circuit: stim.Circuit, p1: float, p2: float) -> stim.Circuit:
    # 1-qubit Clifford gates that appear in surface_code:rotated_memory_z
    one_qubit_gates = {
        "H",
        "S", "S_DAG",
        "SQRT_X", "SQRT_X_DAG",
        "SQRT_Y", "SQRT_Y_DAG",
        # Add more if needed - not the case for this project since it's only quantum memory
    }

    # 2-qubit Clifford gates used (rotated code uses CX)
    two_qubit_gates = {
        "CX",
        "CZ",
        # Add others if needed
    }

    out = stim.Circuit()

    for inst in base_circuit:
        name = inst.name

        # always copy the original instruction
        out.append(inst)

        # get all target indices (for H/CX these are qubits)
        qubits = [t.value for t in inst.targets_copy()]

        # then insert depolarizing depending on gate type
        if name in one_qubit_gates and qubits:
            out.append("DEPOLARIZE1", qubits, p1)

        elif name in two_qubit_gates and qubits:
            # DEPOLARIZE2(p) q0 q1 q2 q3 ... applies to pairs (q0,q1), (q2,q3), ...
            out.append("DEPOLARIZE2", qubits, p2)

    return out

def main():
    # The bottom line indicates how rotated surface code is iterated through a model
    # https://quantumcomputing.stackexchange.com/questions/31782/simulating-the-surface-code-with-stim-meaning-of-qubit-coordinates
    
    #Circuit parameters
    distance = 5
    num_round = 1
    num_shots = 100
    # Uniform noise in circuit 
    p1 = 0.0005  # single qubit gate noise
    p2 = 0.004 # two qubit gate noise
    pRM = 0.00195
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=num_round,
        distance=distance,
        after_clifford_depolarization=0.0,   
        after_reset_flip_probability=pRM,     
        before_measure_flip_probability=pRM,
        before_round_data_depolarization=0.0,
)

    circuit = add_gate_depolarizing(base, p1=p1, p2=p2)
    
    # 2) Compile a detector sampler
    detector_sampler = circuit.compile_detector_sampler()
    detectors, observables = detector_sampler.sample(num_shots, separate_observables=True)
    sampler = circuit.compile_sampler()
    samples= sampler.sample(num_shots)
    coords = circuit.get_final_qubit_coordinates()  # list indexed by qubit id
    data_qubits = [qid for qid, xy in coords.items() if xy[0] % 2 == 1 and  xy[1] % 2 == 1]
    ancilla_qubits = [qid for qid, xy in coords.items() if not (xy[0] % 2 == 1 and  xy[1] % 2 == 1)]
    logical_z_qubits = [qid for qid, c in coords.items() if c[0] == 1]
    # Final 25 measurements are the data-qubit readout
    final_data = samples[:, :(distance**2 - 1)*num_round].astype(np.uint8).astype(np.float64)
    labels = observables.astype(np.uint8)

    # ---- Save to HDF5 ----
    with h5py.File("surface_code_dataset.h5", "w") as hf:
        hf.create_dataset("syndromes", data=final_data, compression="gzip")
        hf.create_dataset("detectors", data=detectors, compression="gzip")
        hf.create_dataset("labels", data=labels, compression="gzip")
    

if __name__ == "__main__":
    main()