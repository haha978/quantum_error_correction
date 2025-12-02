import stim
from stim import Circuit
import numpy as np
import pymatching
import matplotlib.pyplot as plt


def main():
    # The bottom line indicates how rotated surface code is iterated through a model
    # https://quantumcomputing.stackexchange.com/questions/31782/simulating-the-surface-code-with-stim-meaning-of-qubit-coordinates
    
    num_shots = 1000
    num_round = 25
    D = 5
    p = 0.2
    circuit = Circuit.generated("surface_code:rotated_memory_z", 
                                distance=D, 
                                rounds=num_round,
                                after_clifford_depolarization=p,
                                before_round_data_depolarization=p,
                                after_reset_flip_probability=p,
                                before_measure_flip_probability=p)
    
    # 2) Compile a detector sampler
    detector_sampler = circuit.compile_detector_sampler()
    detectors, observables = detector_sampler.sample(num_shots, separate_observables=True)
    sampler = circuit.compile_sampler()
    samples= sampler.sample(num_shots)
    # obs_samples = sampler.sample_observables(num_shots) 
    coords = circuit.get_final_qubit_coordinates()  # list indexed by qubit id
    data_qubits = [qid for qid, xy in coords.items() if xy[0] % 2 == 1 and  xy[1] % 2 == 1]
    ancilla_qubits = [qid for qid, xy in coords.items() if not (xy[0] % 2 == 1 and  xy[1] % 2 == 1)]
    logical_z_qubits = [qid for qid, c in coords.items() if c[0] == 1]
    # Final 25 measurements are the data-qubit readout
    final_data = samples[:, :(D**2 - 1)*num_round].astype(np.uint8).astype(np.float64)
    labels = observables.astype(np.uint8)
    
    

if __name__ == "__main__":
    main()