import stim
from stim import Circuit
import numpy as np
import pymatching
import matplotlib.pyplot as plt

def main():
    # generate the initial circuit
    p = 3e-3
    num_round_l = np.arange(1, 10, 1)
    error_pred_l = []
    for num_round in num_round_l:
        circuit = Circuit.generated("surface_code:rotated_memory_z", 
                                            distance=5, 
                                            rounds=num_round,
                                            after_clifford_depolarization=p,
                                            before_round_data_depolarization=p,
                                            after_reset_flip_probability=p,
                                            before_measure_flip_probability=p)
        model = circuit.detector_error_model(decompose_errors = True)
        matching = pymatching.Matching.from_detector_error_model(model)
        breakpoint()

        total_shots = 10000
        sampler = circuit.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=total_shots, separate_observables=True)
        num_errors = 0
        for i in range(syndrome.shape[0]):
            predicted_observables = matching.decode(syndrome[i])
            if predicted_observables.item() != actual_observables[i, :].item():
                num_errors += 1
        error_pred_l.append(num_errors/total_shots)
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (7, 4), layout = "constrained", dpi = 300)
    ax.plot(num_round_l, error_pred_l, 'r', marker = 'o', rasterized = True)
    ax.set_xlabel("number of rounds")
    ax.set_ylabel("error")
    plt.savefig("trial.pdf")
    

if __name__ == "__main__":
    main()