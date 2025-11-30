import stim
from stim import Circuit
import numpy as np
import pymatching
import matplotlib.pyplot as plt


def main():
    num_shots = 100
    num_round = 1000
    p = 3e-3
    circuit = Circuit.generated("surface_code:rotated_memory_z", 
                                distance=5, 
                                rounds=num_round,
                                after_clifford_depolarization=p,
                                before_round_data_depolarization=p,
                                after_reset_flip_probability=p,
                                before_measure_flip_probability=p)
    sampler = circuit.compile_sampler()
    measurement = sampler.sample(num_shots)
    breakpoint()


if __name__ == "__main__":
    main()