import stim
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from matplotlib.patches import Arc, Wedge, Rectangle
import os

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


def get_qubit_coords(circuit: stim.Circuit):
    coords = {}
    for inst in circuit:
        if inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            if len(args) != 2:
                continue
            x, y = args
            for t in inst.targets_copy():
                coords[t.value] = (x, y)
    return coords

def classify_qubits(circuit: stim.Circuit):

    data_qubits = set()
    ancilla_qubits = set()

    # Find final M
    for inst in reversed(list(circuit)):
        if inst.name == "M":  # final data measurement
            for t in inst.targets_copy():
                data_qubits.add(t.value)
            break

    # Find ancillas from MR instructions
    for inst in circuit:
        if inst.name == "MR":
            for t in inst.targets_copy():
                q = t.value
                if q not in data_qubits:
                    ancilla_qubits.add(q)

    return data_qubits, ancilla_qubits


def _is_data_neighbor(p1, p2):
    """Rotated surface-code adjacency: Manhattan distance 2."""
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2) == 2

def _circle_from_3pts(A, B, C):
    """Return (center, radius) of circle through three points, or (None, None) if colinear."""
    (x1, y1), (x2, y2), (x3, y3) = A, B, C

    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2.0
    cd = (temp - x3**2 - y3**2) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

    if abs(det) < 1e-9:
        return None, None  # points are (almost) colinear

    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = math.hypot(cx - x1, cy - y1)
    return (cx, cy), r

def _choose_arc(theta1, theta2, thetaC):
    """
    Given angles (radians) for data1, data2, ancilla, choose start/end angle
    for the *shorter* arc between data1 and data2 that passes through ancilla.
    Returns (start_deg, end_deg).
    """
    twopi = 2 * math.pi
    def norm(t): return t % twopi

    t1, t2, tC = map(norm, (theta1, theta2, thetaC))

    def contains(start, end, mid):
        """Is mid on the CCW arc from start to end?"""
        return (mid - start) % twopi <= (end - start) % twopi + 1e-9

    d12 = (t2 - t1) % twopi  # CCW distance t1 -> t2

    # Two candidate arcs: t1->t2 and t2->t1
    candidates = []
    if contains(t1, t2, tC):
        candidates.append((d12, t1, t2))
    if contains(t2, t1, tC):
        candidates.append((twopi - d12, t2, t1))

    if not candidates:
        # Fallback: just use the shorter arc between t1 and t2
        if d12 <= twopi - d12:
            start, end = t1, t2
        else:
            start, end = t2, t1
    else:
        _, start, end = min(candidates, key=lambda x: x[0])

    return math.degrees(start), math.degrees(end)

def get_ancilla_order(circuit: stim.Circuit, ancilla_qubits: set[int]) -> list[int]:
    """
    Return the list of ancilla-qubit ids in the order they are measured
    in the first MR block. This order repeats every round in the surface-code
    memory circuits.
    """
    ancilla_order = []
    for inst in circuit:
        if inst.name == "MR":
            for t in inst.targets_copy():
                q = t.value
                if q in ancilla_qubits:
                    ancilla_order.append(q)
            break  # only need the first MR block
    return ancilla_order


import math
import itertools

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Wedge, Rectangle


def plot_surface_code_layout(
    circuit: stim.Circuit,
    title="Rotated surface code",
    ancilla_values: dict[int, int] | None = None,
):
    coords = get_qubit_coords(circuit)
    data_qubits, ancilla_qubits = classify_qubits(circuit)

    # Dicts of qubit -> (x, y)
    data_pts = {q: coords[q] for q in coords if q in data_qubits}
    anc_pts = {q: coords[q] for q in coords if q in ancilla_qubits}

    fig, ax = plt.subplots(figsize=(7, 7))

    # We'll store geometry of boundary plaquettes so we can fill them later
    boundary_blocks: dict[int, tuple[float, float, float, float, float]] = {}

    # --- 1) Straight edges between neighboring data qubits (fixed grey) ---
    for q1, q2 in itertools.combinations(data_pts, 2):
        p1, p2 = data_pts[q1], data_pts[q2]
        if _is_data_neighbor(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="0.6",      # fixed light grey
                linewidth=1.5,
                alpha=0.9,
                zorder=0,
            )

    # --- 2) Semicircles on boundary edges (through boundary ancillas) ---
    for aq, (ax_, ay_) in anc_pts.items():
        # Find data neighbors of this ancilla
        neighbors = [
            dq for dq, p in data_pts.items()
            if _is_data_neighbor((ax_, ay_), p)
        ]
        if len(neighbors) != 2:
            continue  # interior (4 neighbors) or odd case, skip for arcs

        d1, d2 = neighbors
        A = data_pts[d1]
        B = data_pts[d2]
        C = (ax_, ay_)

        center, r = _circle_from_3pts(A, B, C)
        if center is None:
            continue
        cx, cy = center

        # Angles of the three points
        tA = math.atan2(A[1] - cy, A[0] - cx)
        tB = math.atan2(B[1] - cy, B[0] - cx)
        tC = math.atan2(C[1] - cy, C[0] - cx)

        start_deg, end_deg = _choose_arc(tA, tB, tC)

        # Store geometry of this boundary "block" for colouring
        boundary_blocks[aq] = (cx, cy, r, start_deg, end_deg)

        # Outline arc (fixed grey)
        arc = Arc(
            (cx, cy),
            2 * r,
            2 * r,
            angle=0,
            theta1=start_deg,
            theta2=end_deg,
            color="grey",   # grey outline
            linewidth=12,
            linestyle="-",
            alpha=0.9,
            zorder=1,
        )
        ax.add_patch(arc)

    # --- 2b) Fill plaquettes (boundary & interior) based on ancilla_values ---
    if ancilla_values is not None:
        for q, (x, y) in anc_pts.items():
            v = int(ancilla_values.get(q, 0))
            # nice soft colours; feel free to tweak
            if v == 0:
                fill_color = "#A6CEE3"    # Blue Medium
            else:
                fill_color = "#E15759"    # Gold Medium

            if q in boundary_blocks:
                # Boundary ancilla: fill semicircle plaquette
                cx, cy, r, start_deg, end_deg = boundary_blocks[q]
                wedge = Wedge(
                    center=(cx, cy),
                    r=r,
                    theta1=start_deg,
                    theta2=end_deg,
                    facecolor=fill_color,
                    edgecolor="none",
                    alpha=0.9,
                    zorder=0.5,
                )
                ax.add_patch(wedge)
            else:
                # Interior ancilla: draw a small square block behind it
                data_positions = list(data_pts.values())
                if len(data_positions) >= 2:
                    # find nearest Manhattan-2 neighbor distance
                    # (points differ by 2 units in grid  spacing = 2)
                    sample_spacing = min(
                        math.hypot(x1 - x2, y1 - y2)
                        for (x1, y1), (x2, y2) in itertools.combinations(data_positions, 2)
                        if abs(x1 - x2) + abs(y1 - y2) == 2
                    )
                else:
                    sample_spacing = 2.0  # fallback

                block_size = sample_spacing  # 60% of plaquette width
             #   block_size = 0.8  # tweak if you want bigger/smaller blocks
                rect = Rectangle(
                    (x - block_size / 2, y - block_size / 2),
                    block_size,
                    block_size,
                    facecolor=fill_color,
                    edgecolor="none",
                    alpha=0.9,
                    zorder=0.5,
                )
                ax.add_patch(rect)

    # --- 3) Plot ancilla qubits (circles, always orange) ---
    for q, (x, y) in anc_pts.items():
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=140,
            facecolors="#F28E2B",
            edgecolors="white",       # Rose Medium 
            linewidths=0.1,
            zorder=1,
        )
        ax.text(
            x,
            y - 0.07,
            str(q),
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            zorder=4,
        )

    # --- 4) Plot data qubits (squares) ---
    for q, (x, y) in data_pts.items():
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=180,
            facecolors="#4568B0",       # Blue Medium
            edgecolors="white",
            linewidths=0.2,
            zorder=1,
        )
        ax.text(
            x,
            y + 0.12,
            str(q),
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            zorder=6,
        )

    # --- 5) Formatting ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle=":")

    ax.legend(
        handles=[
            plt.Line2D(
                [0], [0],
                marker="s",
                linestyle="",
                markerfacecolor="#4568B0",
                markeredgecolor="white",
                markersize=10,
                label="Data qubits",
            ),
            plt.Line2D(
                [0], [0],
                marker="s",
                linestyle="",
                markerfacecolor="#F28E2B",
                markeredgecolor="white",
                markersize=10,
                label="Ancilla qubits",
            ),
        ]
    )

    ax.invert_yaxis()  # match Stim's orientation
    plt.tight_layout()
    plt.show()



def visualize_syndrome_for_shot_round(
    circuit: stim.Circuit,
    distance: int,
    num_round: int,
    final_data: np.ndarray,
    shot_idx: int = 0,
    round_idx: int | None = None,
    title_prefix: str = "Rotated surface code : Syndromes",
):
    """
    Visualise ancilla (syndrome) measurements for a given shot and round.

    final_data is assumed to be the slice you already have:
        final_data = samples[:, :(distance**2 - 1) * num_round].astype(np.uint8)
    i.e. shape (num_shots, num_round * num_ancilla).
    """
    if round_idx is None:
        round_idx = num_round - 1  # default: last round

    data_qubits, ancilla_qubits = classify_qubits(circuit)
    ancilla_order = get_ancilla_order(circuit, ancilla_qubits)

    num_shots = final_data.shape[0]
    num_ancilla = distance**2 - 1

    # Reshape your existing final_data slice into [shot, round, ancilla_index]
    ancilla_bits = final_data.reshape(num_shots, num_round, num_ancilla)

    # Build {ancilla_qubit_id: 0/1} for the chosen shot and round
    syndrome_for_plot = {
        q: int(ancilla_bits[shot_idx, round_idx, k])
        for k, q in enumerate(ancilla_order)
    }

    title = f"{title_prefix} (shot {shot_idx+1}, round {round_idx+1})"
    plot_surface_code_layout(
        circuit,
        title=title,
        ancilla_values=syndrome_for_plot,
    )


def main():
    # Code distance and shots
    distance = 5
    num_shots = 10**5

    # Noise parameters
    p1 = 0.0005   # single qubit gate depolarizing
    p2 = 0.004    # two qubit gate depolarizing
    pRM = 0.00195 # reset/measurement flip probability

    #Enter the round values you want to generate data for
    round_values = list(range(1, 2))

    output_dir = "Uniform_noise"
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
        visualize_syndrome_for_shot_round(
            circuit=circuit,
            distance=distance,
            num_round=num_round,
            final_data=final_data,
            shot_idx=0,           # which shot
            round_idx=num_round-1 # which round; or None for last round
        )

    

if __name__ == "__main__":
    main()