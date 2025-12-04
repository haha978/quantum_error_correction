import stim
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from matplotlib.patches import Arc, Wedge, Rectangle
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

    # Find final M (data measurements)
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

    def norm(t):
        return t % twopi

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


from matplotlib.patches import Arc, Wedge, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import itertools

def plot_surface_code_heatmap_layout(
    circuit: stim.Circuit,
    ancilla_values: dict[int, float],
    title: str = "Syndrome heatmap",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "white_to_red",
):
    """
    Draw rotated surface code layout with heatmap colouring of ancilla plaquettes.

    ancilla_values: dict[ancilla_qubit_id -> float in [0,1]]
                    e.g. probability that syndrome = 1 over shots.

    Colour scheme:
      - Data qubits:    #4568B0  (your blue)
      - Ancilla qubits: #F28E2B  (your orange)
      - Heatmap:
          if cmap_name == "white_to_red":
              white -> pure red (#FF0000)
          if cmap_name == "white_to_rosered":
              white -> your Rose Medium (#E7115E)
          else:
              use any Matplotlib cmap by name, e.g. "Reds"
    """
    coords = get_qubit_coords(circuit)
    data_qubits, ancilla_qubits = classify_qubits(circuit)

    data_pts = {q: coords[q] for q in coords if q in data_qubits}
    anc_pts = {q: coords[q] for q in coords if q in ancilla_qubits}

    fig, ax = plt.subplots(figsize=(7, 7))

    # ---- choose colormap based on cmap_name ----
    if cmap_name == "white_to_red":
        cmap = LinearSegmentedColormap.from_list(
            "white_to_red",
            ["#FFFFFF", "#FF0000"]
        )
    elif cmap_name == "white_to_rosered":
        cmap = LinearSegmentedColormap.from_list(
            "white_to_rosered",
            ["#FFFFFF", "#E7115E"]
        )
    else:
        cmap = plt.get_cmap(cmap_name)

    eps = 1e-12

    # ---- compute lattice spacing for interior block size ----
    data_positions = list(data_pts.values())
    if len(data_positions) >= 2:
        spacings = [
            math.hypot(x1 - x2, y1 - y2)
            for (x1, y1), (x2, y2) in itertools.combinations(data_positions, 2)
            if abs(x1 - x2) + abs(y1 - y2) == 2
        ]
        lattice_spacing = min(spacings) if spacings else 2.0
    else:
        lattice_spacing = 2.0
    block_size = lattice_spacing

    # store boundary semicircle geometry
    boundary_blocks: dict[int, tuple[float, float, float, float, float]] = {}

    # ---- 1) straight edges between neighbouring data qubits ----
    for q1, q2 in itertools.combinations(data_pts, 2):
        p1, p2 = data_pts[q1], data_pts[q2]
        if _is_data_neighbor(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="grey",
                linewidth=2,
                alpha=0.9,
                zorder=1,
            )

    # ---- 2) boundary semicircles (store geometry) ----
    for aq, (ax_, ay_) in anc_pts.items():
        neighbours = [
            dq for dq, p in data_pts.items()
            if _is_data_neighbor((ax_, ay_), p)
        ]
        if len(neighbours) != 2:
            continue

        d1, d2 = neighbours
        A = data_pts[d1]
        B = data_pts[d2]
        C = (ax_, ay_)

        center, r = _circle_from_3pts(A, B, C)
        if center is None:
            continue
        cx, cy = center

        tA = math.atan2(A[1] - cy, A[0] - cx)
        tB = math.atan2(B[1] - cy, B[0] - cx)
        tC = math.atan2(C[1] - cy, C[0] - cx)

        start_deg, end_deg = _choose_arc(tA, tB, tC)

        boundary_blocks[aq] = (cx, cy, r, start_deg, end_deg)

        arc = Arc(
            (cx, cy),
            2 * r,
            2 * r,
            angle=0,
            theta1=start_deg,
            theta2=end_deg,
            edgecolor="grey",
            facecolor="none",
            linewidth=2,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(arc)

    # ---- 3) fill plaquettes according to ancilla_values ----
    vals = []
    for q, (x, y) in anc_pts.items():
        v = float(ancilla_values.get(q, 0.0))
        vals.append(v)
        t = (v - vmin) / (vmax - vmin + eps)
        t = min(max(t, 0.0), 1.0)
        fill_color = cmap(t)

        if q in boundary_blocks:
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

        for q, (x, y) in anc_pts.items():
            v = float(ancilla_values.get(q, 0.0))

            # Boundary plaquette: text goes at the ancilla coordinate (x,y)
            if q in boundary_blocks:
                tx, ty = x, y
            else:
            # Interior plaquette: center of the square
                tx, ty = x, y

            ax.text(
                tx,
                ty+0.5,
                f"{v:.2f}",          # format to 2 decimal places
                ha="center",
                va="top",
                fontsize=7,
                color="black",
                zorder=20,           # ensure text is on top of the heatmap
                clip_on=False,
            )

    # ---- 4) ancilla markers (your orange circles + labels) ----
    for q, (x, y) in anc_pts.items():
        ax.scatter(
            [x],
            [y],
            marker="o",
            s=200,
            facecolors="#F28E2B",
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
        )
        ax.text(
            x,
            y,
            str(q),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=10,
            clip_on=False,
        )

    # ---- 5) data qubits (your blue squares + labels) ----
    for q, (x, y) in data_pts.items():
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=200,
            facecolors="#4568B0",
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
        )
        ax.text(
            x,
            y,
            str(q),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            zorder=10,
            clip_on=False,
        )

    # ---- 6) formatting + colorbar ----
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle=":")

    ax.invert_yaxis()
    plt.tight_layout()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(vals)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(syndrome = 1)")

    plt.show()



def visualize_syndrome_heatmap_for_round(
    circuit: stim.Circuit,
    distance: int,
    num_round: int,
    final_data: np.ndarray,
    round_idx: int,
    title_prefix: str = "Rotated surface code : averaged syndromes",
    cmap_name: str = "Reds",
):
    """
    Build and plot ancilla heatmap for a given round, averaged over shots.

    final_data is assumed to be:
        final_data = samples[:, :(distance**2 - 1) * num_round].astype(np.uint8)
    with shape (num_shots, num_round * num_ancilla).
    """
    num_shots = final_data.shape[0]
    num_ancilla = distance**2 - 1

    # [shots, rounds, ancilla_index]
    ancilla_bits = final_data.reshape(num_shots, num_round, num_ancilla)

    # average over shots for this round
    mean_per_ancilla = ancilla_bits[:, round_idx, :].mean(axis=0)

    data_qubits, ancilla_qubits = classify_qubits(circuit)
    ancilla_order = get_ancilla_order(circuit, ancilla_qubits)

    ancilla_values = {
        q: float(mean_per_ancilla[k])
        for k, q in enumerate(ancilla_order)
    }

    title = f"{title_prefix} (round {round_idx+1})"
    plot_surface_code_heatmap_layout(
        circuit=circuit,
        ancilla_values=ancilla_values,
        title=title,
        vmin=0.0,
        vmax=1.0,
        cmap_name=cmap_name,
    )


def main():
    # Circuit parameters
    distance = 5
    num_shots = 1000000  # increase later if you want better statistics

    # Uniform noise in circuit 
    p1 = 0.0005     # single qubit gate noise
    p2 = 0.004      # two qubit gate noise
    pRM = 0.00195   # reset and measurement noise

    bad_qubit_num = 16   # bad data qubit index
    noise_factor = 10   # factor by which to increase noise on bad qubit

    # Rounds we sweep over: 1, 4, 7, 10, 13, 16
    round_values = list(range(20, 21, 1))

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

        # One heatmap per round (averaged over all shots)
        for r in range(num_round):
            visualize_syndrome_heatmap_for_round(
                circuit=circuit,
                distance=distance,
                num_round=num_round,
                final_data=final_data,
                round_idx=r,
                cmap_name="white_to_red",      # or "white_to_rosered" or "Reds"
            )


if __name__ == "__main__":
    main()
