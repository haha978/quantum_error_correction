import stim
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from matplotlib.patches import Arc


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

def plot_surface_code_layout(circuit: stim.Circuit, title="Rotated surface code"):
    coords = get_qubit_coords(circuit)
    data_qubits, ancilla_qubits = classify_qubits(circuit)

    # Dicts of qubit -> (x, y)
    data_pts = {q: coords[q] for q in coords if q in data_qubits}
    anc_pts  = {q: coords[q] for q in coords if q in ancilla_qubits}

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- 1) Straight edges between neighboring data qubits ---
    for q1, q2 in itertools.combinations(data_pts, 2):
        p1, p2 = data_pts[q1], data_pts[q2]
        if _is_data_neighbor(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1.2, alpha=0.7)

    # --- 2) Semicircles on boundary edges (through boundary ancillas) ---
    for aq, (ax_, ay_) in anc_pts.items():
        # Find data neighbors of this ancilla
        neighbors = [dq for dq, p in data_pts.items()
                     if _is_data_neighbor((ax_, ay_), p)]
        if len(neighbors) != 2:
            continue  # interior (4 neighbors) or odd case, skip

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

        # Draw arc only (no sector fill)
        arc = Arc(
            (cx, cy),
            2 * r, 2 * r,
            angle=0,
            theta1=start_deg,
            theta2=end_deg,
            color="gray",
            linewidth=1.5,
            linestyle="-",
            alpha=0.9,
        )
        ax.add_patch(arc)

    # --- 3) Plot ancilla qubits (circles) ---
    for q, (x, y) in anc_pts.items():
        ax.scatter([x], [y], marker="o", s=140,
                   facecolors="none", edgecolors="tab:orange",
                   linewidths=2, zorder=3)
        ax.text(x, y - 0.07, str(q), ha="center", va="top",
                fontsize=8, color="black", zorder=4)

    # --- 4) Plot data qubits (squares) ---
    for q, (x, y) in data_pts.items():
        ax.scatter([x], [y], marker="s", s=180,
                   facecolors="tab:blue", edgecolors="navy",
                   linewidths=1.5, zorder=5)
        ax.text(x, y + 0.12, str(q), ha="center", va="bottom",
                fontsize=8, color="white", zorder=6)

    # --- 5) Formatting ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='tab:blue', markeredgecolor='black',
                   markersize=10, label='Data qubits'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='none', markeredgecolor='tab:orange',
                   markersize=10, label='Ancilla qubits'),
    ])
    ax.invert_yaxis()  # match Stim's orientation
    plt.tight_layout()
    plt.show()



def main():
    distance = 5                 # code distance
    rounds = 2                  # QEC cycles (syndrome rounds)
    num_shots = 1           # number of samples for dataset / decoding
    print(f"d={distance}, rounds={rounds}, shots={num_shots}")
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
    )
    plot_surface_code_layout(circuit, title="Rotated surface code : Internal layout")

if __name__ == "__main__":
    main()