import stim
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from matplotlib.patches import Arc, Wedge, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


# ----------------- geometry / layout helpers ----------------- #

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


# ----------------- plotting the layout + heatmap ----------------- #

def plot_surface_code_heatmap_layout(
    circuit: stim.Circuit,
    attributions: dict[int, float],
    title: str = "Global attribution heatmap",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name="viridis",        # can be string OR colormap object
):
    """
    Draw rotated surface code layout with heatmap colouring of ancilla plaquettes.

    attributions: dict[ancilla_qubit_id -> float]
    vmin, vmax: color scale limits
    cmap_name: string name of cmap or actual matplotlib colormap
    """
    coords = get_qubit_coords(circuit)
    data_qubits, ancilla_qubits = classify_qubits(circuit)

    data_pts = {q: coords[q] for q in coords if q in data_qubits}
    anc_pts = {q: coords[q] for q in coords if q in ancilla_qubits}

    fig, ax = plt.subplots(figsize=(7, 7))

    # ---- choose colormap based on cmap_name OR actual colormap ----
    if isinstance(cmap_name, str):
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
            cmap = plt.get_cmap(cmap_name)  # any standard matplotlib cmap by name
    else:
        # Assume user passed a colormap object
        cmap = cmap_name

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

    # ---- 3) fill plaquettes according to attributions ----
    vals = []
    for q, (x, y) in anc_pts.items():
        v = float(attributions.get(q, 0.0))
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

    # ---- labels on ancilla plaquettes ----
    for q, (x, y) in anc_pts.items():
        v = float(attributions.get(q, 0.0))

        tx, ty = x, y  # you can offset if you like

        ax.text(
            tx,
            ty + 0.5,
            f"{v:.2f}",          # format to 2 decimal places
            ha="center",
            va="top",
            fontsize=7,
            color="black",
            zorder=20,
            clip_on=False,
        )

    # ---- ancilla markers (orange circles + labels) ----
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

    # ---- data qubits (blue squares + labels) ----
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
    cbar.set_label("Attribution")

    plt.show()


# ----------------- main wrapper: normalization + cmap option ----------------- #

def visualize_heatmap(
    distance: int,
    attribution: np.ndarray,
    title_prefix: str = "Rotated surface code : Global attribution heatmap",
    cmap_name="viridis",
    normalize: bool = True,
):
    """
    Plot a single global attribution heatmap.

    attribution: 1D array of length num_ancilla
    normalize: if True, scale to [0,1], else use raw values
    cmap_name: string or colormap object passed through to plot_surface_code_heatmap_layout
    """

    num_ancilla = distance**2 - 1

    attribution = np.asarray(attribution).flatten()
    if attribution.shape[0] != num_ancilla:
        raise ValueError(
            f"Expected attribution of length {num_ancilla} "
            f"for distance={distance}, got shape {attribution.shape}"
        )

    # --------- NORMALIZATION OPTION ---------
    if normalize:
        a_min = attribution.min()
        a_max = attribution.max()

        if a_max - a_min < 1e-12:
            # constant array → make everything zero
            norm_attr = np.zeros_like(attribution, dtype=float)
        else:
            norm_attr = (attribution - a_min) / (a_max - a_min)

        attr_to_use = norm_attr
        vmin, vmax = 0.0, 1.0

    else:
        # Use raw attribution values
        attr_to_use = attribution
        vmin = float(attribution.min())
        vmax = float(attribution.max())

    # --------- Build circuit layout ---------
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
    )

    data_qubits, ancilla_qubits = classify_qubits(circuit)
    ancilla_order = get_ancilla_order(circuit, ancilla_qubits)

    # Map ancilla → attribution value
    attributions_dict = {
        q: float(attr_to_use[k])
        for k, q in enumerate(ancilla_order)
    }

    title = f"{title_prefix}"

    plot_surface_code_heatmap_layout(
        circuit=circuit,
        attributions=attributions_dict,
        title=title,
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
    )


# ----------------- main ----------------- #

def main():
    distance = 5
    attr = np.load(
        r"C:/Research_Chaitali/phy191a/global_attribution.npy",
        allow_pickle=True,
    )

    print("global_attribution shape:", attr.shape)

    visualize_heatmap(
        distance=distance,
        attribution=attr,
        normalize=True,      # or False to use raw values
        cmap_name="viridis", # or "white_to_red", "white_to_rosered", "Reds", etc.
    )


if __name__ == "__main__":
    main()
