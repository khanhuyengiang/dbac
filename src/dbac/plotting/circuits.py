"""
circuits.py

Matplotlib drawings of the DBAC experiment circuits (Fig. 4 of the paper), so
the documentation can show them without a qiskit/LaTeX-qcircuit toolchain.

A circuit is described by a small dict (see `DBAC_CIRCUITS`): the number of
qubits, which one is the data qubit, vertical "barriers" between recursion
steps, and a list of operations placed in columns.
"""

__all__ = ["DBAC_CIRCUITS", "draw_circuit", "draw_dbac_circuit"]

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_RX = r'$\mathrm{R}_\mathrm{X}(\theta)$'
_RZ = r'$\mathrm{R}_\mathrm{Z}(\pi/4)$'


def _dme(angle):
    return rf'$U_\mathrm{{DME}}({angle})$'


# Each op is (column, qubits, label); a gate spanning two qubits is drawn as one tall box.
# `data` is the (0-based) qubit whose energy is read out; `barriers` lists the columns
# after which a dashed line separates the DBAC steps.
DBAC_CIRCUITS = {
    "A": {  # k = 1, M = 1, n = 2
        "n": 2, "data": 0, "barriers": [0],
        "ops": [(0, [0], _RX), (0, [1], _RX),
                (1, [1], _RZ),
                (2, [0, 1], _dme(r'\pi/4'))],
    },
    "B": {  # k = 1, M = 2, n = 3
        "n": 3, "data": 1, "barriers": [0],
        "ops": [(0, [0], _RX), (0, [1], _RX), (0, [2], _RX),
                (1, [0], _RZ), (1, [2], _RZ),
                (2, [0, 1], _dme(r'\pi/8')),
                (3, [1, 2], _dme(r'\pi/8'))],
    },
    "C": {  # k = 2, M = 1, n = 4
        "n": 4, "data": 1, "barriers": [0, 2],
        "ops": [(0, [0], _RX), (0, [1], _RX), (0, [2], _RX), (0, [3], _RX),
                (1, [0], _RZ), (1, [3], _RZ),
                (2, [0, 1], _dme(r'\pi/4')), (2, [2, 3], _dme(r'\pi/4')),
                (3, [2], _RZ),
                (4, [1, 2], _dme(r'\pi/4'))],
    },
}

_ROW = 0.9      # vertical distance between qubit wires
_GAP = 0.35     # horizontal gap between columns
_BOX_H = 0.55   # height of a single-qubit box


def draw_circuit(spec, *, ax=None, data_label=True, fontsize=9):
    """
    Draw one circuit described by `spec` (an entry of `DBAC_CIRCUITS`).

    Parameters:
        spec (dict): Circuit description, see the module docstring.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; creates a figure if None.
        data_label (bool): Mark the data qubit at the right-hand end of its wire.
        fontsize (float): Font size of the gate labels.

    Returns:
        matplotlib.axes.Axes
    """
    n = spec["n"]
    n_cols = max(col for col, _, _ in spec["ops"]) + 1
    col_width = [0.0] * n_cols
    for col, qubits, label in spec["ops"]:
        width = 1.9 if 'DME' in label else 1.15
        col_width[col] = max(col_width[col], width)

    left, x = [], 0.0
    for w in col_width:
        left.append(x)
        x += w + _GAP
    x_end = x - _GAP + 0.3
    y = [-i * _ROW for i in range(n)]

    if ax is None:
        fig = plt.figure(figsize=(0.62 * x_end + 0.8, 0.5 * n + 0.2))
        fig.set_layout_engine('none')
        ax = fig.add_axes([0, 0, 1, 1])

    for i in range(n):
        ax.plot([-0.3, x_end], [y[i], y[i]], color='k', lw=0.8, zorder=0)
        ax.text(-0.45, y[i], rf'$q_{i + 1}:$', ha='right', va='center', fontsize=fontsize + 1)
    if data_label:
        ax.text(x_end + 0.15, y[spec["data"]], 'data qubit', ha='left', va='center',
                fontsize=fontsize - 1, style='italic', color='0.35')

    for col, qubits, label in spec["ops"]:
        y_top, y_bot = y[min(qubits)] + _BOX_H / 2, y[max(qubits)] - _BOX_H / 2
        ax.add_patch(Rectangle((left[col], y_bot), col_width[col], y_top - y_bot,
                               fc='white', ec='k', lw=0.8, zorder=2))
        ax.text(left[col] + col_width[col] / 2, (y_top + y_bot) / 2, label, ha='center',
                va='center', fontsize=fontsize, zorder=3)

    for col in spec["barriers"]:
        xb = left[col] + col_width[col] + _GAP / 2
        ax.plot([xb, xb], [y[0] + 0.55, y[-1] - 0.55], color='k', lw=0.8, ls=(0, (3, 3)), zorder=1)

    ax.set_xlim(-1.3, x_end + (1.6 if data_label else 0.3))
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def draw_dbac_circuit(name, **kwargs):
    """Draw circuit "A", "B" or "C" of the paper's Fig. 4. See `draw_circuit`."""
    return draw_circuit(DBAC_CIRCUITS[name], **kwargs)
