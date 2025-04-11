"""
plot_process_matrix.py

Visualization utilities for quantum process matrices using color to encode 
phase (hue) and magnitude (size) of matrix elements. This is useful for 
analyzing quantum channels or gates represented in the Pauli basis.

Functions:
- plot_process_matrix: Visualizes a complex-valued process matrix in the Pauli basis 
  using hue for phase and size for magnitude.

Globals:
- pauli_labels: Labels for single-qubit Pauli basis.
- pauli_labels_2q: Labels for two-qubit Pauli basis.

Dependencies:
- numpy
- matplotlib
"""

__all__ = [
    "pauli_labels",
    "pauli_labels_2q",
    "plot_process_matrix"
]

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.patches as mpatches

pauli_labels = ["I", "X", "Y", "Z"]
pauli_labels_2q = ['II', 'IX', 'IY', 'IZ',
 'XI', 'XX', 'XY', 'XZ',
 'YI', 'YX', 'YY', 'YZ',
 'ZI', 'ZX', 'ZY', 'ZZ']

def plot_process_matrix(m: np.ndarray, basis_name: list[str], ax=None, title=None, base: int = 2):
    """
    Plots a process matrix using color-coded phases and magnitudes.

    Parameters:
        m (np.ndarray): The process matrix to plot.
        basis_name (list[str]): List of Pauli basis labels.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates one if None.
        title (str, optional): Title of the plot.
        base (int): Base for computing qubit number from matrix size.
    """
    nx, ny = m.shape
    qubit_number = int(np.round(np.log(nx) / np.log(base ** 2)))

    do_adjust = False
    if ax is None:
        do_adjust = True
        fig, ax = plt.subplots()
    ax.axis("scaled")
    ax.patch.set_facecolor("white")

    mflat = m.flatten()
    hues = np.arctan2(-mflat.imag, -mflat.real) / (2.0 * np.pi) + 0.5
    colors = hsv_to_rgb(np.dstack((hues, np.ones_like(hues), np.ones_like(hues))))
    sizes = np.abs(mflat) ** 0.5

    i = 0
    for ix in range(nx):
        for iy in range(ny):
            ax.add_artist(
                mpatches.RegularPolygon(
                    xy=(ix, iy),
                    numVertices=4,
                    orientation=np.pi / 4,
                    color=colors[0, i],
                    radius=2.0 ** -0.5 * sizes[i],
                    linestyle="None",
                    linewidth=0.0,
                )
            )
            i += 1

    ax.set_xticks(np.arange(nx))
    ax.set_yticks(np.arange(ny))
    ax.set_xticklabels(basis_name)
    ax.set_yticklabels(basis_name)
    ax.axis([-0.5, nx - 0.5, ny - 0.5, -0.5])

    for iy in range(ny + 1):
        ax.axhline(y=iy - 0.5, ls=":", color="k")
    for ix in range(nx + 1):
        ax.axvline(x=ix - 0.5, ls=":", color="k")

    if title is not None:
        ax.set_title(title)

    if do_adjust:
        legendhandles = []
        leghues = np.linspace(0.0, 1.0, 17)
        legendcolors = hsv_to_rgb(np.dstack((leghues, np.ones(17), np.ones(17))))
        for i in range(17):
            legendhandles.append(mpatches.Patch(color=legendcolors[0, i]))
        fig.subplots_adjust(left=0.05, bottom=0.1, top=0.9, right=0.85)
        fig.legend(
            legendhandles,
            ["+1", "", "", "", "+i", "", "", "", "-1", "", "", "", "-i", "", "", "", "+1"],
            loc="right",
            bbox_to_anchor=(1.0, 0.5),
        )
        plt.show()
