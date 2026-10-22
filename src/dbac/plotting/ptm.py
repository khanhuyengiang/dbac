"""
ptm.py

Visualization utilities for quantum process matrices, either as a
Pauli-transfer matrix (PTM) heatmap (`plot_ptm`, `plot_ptm_sweep`) or using
color to encode phase (hue) and magnitude (size) of each matrix element
(`plot_process_matrix`).

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
    "plot_process_matrix",
    "plot_ptm",
    "plot_ptm_sweep",
]

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
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


def plot_ptm(ptm, labels, *, ax=None, title=None, cmap='RdBu_r', vmin=-1, vmax=1):
    """
    Plot the real part of a Pauli-transfer matrix (PTM) as a heatmap.

    Parameters:
        ptm (np.ndarray): The (real or complex) Pauli-transfer matrix.
        labels (list[str]): Tick labels for both axes (e.g. `pauli_labels_2q`).
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates one if None.
        title (str, optional): Title of the plot.
        cmap (str): Colormap for the heatmap.
        vmin, vmax (float): Color scale limits.

    Returns:
        (fig, ax, image): The figure, axis, and the `imshow` artist (useful for
        attaching a shared colorbar when plotting several PTMs side by side).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    image = ax.imshow(np.real(ptm), cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title, fontsize=14)

    return fig, ax, image


def plot_ptm_sweep(ptms, titles, labels, *, figsize=(18, 6), colorbar_label="Real PTM",
                    save_path=None):
    """
    Plot several PTMs side by side (e.g. a sweep over a rotation angle), sharing
    a single colorbar.

    Parameters:
        ptms (sequence[np.ndarray]): One Pauli-transfer matrix per panel.
        titles (sequence[str]): Title for each panel.
        labels (list[str]): Shared tick labels (e.g. `pauli_labels_2q`).
        figsize (tuple): Size of the overall figure.
        colorbar_label (str): Label of the shared colorbar.
        save_path (str, optional): If given, saves the figure to this path.

    Returns:
        (fig, axes): The figure and the array of axes.
    """
    fig, axes = plt.subplots(1, len(ptms), figsize=figsize)
    axes = np.atleast_1d(axes)

    image = None
    for ax, ptm, title in zip(axes, ptms, titles):
        _, _, image = plot_ptm(ptm, labels, ax=ax, title=title)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02).set_label(
        colorbar_label, fontsize=14)

    if save_path is not None:
        fig.savefig(save_path)

    return fig, axes
