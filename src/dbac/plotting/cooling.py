"""
cooling.py

Plotting for the DB-QITE / DBAC / HBAC cooling-performance figures: fidelity
of the output state with the ground state as a function of the fidelity (or
angle) of the input state. These functions generalize plotting code that used
to be copy-pasted, with small variations, into every DBAC notebook.
"""

__all__ = [
    "DEFAULT_COLORS",
    "DEFAULT_MARKERS",
    "plot_fidelity_fidelity",
    "plot_energy_vs_theta",
    "plot_single_energy_curve",
    "plot_energy_grid",
    "plot_optimal_s_vs_M",
    "plot_max_excitation_vs_k",
    "plot_ideal_dme_summary",
    "plot_hbac_fidelity_fidelity",
]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dbac.gci import max_excitation as _max_excitation

# Shared default styling so every figure uses the same palette.
DEFAULT_COLORS = ['#2c0084', '#7a0085', '#006f7a', 'C0']
DEFAULT_MARKERS = ['o', 'x', '^', '*', 'P', '*', 'h']


def _percent(fraction):
    """Format a fraction as e.g. '90.0%', escaping the % when LaTeX renders text."""
    sign = r'\%' if plt.rcParams['text.usetex'] else '%'
    return f'{fraction * 100:.1f}{sign}'


def _annotate_poles(ax, x, y, dx=(0.25, 0.04), dy=(0.03, -0.22)):
    ax.annotate(r'$|1\rangle\langle 1|$', xy=(x[0], y[0]),
                xytext=(x[0] + dx[0], y[0] + dy[0]), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate(r'$|0\rangle\langle 0|$', xy=(x[-1], y[-1]),
                xytext=(x[-1] + dx[1], y[-1] + dy[1]), arrowprops=dict(arrowstyle='->', lw=1.5))


def plot_fidelity_fidelity(fidelities_initial, initial_curve, results, *, colors=None,
                            markers=None, target_fidelity=None, mark_max_excitation=False,
                            xlabel='Fidelity of initial state\nwith ground state',
                            ylabel='Fidelity of final state\nwith ground state',
                            annotate_poles=True, markersize=6, figsize=(5.5, 4), ax=None,
                            save_path=None):
    """
    "Fidelity-fidelity" plot: final-state fidelity with the ground state
    against initial-state fidelity, for one or more DB-QITE/HBAC
    configurations. This is the main result plot of the DBAC paper (Fig. 4).

    Parameters:
        fidelities_initial (sequence[float]): Fidelity of each initial state with |0>.
        initial_curve (sequence[float]): Baseline "no cooling" fidelity to plot
            as a dotted reference line (equal to `fidelities_initial` for a
            noiseless initial state).
        results (dict[str, sequence[float]]): label -> final-state fidelities,
            one curve per circuit/configuration, in the order to be styled.
        colors, markers (sequence, optional): Styling cycled over `results`.
        target_fidelity (float, optional): Draws a dashed horizontal reference line.
        mark_max_excitation (bool): If True (requires `target_fidelity`), draws
            for each curve a vertical dashed line at the largest
            `fidelities_initial` for which the final fidelity still reaches
            `target_fidelity`, and returns those positions.
        annotate_poles (bool): Annotate the |1><1| and |0><0| corners.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; creates one if None.
        save_path (str, optional): If given, saves the figure.

    Returns:
        matplotlib.axes.Axes, or (Axes, list[float | None]) if `mark_max_excitation`.
    """
    colors = colors or DEFAULT_COLORS
    markers = markers or DEFAULT_MARKERS
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(fidelities_initial, initial_curve, label="Initial state", alpha=0.5, linestyle=':')

    max_ext_list = []
    last_curve = initial_curve
    for i, (label, curve) in enumerate(results.items()):
        if mark_max_excitation and target_fidelity is not None:
            max_ext = _max_excitation(fidelities_initial, np.real(curve), target_fidelity)
            if max_ext is not None:
                ax.axvline(x=max_ext, ymax=0.2, color=colors[i % len(colors)], alpha=0.8, linestyle='--')
            max_ext_list.append(max_ext)
        ax.plot(fidelities_initial, curve, label=label, color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], markevery=5, markersize=markersize, alpha=0.9)
        last_curve = curve

    ax.invert_xaxis()
    ax.invert_yaxis()
    if target_fidelity is not None:
        ax.axhline(y=target_fidelity, color='k', alpha=0.6, linestyle='--',
                    label=f'Target F = {_percent(target_fidelity)}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if annotate_poles:
        _annotate_poles(ax, fidelities_initial, last_curve)

    if created_fig:
        plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    return (ax, max_ext_list) if mark_max_excitation else ax


def plot_energy_vs_theta(theta_angles, initial_energy, results, *, colors=None, markers=None,
                          target_energy=None, xlabel=r'$\theta,$ Angle of initialisation',
                          ylabel='Data qubit energy', xticks=None, xtick_labels=None,
                          annotate_poles=True, ax=None, save_path=None):
    """
    Plot the data-qubit energy after DB-QITE against the initialization angle
    theta, for one or more circuits/configurations.

    Parameters:
        theta_angles (sequence[float]): Initialization angle (radians) per sample.
        initial_energy (sequence[float]): Energy of the un-cooled initial state.
        results (dict[str, sequence[float]]): label -> final energy curve.
        colors, markers (sequence, optional): Styling cycled over `results`.
        target_energy (float, optional): Draws a dashed horizontal reference line.
        xticks, xtick_labels (sequence, optional): Custom tick marks (e.g. in units of pi).
        annotate_poles (bool): Annotate the |1><1| and |0><0| corners.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; creates one if None.
        save_path (str, optional): If given, saves the figure.

    Returns:
        matplotlib.axes.Axes
    """
    colors = colors or DEFAULT_COLORS
    markers = markers or DEFAULT_MARKERS
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(theta_angles, initial_energy, label="Initial state", alpha=0.5, linestyle=':')
    for i, (label, curve) in enumerate(results.items()):
        ax.plot(theta_angles, curve, label=label, color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], markevery=5, markersize=6, alpha=0.9)

    if target_energy is not None:
        ax.axhline(y=target_energy, color='k', alpha=0.6, linestyle='--')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
    ax.legend()
    if annotate_poles:
        ax.annotate(r'$|1\rangle\langle 1|$', xy=(theta_angles[-1], initial_energy[-1]),
                     xytext=(theta_angles[-1] - 0.13, initial_energy[-1] - 0.3),
                     arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate(r'$|0\rangle\langle 0|$', xy=(theta_angles[0], initial_energy[0]),
                     xytext=(theta_angles[0], initial_energy[0] + 0.2),
                     arrowprops=dict(arrowstyle='->', lw=1.5))

    if created_fig:
        plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    return ax


def plot_single_energy_curve(theta_angles, initial_energy, label, curve, color, marker, *,
                              experimental=None, xticks=None, xtick_labels=None,
                              annotate_poles=True, figsize=None, save_path=None, show=False):
    """
    Plot a single DB-QITE energy-vs-theta curve against the un-cooled baseline,
    optionally overlaid with experimental data points.

    Parameters:
        theta_angles (sequence[float]): Initialization angle (radians) per sample.
        initial_energy (sequence[float]): Energy of the un-cooled initial state.
        label (str): Legend label for the analytical curve.
        curve (sequence[float]): Analytical final-energy curve.
        color, marker: Styling for the analytical curve / experimental markers.
        experimental (tuple[sequence, sequence], optional): (theta, energy)
            experimental data points to overlay as open markers.
        xticks, xtick_labels (sequence, optional): Custom tick marks.
        annotate_poles (bool): Annotate the |1><1| and |0><0| corners.
        save_path (str, optional): If given, saves the figure.
        show (bool): If True, calls plt.show() (useful in a notebook).

    Returns:
        matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(theta_angles, initial_energy, label="Initial state", color='C0', linestyle=':', alpha=0.6)
    ax.plot(theta_angles, curve, label='Analytical', color=color, alpha=0.9)

    ax.set_xlabel(r'$\theta,$ Angle of initialisation')
    ax.set_ylabel('Data qubit energy')
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

    if annotate_poles:
        ax.annotate(r'$|1\rangle\langle 1|$', xy=(theta_angles[-1], initial_energy[-1]),
                     xytext=(theta_angles[-1] - 0.17, initial_energy[-1] - 0.6),
                     arrowprops=dict(arrowstyle='->', lw=1.5), zorder=10)
        ax.annotate(r'$|0\rangle\langle 0|$', xy=(theta_angles[0], initial_energy[0]),
                     xytext=(theta_angles[0], initial_energy[0] + 0.5),
                     arrowprops=dict(arrowstyle='->', lw=1.5), zorder=10)

    if experimental is not None:
        theta_exp, energy_exp = experimental
        ax.scatter(theta_exp, energy_exp, label='Experiment', marker=marker, color=color,
                   zorder=4, facecolor='none')

    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, transparent=False)
    if show:
        plt.show()
        plt.close(fig)

    return ax


def plot_energy_grid(theta_angles, initial_energy, plot_dict, plot_map, *, colors=None,
                      markers=None, target_energy=None, nrows=2, ncols=2, figsize=None,
                      titles=None, save_path=None):
    """
    Lay out several `plot_single_energy_curve`-style panels on a grid, e.g. one
    panel per number of quantum instructions `n` used in DB-QITE.

    Parameters:
        theta_angles (sequence[float]): Initialization angle (radians) per sample.
        initial_energy (sequence[float]): Energy of the un-cooled initial state.
        plot_dict (dict[str, sequence[float]]): label -> final energy curve.
        plot_map (dict[int, str]): Maps a flattened subplot index to a label in
            `plot_dict`; indices not present are left blank.
        colors, markers (sequence, optional): Styling cycled over `plot_map`.
        target_energy (float, optional): Draws a dashed horizontal reference line.
        nrows, ncols (int): Grid shape.
        titles (dict[int, str], optional): Title per subplot index.
        save_path (str, optional): If given, saves the figure.

    Returns:
        (fig, axes)
    """
    colors = colors or DEFAULT_COLORS
    markers = markers or DEFAULT_MARKERS

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i, ax in enumerate(axs):
        if i not in plot_map:
            ax.axis('off')
            continue

        label = plot_map[i]
        curve = plot_dict[label]
        idx = list(plot_map.keys()).index(i)

        ax.plot(theta_angles, initial_energy, label=r"Initial state", color=colors[-1],
                 linestyle=':', alpha=0.6)
        ax.plot(theta_angles, curve, label=label, color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)], markerfacecolor='none', markevery=9, markersize=6)

        if target_energy is not None:
            ax.axhline(y=target_energy, color='k', alpha=0.6, linestyle='--')

        ax.annotate(r'$|1\rangle\langle 1|$', xy=(theta_angles[-1], initial_energy[-1]),
                     xytext=(theta_angles[-1] - 0.17, initial_energy[-1] - 0.6),
                     arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate(r'$|0\rangle\langle 0|$', xy=(theta_angles[0], initial_energy[0]),
                     xytext=(theta_angles[0], initial_energy[0] + 0.5),
                     arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.set_box_aspect(1)
        if titles and i in titles:
            ax.set_title(titles[i])

    fig.supxlabel(r'$\theta$, Angle of initialization')
    fig.supylabel('Data qubit energy')

    if save_path is not None:
        fig.savefig(save_path)

    return fig, axs


def plot_optimal_s_vs_M(optimal_s_kM, *, colors=None, markers=None, markersize=6, ax=None):
    """
    Plot the optimized DB-QITE step size s against the number of DME Trotter
    steps M, one curve per recursion depth k.

    Parameters:
        optimal_s_kM (np.ndarray): Shape (k_max, M_max) array of optimal s
            values, as produced by transposing `optimal_s_Mk`.
        colors, markers (sequence, optional): Styling cycled over k.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; creates one if None.

    Returns:
        matplotlib.axes.Axes
    """
    from matplotlib import cm
    if colors is None:
        n_colors = len(optimal_s_kM)
        colors = [(r * 0.8, g * 0.8, b * 0.8, a) for r, g, b, a in cm.rainbow(np.linspace(1, 0, n_colors))]
    markers = markers or DEFAULT_MARKERS

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots()

    max_M = optimal_s_kM.shape[1]
    for i, s_list in enumerate(optimal_s_kM):
        ax.plot(range(1, max_M + 1), s_list, color=colors[i % len(colors)], label=f'k={i + 1}',
                 marker=markers[i % len(markers)], markersize=markersize)

    ax.set_ylabel(r'Optimal step size $s$')
    ax.set_xlabel(r'$M$')

    return ax


def plot_max_excitation_vs_k(max_ext_list, *, target_fidelity=0.9, colors=None, markers=None,
                              markersize=6, ax=None):
    """
    Plot "how far from the ground state the initial state can still be while
    reaching the target fidelity" (1 - max-excitation fidelity) against the
    DB-QITE recursion depth k.

    Parameters:
        max_ext_list (sequence[float]): Max-excitation fidelity per k (output
            of `plot_fidelity_fidelity(..., mark_max_excitation=True)`).
        target_fidelity (float): The target fidelity `max_ext_list` was computed
            for, shown in the y-axis label.
        colors, markers (sequence, optional): Styling cycled over k.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; creates one if None.

    Returns:
        matplotlib.axes.Axes
    """
    colors = colors or DEFAULT_COLORS
    markers = markers or DEFAULT_MARKERS

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots()

    x_vals = range(1, len(max_ext_list) + 1)
    y_vals = [1 - max_ext for max_ext in max_ext_list]

    ax.plot(x_vals, y_vals, color='k', alpha=0.6, linestyle='--')
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        ax.plot(x, y, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                 markersize=markersize, linestyle='None')

    ax.set_xlabel(r'Number of DB-QITE step $k$')
    ax.set_ylabel(f'Initial fidelity can be\nreset to F = {_percent(target_fidelity)}')

    return ax


def plot_ideal_dme_summary(fidelities_initial, initial_curve, dict_fid, optimal_s_kM, *,
                            target_fidelity, colors=None, markers=None, save_path=None):
    """
    Three-panel summary figure for the ideal-DME DBAC recursion: (a) the
    fidelity-fidelity plot across recursion depths k, (b) how the
    max-excitation fidelity improves with k, and (c) the optimized step size s
    against the number of Trotter steps M. This reproduces the DBAC paper's
    "ideal DME" rainbow-colored figure.

    Parameters:
        fidelities_initial (sequence[float]): Fidelity of each initial state with |0>.
        initial_curve (sequence[float]): Baseline "no cooling" fidelity curve.
        dict_fid (dict[str, sequence[float]]): label (e.g. 'k=1', 'k=2', ...)
            -> final-state fidelity curve.
        optimal_s_kM (np.ndarray): Shape (k_max, M_max) optimal step sizes.
        target_fidelity (float): Target fidelity used to mark max-excitation points.
        colors, markers (sequence, optional): Styling cycled over `dict_fid`.
        save_path (str, optional): If given, saves the figure.

    Returns:
        matplotlib.figure.Figure
    """
    # Constrained layout keeps every axis label inside the figure.
    fig = plt.figure(layout='constrained')
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[:, 0])
    _, max_ext_list = plot_fidelity_fidelity(
        fidelities_initial, initial_curve, dict_fid, colors=colors, markers=markers,
        target_fidelity=target_fidelity, mark_max_excitation=True, markersize=8, ax=ax1)
    ax1.set_xticks([1.0, 0.75, 0.5, 0.25, 0.0])

    ax2 = fig.add_subplot(gs[0, 1])
    plot_max_excitation_vs_k([m for m in max_ext_list if m is not None],
                              target_fidelity=target_fidelity, colors=colors,
                              markers=markers, markersize=10, ax=ax2)
    ax2.set_xlabel(r'$k$')
    ax2.set_xticks(range(2, len(dict_fid) + 1, 2))

    ax3 = fig.add_subplot(gs[1, 1])
    plot_optimal_s_vs_M(optimal_s_kM[:-1], colors=colors, markers=markers, markersize=8, ax=ax3)
    ax3.set_xlabel(r'$M$')

    # Panel letters: a) at the top-left corner, b) and c) at the bottom-left.
    ax1.text(-0.3, 1.0, 'a)', transform=ax1.transAxes, ha='left', va='top', fontsize='large')
    for ax, letter in ((ax2, 'b)'), (ax3, 'c)')):
        ax.text(-0.3, -0.1, letter, transform=ax.transAxes, ha='left', va='top', fontsize='large')

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def plot_hbac_fidelity_fidelity(fidelities_initial, baselines, results, *, markevery=4,
                                 title=None, save_path=None):
    """
    Fidelity-fidelity plot comparing HBAC on pure vs. depolarized ("mixed")
    input states, for different numbers of cooling rounds.

    Parameters:
        fidelities_initial (sequence[float]): Fidelity of each initial (pure) state with |0>.
        baselines (dict[str, sequence[float]]): label -> dotted reference curve
            (e.g. 'Initial pure state', 'Initial mixed state').
        results (dict[str, sequence[float]]): label -> HBAC output fidelity curve.
        markevery (int): Marker spacing for the result curves.
        title (str, optional): Plot title.
        save_path (str, optional): If given, saves the figure.

    Returns:
        matplotlib.axes.Axes
    """
    colors = DEFAULT_COLORS
    markers = ['o', 'x', '^', 'D']

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, curve in baselines.items():
        ax.plot(fidelities_initial, curve, label=label, alpha=0.5, linestyle=':')

    for i, (label, curve) in enumerate(results.items()):
        ax.plot(fidelities_initial, curve, label=label, color=colors[i % len(colors)],
                 marker=markers[i % len(markers)], markevery=markevery, markerfacecolor='None')

    ax.set_xlabel('Fidelity of initial state\nwith ground state')
    ax.set_ylabel('Fidelity of final state\nwith ground state')
    ax.legend(fontsize=10)
    if title is not None:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.invert_xaxis()

    if save_path is not None:
        plt.savefig(save_path)

    return ax
