"""
gci.py

Plotting for the raw output of `dbac.gci.GCI.execute_gci`: energy or
fidelity versus the total DB-QITE evolution duration. This used to be a method
on the `GCI` class (`GCI.plot_GCI_results`); it is a plain function here so
that `dbac.gci` has no matplotlib dependency.
"""

__all__ = ["plot_gci_results"]

import matplotlib.pyplot as plt


def plot_gci_results(*results, target_energy, mode='Default', save_path=None):
    """
    Plot energy or fidelity vs evolution duration for one or more GCI runs.

    Parameters:
        *results: Output tuples from `GCI.execute_gci()`.
        target_energy (float): The energy of the target (ground) state, used to
            draw a horizontal reference line in 'Default' mode.
        mode (str): 'Default' for energy vs duration, 'Fidelity' for fidelity
            vs duration.
        save_path (str, optional): If given, saves the figure to this path.
    """
    for result in results:
        energies, _, _, durations, label, marker, _, _, _, _, fidelities = result
        y = energies if mode == 'Default' else fidelities
        if marker == '-':
            plt.plot(durations, y, label=label, linestyle=marker)
        else:
            plt.plot(durations, y, label=label, marker=marker)

    last_duration = results[0][3][-1]
    if mode == 'Default':
        plt.ylabel('Energy')
        plt.axline((0, target_energy), (last_duration, target_energy), linestyle=":")
    elif mode == 'Fidelity':
        plt.ylabel('Fidelity')
        plt.axline((0, 1), (last_duration, 1), linestyle=":")

    plt.xlabel('Evolution duration')
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
