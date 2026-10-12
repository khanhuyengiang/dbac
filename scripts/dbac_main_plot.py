"""
scripts/dbac_main_plot.py

Main DBAC result (Fig. 4 of the paper): DB-QITE cooling performance using a
Trotterized DME subroutine, for circuits using n = 2, 3, 4 total qubits
(n - 1 quantum "instructions" plus the data qubit). Produces:

- a fidelity-fidelity plot comparing circuits A/B/C (n=2,3,4),
- the corresponding data-qubit energy vs. initialization angle theta,
- the same energy curves overlaid with hardware experiment data, and
- a 2x2 grid layout of the energy-vs-theta panels.

Run directly to regenerate the figures under `jupyter_notebooks/plots/`.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from functions.qite_gci import GCI
from functions.qite_dme import QITE_dme_select_M, initial_psi
from functions.utils import fidelity_with_0, rx_sweep
from visualization import (
    use_paper_style, figure_size,
    plot_fidelity_fidelity, plot_energy_vs_theta, plot_single_energy_curve, plot_energy_grid,
)

OPTIMAL_S_PATH = REPO_ROOT / "jupyter_notebooks" / "optimal_s_Mk.npy"
PLOTS_DIR = REPO_ROOT / "jupyter_notebooks" / "plots"

# Circuit A (n=2, k=1), B (n=3, k=1, M=2), C (n=4, k=2, M=1): label -> (M, k)
CIRCUITS = {
    r'$n=2$': (1, 1),
    r'$n=3$': (2, 1),
    r'$n=4$': (1, 2),
}

# Hardware calibration-experiment data points (theta, energy), digitized from
# the device runs reported in the paper, used to validate the simulation.
EXPERIMENTAL_ENERGY = {
    r'$n=3$': (
        [0, 0.195, 0.312, 0.445, 0.62, 0.777, 0.928, 1.105, 1.237, 1.352, 1.546],
        [-1, -1.001, -0.99, -0.91, -0.707, -0.439, -0.095, 0.166, 0.453, 0.737, 0.922],
    ),
    r'$n=2$': (
        [1.5708, 1.37, 1.255, 1.123, 0.941, 0.79, 0.631, 0.45, 0.32, 0.206, 0],
        [1, 0.89, 0.823, 0.648, 0.375, 0.075, -0.204, -0.44, -0.64, -0.797, -1],
    ),
    r'$n=4$': (
        [0, 0.204, 0.321, 0.453, 0.631, 0.787, 0.939, 1.119, 1.249, 1.367, np.pi / 2],
        [-1, -0.97, -0.929, -0.877, -0.65, -0.405, -0.062, 0.217, 0.543, 0.805, 0.863],
    ),
}


def run_dbac_circuits(psis, optimal_s_Mk, *, as_fidelity):
    """
    Run DB-QITE on `psis` for every circuit in `CIRCUITS`, grouped by k so the
    recursion steps s_1, ..., s_k are shared between circuits of the same k
    (e.g. n=2 and n=3 both use k=1, with different M).

    Returns:
        dict[str, list[float]]: label -> final energy (or fidelity) curve.
    """
    results = {}
    for label, (M, k) in CIRCUITS.items():
        s_vals = optimal_s_Mk[M - 1][:k]
        energies = [GCI(QITE_dme_select_M(M), psi, None, s_vals).execute_gci()[0][-1] for psi in psis]
        results[label] = [(1 - e) / 2 for e in energies] if as_fidelity else energies
    return results


def main():
    use_paper_style(font_size=10, figsize=figure_size("PRD", "onecol"))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    optimal_s_Mk = np.load(OPTIMAL_S_PATH)

    thetas = np.linspace(0, 1, 100)

    # --- Fidelity-fidelity plot: sweep R_X(theta)|0> -------------------------
    psis_fidelity = rx_sweep(thetas, start=np.array([[0], [1]], dtype=complex))
    fidelities_initial = [fidelity_with_0(psi) for psi in psis_fidelity]
    fid_results = run_dbac_circuits(psis_fidelity, optimal_s_Mk, as_fidelity=True)

    plot_fidelity_fidelity(fidelities_initial, fidelities_initial, fid_results,
                            save_path=PLOTS_DIR / "3_circuit_fid.pdf")

    # --- Energy vs theta: sweep R_X(theta)|1> --------------------------------
    psis_energy = rx_sweep(thetas, start=np.array([[1], [0]], dtype=complex))
    theta_angles = thetas * np.pi / 2
    res_init = [GCI(initial_psi, psi, 1, np.linspace(0.01, 1, 2)).execute_gci()[0][-1] for psi in psis_energy]
    energy_results = run_dbac_circuits(psis_energy, optimal_s_Mk, as_fidelity=False)

    xticks = np.linspace(0, np.pi / 2, 6)
    xtick_labels = [r'$0$', r'$0.1\pi$', r'$0.2\pi$', r'$0.3\pi$', r'$0.4\pi$', r'$0.5\pi$']

    plot_energy_vs_theta(theta_angles, res_init, energy_results, target_energy=-1,
                          xticks=xticks, xtick_labels=xtick_labels,
                          save_path=PLOTS_DIR / "3_circuit_energy_theta.pdf")

    # --- Per-circuit energy-vs-theta with experimental overlay ---------------
    colors = ['#7a0085', '#2c0084', '#006f7a']
    markers = ['o', 'o', 's']
    small_xticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
    small_xtick_labels = [r'$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']

    for i, (label, curve) in enumerate(energy_results.items()):
        fname = PLOTS_DIR / (label.replace(' ', '_').replace(',', '').replace('\n', '') + '.pdf')
        plot_single_energy_curve(
            theta_angles, res_init, label, curve, colors[i], markers[i],
            experimental=EXPERIMENTAL_ENERGY.get(label),
            xticks=small_xticks, xtick_labels=small_xtick_labels, save_path=fname)

    # --- 2x2 grid layout: n=2 (k=1), n=4 (k=2), n=3 (k=1) ---------------------
    plot_map = {0: r'$n=2$', 2: r'$n=4$', 1: r'$n=3$'}
    titles = {0: r'$k=1$', 1: r'$k=2$'}
    plot_energy_grid(theta_angles, res_init, energy_results, plot_map, target_energy=-1,
                      figsize=(figure_size("PRD", "onecol")[0], figure_size("PRD", "onecol")[0]),
                      titles=titles, save_path=PLOTS_DIR / "dbac_grid_layout.pdf")


if __name__ == "__main__":
    main()
