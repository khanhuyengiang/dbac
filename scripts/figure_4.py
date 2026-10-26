"""
scripts/figure_4.py

Main DBAC result (Fig. 4 of the paper): DB-QITE cooling performance using a
Trotterized DME subroutine, for circuits using n = 2, 3, 4 total qubits
(n - 1 quantum "instructions" plus the data qubit). Produces:

- one energy-vs-theta plot per circuit (A: n=2, B: n=3, C: n=4), overlaid
  with the hardware experiment data (`F4_Circuit_*_n_*.pdf`).

Run directly to regenerate the figures under `figures/`.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbac.gci import GCI
from dbac.dbqite import QITE_dme_select_M, initial_psi
from dbac.utils import rx_sweep
from dbac.plotting import (
    use_paper_style, figure_size, plot_single_energy_curve,
)

OPTIMAL_S_PATH = REPO_ROOT / "notebooks" / "optimal_s_Mk.npy"
PLOTS_DIR = REPO_ROOT / "figures"

# Circuit A (n=2, k=1), B (n=3, k=1, M=2), C (n=4, k=2, M=1): label -> (M, k)
CIRCUITS = {
    r'$n=2$': (1, 1),
    r'$n=3$': (2, 1),
    r'$n=4$': (1, 2),
}

# Output file name for each circuit's single-curve plot.
CIRCUIT_FILENAMES = {
    r'$n=2$': "F4_Circuit_A_n_2.pdf",
    r'$n=3$': "F4_Circuit_B_n_3.pdf",
    r'$n=4$': "F4_Circuit_C_n_4.pdf",
}

# Styling of each circuit's curve and experimental markers, matching the paper's Fig. 4.
CIRCUIT_COLORS = {
    r'$n=2$': '#7a0085',
    r'$n=3$': '#2c0084',
    r'$n=4$': '#1a1a8d',
}
CIRCUIT_MARKERS = {label: 'o' for label in CIRCUITS}

# Tick marks (in units of pi) of the energy-vs-theta plots.
XTICKS = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
XTICK_LABELS = [r'$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']

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

    # --- Energy vs theta: sweep the initial state away from |0> ---------------
    psis_energy = rx_sweep(thetas, start=np.array([[1], [0]], dtype=complex))
    theta_angles = thetas * np.pi / 2
    res_init = [GCI(initial_psi, psi, 1, np.linspace(0.01, 1, 2)).execute_gci()[0][-1] for psi in psis_energy]
    energy_results = run_dbac_circuits(psis_energy, optimal_s_Mk, as_fidelity=False)

    # --- Per-circuit energy-vs-theta with experimental overlay ---------------
    for label, curve in energy_results.items():
        plot_single_energy_curve(
            theta_angles, res_init, label, curve, CIRCUIT_COLORS[label], CIRCUIT_MARKERS[label],
            experimental=EXPERIMENTAL_ENERGY.get(label),
            xticks=XTICKS, xtick_labels=XTICK_LABELS, save_path=PLOTS_DIR / CIRCUIT_FILENAMES[label])

if __name__ == "__main__":
    main()
