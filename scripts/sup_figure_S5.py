"""
scripts/sup_figure_S5.py

DB-QITE cooling performance when the input state has been passed through a
single-qubit depolarizing channel,

    rho_mixed = (1 - p) * rho + p * I / 2,

compared to the noiseless (pure-state) case (`compare_pure_vs_mixed`, used by
the notebooks).

Run directly to render the Bloch-sphere illustration of the DB-QITE recursion
trajectory for three representative initial fidelities (good/mid/bad), saved as
`figures/S5_bloch_{good,mid,bad}.pdf`.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbac.gci import GCI
from dbac.dbqite import QITE_dme_select_M, initial_psi
from dbac.utils import apply_depolarizing_noise, fidelity_with_0, rx_sweep
from dbac.plotting import use_paper_style, plot_bloch_trajectory

OPTIMAL_S_PATH = REPO_ROOT / "notebooks" / "optimal_s_Mk.npy"
PLOTS_DIR = REPO_ROOT / "figures"

TARGET_FIDELITY = 0.9


def compare_pure_vs_mixed(optimal_s_Mk, *, M=1, k=1, depolarizing_p=0.3):
    """
    Run DB-QITE (M, k) on a sweep of pure initial states rotated away from |1>
    (see `dbac.utils.rx_sweep`), and on
    the same states passed through a depolarizing channel.

    Returns:
        (fidelities_initial, results): fidelities of the pure initial states,
        and a dict {"Circuit A": pure fidelities, "Circuit A on mixed state
        (p=...)": mixed fidelities}.
    """
    thetas = np.linspace(0, 1, 100)
    psis = rx_sweep(thetas, start=np.array([[0], [1]], dtype=complex))
    fidelities_initial = [fidelity_with_0(psi) for psi in psis]

    s_vals = optimal_s_Mk[M - 1][:k]
    energies_pure = [GCI(QITE_dme_select_M(M), psi, None, s_vals).execute_gci()[0][-1] for psi in psis]

    mixed_states = [apply_depolarizing_noise(psi @ psi.conj().T, p=depolarizing_p) for psi in psis]
    energies_mixed = [GCI(QITE_dme_select_M(M), rho, None, s_vals).execute_gci()[0][-1] for rho in mixed_states]

    results = {
        "Circuit A": [(1 - e) / 2 for e in energies_pure],
        f"Circuit A on\nmixed state (p={depolarizing_p})": [(1 - e) / 2 for e in energies_mixed],
    }
    return fidelities_initial, results


def bloch_trajectory_states(optimal_s_Mk, psis, index, M, k):
    """Return the GCI recursion states (initial, then after each of k steps)."""
    s_vals = optimal_s_Mk[M - 1][:k]
    return GCI(QITE_dme_select_M(M), psis[index], None, s_vals).execute_gci()[-3]


def main():
    use_paper_style(font_size=16, figsize=(5, 3.5))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    optimal_s_Mk = np.load(OPTIMAL_S_PATH)

    # --- Bloch-sphere illustration of the DB-QITE recursion ------------------
    thetas = np.linspace(0, 1, 100)
    psis = rx_sweep(thetas, start=np.array([[0], [1]], dtype=complex))

    texts = ["Initial state", r"$k=1$", r"$k=2$"]
    states_good = bloch_trajectory_states(optimal_s_Mk, psis, 70, M=1, k=1)
    states_mid = bloch_trajectory_states(optimal_s_Mk, psis, 55, M=1, k=2)
    states_bad = bloch_trajectory_states(optimal_s_Mk, psis, 30, M=2, k=2)

    plot_bloch_trajectory(states_good, texts[:len(states_good)], marker='o', marker_color='C2',
                           target_fidelity=TARGET_FIDELITY, save_path=PLOTS_DIR / "S5_bloch_good.pdf")
    plot_bloch_trajectory(states_mid, texts[:len(states_mid)], marker='s', marker_color='C1',
                           target_fidelity=TARGET_FIDELITY, save_path=PLOTS_DIR / "S5_bloch_mid.pdf")
    plot_bloch_trajectory(states_bad, texts[:len(states_bad)], marker='d', marker_color='C3',
                           target_fidelity=TARGET_FIDELITY, save_path=PLOTS_DIR / "S5_bloch_bad.pdf")


if __name__ == "__main__":
    main()
