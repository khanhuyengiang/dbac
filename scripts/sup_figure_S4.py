"""
scripts/sup_figure_S4.py

DB-QITE cooling performance in the idealized limit where DME is replaced by
the exact unitary it approximates (`dbac.dbqite.ideal_DME`), i.e. the
M -> infinity / no-Trotter-error limit. Sweeps the recursion depth k = 1..6
and shows how the fidelity-fidelity curve improves with k, together with the
optimized step sizes. Reproduces the "rainbow plot" figure
(`figures/S4_QITE_ideal_DME_merged.pdf`) shown in
`sup_figure_S4.ipynb`.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import cm
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbac.gci import GCI, max_excitation
from dbac.dbqite import QITE_dme_select_M, initial_psi, ideal_DME
from dbac.utils import fidelity_with_0, rx_sweep
from dbac.plotting import use_paper_style, figure_size, plot_ideal_dme_summary

OPTIMAL_S_PATH = REPO_ROOT / "notebooks" / "optimal_s_Mk.npy"
PLOTS_DIR = REPO_ROOT / "figures"

RAINBOW_COLORS = [
    (r * 0.8, g * 0.8, b * 0.8, a) for r, g, b, a in cm.rainbow(np.linspace(1, 0, 6))
]
MARKERS = ['o', 's', '^', '*', 'P', '*', 'h']


def optimize_ideal_dme_steps(max_k=7, psi0=None):
    """
    Greedily optimize the DB-QITE step sizes s_1, ..., s_max_k for the ideal
    (M -> infinity) DME, one step at a time, to minimize the final energy.
    """
    if psi0 is None:
        psi0 = np.sqrt(0.5) * np.array([[1], [1j]])

    optimal_s = []
    for _ in range(max_k):
        def objective(s, fixed=tuple(optimal_s)):
            return GCI(ideal_DME, psi0, None, list(fixed) + [s[0]]).execute_gci()[0][-1].real

        result = minimize(objective, 0.7, method='BFGS')
        optimal_s.append(result.x[0])
    return optimal_s


def load_or_fit_optimal_s_Mk(max_k=7, max_M=10, psi0=None):
    """
    Load the cached (M, k) -> optimal step-size table used by the Trotterized
    DME scripts (`figure_4.py`, `sup_figure_S5.py`), fitting and caching it
    to `OPTIMAL_S_PATH` if it does not exist yet.
    """
    if OPTIMAL_S_PATH.exists():
        return np.load(OPTIMAL_S_PATH)

    if psi0 is None:
        psi0 = np.sqrt(0.5) * np.array([[1], [1j]])

    optimal_s_Mk = []
    for M in range(1, max_M + 1):
        s_list = []
        for _ in range(max_k):
            def objective(s, fixed=tuple(s_list)):
                return GCI(QITE_dme_select_M(M), psi0, None, list(fixed) + [s[0]]).execute_gci()[0][-1].real

            result = minimize(objective, np.pi / 4, method='Nelder-Mead')
            s_list.append(result.x[0])
        optimal_s_Mk.append(s_list)

    optimal_s_Mk = np.array(optimal_s_Mk)
    np.save(OPTIMAL_S_PATH, optimal_s_Mk)
    return optimal_s_Mk


def run_ideal_dme_sweep(optimal_s, max_k=6, start=np.array([[0], [1]], dtype=complex)):
    """
    Sweep the initial state away from |1> (see `dbac.utils.rx_sweep`) and run ideal DB-QITE for each
    recursion depth k = 1, ..., max_k.

    Returns:
        (fidelities_initial, initial_curve, dict_fid): the initial-state
        fidelities, the "no cooling" baseline curve, and a dict
        'k=1', ..., 'k=max_k' -> final-state fidelity curve.
    """
    thetas = np.linspace(0, 1, 100)
    psis = rx_sweep(thetas, start=start)

    fidelities_initial = [fidelity_with_0(psi) for psi in psis]
    res_init = [GCI(initial_psi, psi, 1, np.linspace(0.01, 1, 2)).execute_gci()[0][-1] for psi in psis]
    initial_curve = [(1 - e) / 2 for e in res_init]

    dict_fid = {}
    for k in range(1, max_k + 1):
        energies = [GCI(ideal_DME, psi, None, optimal_s[:k]).execute_gci()[0][-1] for psi in psis]
        dict_fid[f'k={k}'] = [(1 - e) / 2 for e in energies]

    return fidelities_initial, initial_curve, dict_fid


def main():
    use_paper_style(font_size=10, figsize=figure_size("PRD", "twocol", aspect=2))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    optimal_s = optimize_ideal_dme_steps(max_k=7)
    optimal_s_Mk = load_or_fit_optimal_s_Mk(max_k=7, max_M=10)
    optimal_s_kM = optimal_s_Mk.T

    fidelities_initial, initial_curve, dict_fid = run_ideal_dme_sweep(optimal_s, max_k=6)

    plot_ideal_dme_summary(
        fidelities_initial, initial_curve, dict_fid, optimal_s_kM,
        target_fidelity=0.9, colors=RAINBOW_COLORS, markers=MARKERS,
        save_path=PLOTS_DIR / "S4_QITE_ideal_DME_merged.pdf",
    )


if __name__ == "__main__":
    main()
