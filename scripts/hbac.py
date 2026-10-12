"""
scripts/hbac.py

Heat-bath algorithmic cooling (HBAC) baseline for a 3-qubit chain: compares
how well HBAC resets the data qubit, starting from pure vs. depolarized input
states, after k=1 and k=6 cooling rounds. This is the classical baseline that
DBAC (see `scripts/dbac_main_plot.py`) is compared against.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from functions.hbac import hbac
from functions.utils import apply_depolarizing_noise, fidelity_with_0, rx_sweep
from visualization import use_paper_style, plot_hbac_fidelity_fidelity

PLOTS_DIR = REPO_ROOT / "jupyter_notebooks" / "plots"


def three_qubit_input(psi, depolarizing_p=None):
    """Build the 3-qubit input rho = psi x psi x psi, optionally depolarized."""
    rho = psi @ psi.conj().T
    rho = np.kron(np.kron(rho, rho), rho)
    if depolarizing_p is not None:
        rho = apply_depolarizing_noise(rho, p=depolarizing_p)
    return rho


def run_hbac_sweep(psis, *, epsilon_b=0.8, step=1, depolarizing_p=None):
    """Run HBAC for every initial state in `psis` and return the output fidelities."""
    fidelities = []
    for psi in psis:
        rho = three_qubit_input(psi, depolarizing_p=depolarizing_p)
        fidelities.append(fidelity_with_0(hbac(rho, epsilon_b=epsilon_b, step=step)))
    return fidelities


def main():
    use_paper_style(font_size=16, figsize=(5, 3.5))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    thetas = np.linspace(0, 1, 50)
    psis = rx_sweep(thetas)
    fidelities_initial = [fidelity_with_0(psi) for psi in psis]
    mixed_states = [apply_depolarizing_noise(psi @ psi.conj().T, p=0.3) for psi in psis]
    fidelities_initial_mixed = [fidelity_with_0(rho) for rho in mixed_states]

    hbac_fid_pure_k1 = run_hbac_sweep(psis, step=1)
    hbac_fid_mixed_k1 = run_hbac_sweep(psis, step=1, depolarizing_p=0.3)
    hbac_fid_pure_k6 = run_hbac_sweep(psis, step=6)
    hbac_fid_mixed_k6 = run_hbac_sweep(psis, step=6, depolarizing_p=0.3)

    plot_hbac_fidelity_fidelity(
        fidelities_initial,
        baselines={
            "Initial pure state": fidelities_initial,
            "Initial mixed state": fidelities_initial_mixed,
        },
        results={
            "Pure, k=1": hbac_fid_pure_k1,
            "Mixed, k=1": hbac_fid_mixed_k1,
            "Pure, k=6": hbac_fid_pure_k6,
            "Mixed, k=6": hbac_fid_mixed_k6,
        },
        title="HBAC for 3 qubits",
        save_path=PLOTS_DIR / "hbac_3_qubits.pdf",
    )


if __name__ == "__main__":
    main()
