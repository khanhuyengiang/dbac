"""
scripts/sup_figure_S2.py

Benchmarks the different ways of implementing a single step of density matrix
exponentiation (DME):

    E_t^(rho)(sigma) = Tr_2[ e^{i t SWAP} (sigma (x) rho) e^{-i t SWAP} ]

with data sigma and instruction rho, against the exact unitary
e^{it*rho} sigma e^{-it*rho} (the sign convention used throughout
`dbac.dme`), for an increasing
number of Trotter steps M. Three implementations are compared:

- `dme_channel`: explicit SWAP evolution + numpy partial trace.
- `dme_channel_equivalent`: the same channel, written analytically with
  cos/sin so no partial trace is needed.
- `dme_qibo`: the same SWAP evolution, built from a qibo circuit.

Run directly (`python scripts/sup_figure_S2.py`) to reproduce the figure in
`figures/S2_DME_norm.pdf`, or import `run_dme_channel_benchmark`
to get the raw data (e.g. from a notebook).
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # allow `python scripts/sup_figure_S2.py` to find `dbac`

from dbac.dme import dme_expm, dme_channel, dme_channel_equivalent, dme_qibo
from dbac.plotting import use_paper_style, plot_dme_channel_norms

PLOTS_DIR = REPO_ROOT / "figures"


def random_pure_state_dm(rng):
    """Draw a Haar-random single-qubit pure-state density matrix."""
    psi = rng.random(2) + 1j * rng.random(2)
    psi /= np.linalg.norm(psi)
    psi = psi[:, np.newaxis]
    return psi @ psi.conj().T


def run_dme_channel_benchmark(t=1.0, max_M=50, seed=None):
    """
    Compare the DME channel implementations against the exact evolution.

    Parameters:
        t (float): DME channel duration.
        max_M (int): Benchmark M = 1, ..., max_M - 1 Trotter steps.
        seed (int, optional): Seed for the random data/instruction states.

    Returns:
        dict[str, list[float]]: label -> trace-norm distance to the exact
        evolution, one entry per M.
    """
    rng = np.random.default_rng(seed)
    data = random_pure_state_dm(rng)
    instruction = random_pure_state_dm(rng)

    expected_state = dme_expm(data, instruction, t, 0)

    norms = {"DME channel": [], "Equivalent channel": [], "Qibo": []}
    for M in range(1, max_M):
        _, rho_list = dme_channel(data, instruction, t, M)
        norms["DME channel"].append(np.linalg.norm(expected_state - rho_list[-1]))

        _, rho_list_eq = dme_channel_equivalent(data, instruction, t, M)
        norms["Equivalent channel"].append(np.linalg.norm(expected_state - rho_list_eq[-1]))

        _, rho_list_qibo = dme_qibo(data, instruction, t, M)
        norms["Qibo"].append(np.linalg.norm(expected_state - rho_list_qibo[-1]))

    return norms


def main():
    use_paper_style(font_size=16, figsize=(5, 3.5))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    norms = run_dme_channel_benchmark()

    # Reproduce the paper figure: the "DME channel" curve alone.
    plot_dme_channel_norms({"DME channel": norms["DME channel"]}, colors=['#7a0085'],
                            save_path=PLOTS_DIR / "S2_DME_norm.pdf")

    # All three implementations together, annotated with their final norm.
    plot_dme_channel_norms(norms, show_final_value=True)


if __name__ == "__main__":
    main()
