"""
scripts/sup_figure_S6.py

Supplementary Figure S6 (two files): final fidelity with the ground state |0> of DB-QITE
(k=1) as a function of the initial-state angle theta and the step size s, for
M=1 and M=2 Trotter steps of the DME subroutine. The solid line marks
the optimal step size used for that M (from `optimal_s_Mk.npy`); in the M=2 plot
the dashed line repeats the optimal step size of M=1 for comparison.

Run directly (`python scripts/sup_figure_S6.py`) to save
`figures/S6_M1k1_s_sweep.pdf` and
`figures/S6_M2k1_s_sweep.pdf`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbac.gci import GCI
from dbac.dbqite import QITE_dme_select_M
from dbac.utils import rx_sweep
from dbac.plotting import use_paper_style

OPTIMAL_S_PATH = REPO_ROOT / "notebooks" / "optimal_s_Mk.npy"
PLOTS_DIR = REPO_ROOT / "figures"

K = 1


def s_sweep_fidelities(M, thetas, s_list):
    """Final fidelity with |0>, indexed [s, theta], for one DB-QITE step with M Trotter steps."""
    psis = rx_sweep(thetas, start=np.array([[0], [1]], dtype=complex))
    fidelities = []
    for s in s_list:
        energies = [GCI(QITE_dme_select_M(M), psi, None, [s]).execute_gci()[0][-1] for psi in psis]
        fidelities.append([np.real((1 - E) / 2) for E in energies])
    return np.asarray(fidelities)


def plot_panel(ax, fidelities, thetas, s_list, *, M, s_line, s_dashed=None):
    im = ax.imshow(fidelities, aspect="auto", origin="lower", cmap="bwr", vmin=0, vmax=1,
                   extent=[thetas.min(), thetas.max(), s_list.min(), s_list.max()])
    ax.axhline(s_line, color="navy", linewidth=0.8)
    if s_dashed is not None:
        ax.axhline(s_dashed, color="k", linestyle="--", linewidth=1.2)
    ax.set_title(f"Final fidelities with ground state\nM={M},k={K}")
    ax.set_xlabel(r"$\theta$, Angle of initialisation")
    ax.set_ylabel("Step size s")
    return im


def main():
    use_paper_style(font_size=12, figsize=(6, 3.8))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    optimal_s_Mk = np.load(OPTIMAL_S_PATH)
    thetas = np.linspace(0, 1, 100)
    s_list = np.linspace(0.01, 1, 100)
    s_opt = {M: optimal_s_Mk[M - 1][K - 1] for M in (1, 2)}

    for M, dashed in ((1, None), (2, s_opt[1])):
        fid = s_sweep_fidelities(M, thetas, s_list)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        im = plot_panel(ax, fid, thetas, s_list, M=M, s_line=s_opt[M], s_dashed=dashed)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.text(1.5, 1.05, r"$|0\rangle$", ha="center", va="bottom", transform=cbar.ax.transAxes)
        cbar.ax.text(1.5, -0.05, r"$|1\rangle$", ha="center", va="top", transform=cbar.ax.transAxes)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"S6_M{M}k{K}_s_sweep.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
