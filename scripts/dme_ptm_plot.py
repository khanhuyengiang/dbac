"""
scripts/dme_ptm_plot.py

Pauli-transfer-matrix (PTM) view of the two-qubit unitary implementing one
DME/DBAC step. Compares:

- the ideal ZZ(theta) interaction,
- the DME unitary built out of ZZ/XX/YY interactions ("Heisenberg" DME), and
- two concrete circuits (A, B) used in the experiment, each a DME unitary
  followed by a single-qubit RZ correction.

Run directly to reproduce the PTM figures; import `*_ptm` functions or
`zz_unitary`/`dme_heisenberg` to compute the matrices elsewhere (e.g. from the
`dme_ptm_plot.ipynb` notebook).
"""

import sys
from pathlib import Path

import numpy as np
from qutip import Qobj, qeye, sigmax, sigmay, sigmaz
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from visualization import use_paper_style, pauli_labels_2q, plot_ptm, plot_ptm_sweep

PLOTS_DIR = REPO_ROOT / "jupyter_notebooks" / "plots"

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
S = np.array([[1, 0], [0, 1j]])
S_dagger = S.conj().T
Z = np.array([[1, 0], [0, -1]])


def zz_unitary(theta):
    """Ideal ZZ(theta) interaction, diagonal in the computational basis."""
    return np.diag([
        np.exp(-1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(-1j * theta / 2),
    ])


def dme_heisenberg(phi):
    """
    DME unitary built from ZZ, XX and YY interactions of the same angle phi,
    implementing (an approximation to) the Heisenberg exchange used in the
    hardware DME/DBAC circuits.
    """
    zz_phi = zz_unitary(phi)
    yy_phi = (np.kron(S_dagger, I) @ np.kron(H, I) @ np.kron(I, S_dagger) @ np.kron(I, H)
              @ zz_unitary(phi) @ np.kron(H, I) @ np.kron(S, I) @ np.kron(I, H) @ np.kron(I, S))
    xx_phi = np.kron(H, I) @ np.kron(I, H) @ zz_unitary(phi) @ np.kron(H, I) @ np.kron(I, H)
    return zz_phi @ yy_phi @ xx_phi


def rz(phi):
    """Single-qubit Z rotation."""
    return expm(-1j * phi * Z)


def unitary_to_ptm(u: Qobj):
    """Convert a two-qubit unitary to its Pauli-transfer matrix."""
    paulis = [qeye(2), sigmax(), sigmay(), sigmaz()]
    basis = [Qobj(np.kron(p1.full(), p2.full())) for p1 in paulis for p2 in paulis]
    t = np.zeros((16, 16), dtype=np.complex128)
    for i, pi in enumerate(basis):
        rho_out = u * pi * u.dag()
        for j, pj in enumerate(basis):
            t[j, i] = (pj.dag() * rho_out).tr() / 2
    return t


def main():
    use_paper_style(font_size=14, figsize=(10, 6))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    theta_list = [0, np.pi / 8, np.pi / 4, np.pi / 2]
    theta_names = ['0', r'$\pi/8$', r'$\pi/4$', r'$\pi/2$']

    ideal_ptms = [unitary_to_ptm(Qobj(zz_unitary(theta))) for theta in theta_list]
    plot_ptm_sweep(ideal_ptms, [f"Ideal PTM (ZZ({name}))" for name in theta_names],
                    pauli_labels_2q, save_path=PLOTS_DIR / "ptm_zz_sweep.pdf")

    dme_ptms = [unitary_to_ptm(Qobj(dme_heisenberg(theta))) for theta in theta_list]
    plot_ptm_sweep(dme_ptms, [f"Ideal PTM (DME({name}))" for name in theta_names],
                    pauli_labels_2q, save_path=PLOTS_DIR / "ptm_dme_sweep.pdf")

    # Circuit A: DME(pi/4) followed by an RZ(-0.79) correction on qubit 2.
    circuit_a = Qobj(dme_heisenberg(np.pi / 4) @ np.kron(I, rz(-0.79)))
    plot_ptm(unitary_to_ptm(circuit_a), pauli_labels_2q, title=r"Circuit A (DME($\pi/4$))")[0] \
        .savefig(PLOTS_DIR / "ptm_circuit_a.pdf")

    # Circuit B: DME(pi/8) followed by an RZ(-pi/4) correction on qubit 2.
    circuit_b = Qobj(dme_heisenberg(np.pi / 8) @ np.kron(I, rz(-np.pi / 4)))
    plot_ptm(unitary_to_ptm(circuit_b), pauli_labels_2q, title=r"Circuit B (DME($\pi/8$))")[0] \
        .savefig(PLOTS_DIR / "ptm_circuit_b.pdf")


if __name__ == "__main__":
    main()
