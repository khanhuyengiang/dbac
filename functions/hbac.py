"""
hbac.py

Heat-bath algorithmic cooling (HBAC) baseline used for comparison against
DBAC. HBAC repeatedly SWAPs entropy onto a fresh, highly-polarized "reset"
qubit and discards it, compressing the joint state towards its ground state
one qubit at a time.
"""

__all__ = ["refresh_reset_qubit", "compress_density_matrix", "hbac"]

import numpy as np

from functions.dme import ptrace


def refresh_reset_qubit(rho, epsilon_b):
    """
    Reset the last qubit of a joint density matrix to a thermal state with
    polarization `epsilon_b`, by rescaling each 2x2 block of `rho` that acts
    on that qubit.

    Parameters:
        rho (ndarray): 2^n x 2^n density matrix of n qubits.
        epsilon_b (float): Polarization of the reset qubit.

    Returns:
        ndarray: New density matrix after resetting the last qubit.
    """
    d = rho.shape[0]
    assert rho.shape == (d, d), "Input must be a square matrix"
    assert (d & (d - 1)) == 0, "Dimension must be a power of 2"

    num_blocks = d // 2  # since each M_ij is 2x2
    new_rho = np.zeros_like(rho, dtype=complex)

    thermal_matrix = np.array([
        [(1 + epsilon_b) / 2, 0],
        [0, (1 - epsilon_b) / 2],
    ])

    for i in range(num_blocks):
        for j in range(num_blocks):
            row_start, row_end = i * 2, i * 2 + 2
            col_start, col_end = j * 2, j * 2 + 2
            m_ij = rho[row_start:row_end, col_start:col_end]
            new_rho[row_start:row_end, col_start:col_end] = np.trace(m_ij) * thermal_matrix

    return new_rho


def compress_density_matrix(rho):
    """
    Sort-and-compress step of HBAC: reorder the populations of `rho` in
    descending order, discarding coherences, to approximate the state
    achievable by the optimal (sorting) unitary.

    Parameters:
        rho (ndarray): Density matrix (Hermitian, positive semidefinite, trace 1).

    Returns:
        ndarray: Diagonal density matrix with sorted probabilities.
    """
    assert rho.shape[0] == rho.shape[1], "Matrix must be square"
    assert np.allclose(rho, rho.conj().T), "Matrix must be Hermitian"

    probs = np.real(np.diag(rho))
    sorted_probs = np.sort(probs)[::-1]
    return np.diag(sorted_probs)


def hbac(rho, epsilon_b=0.8, step=1):
    """
    Run `step` rounds of heat-bath algorithmic cooling (reset + compress) on
    the reset qubit of `rho`, then trace it out.

    Parameters:
        rho (ndarray): 2^n x 2^n density matrix, last qubit is the reset qubit.
        epsilon_b (float): Polarization of the fresh reset qubit each round.
        step (int): Number of reset-and-compress rounds.

    Returns:
        ndarray: Reduced density matrix of the remaining n-1 qubits.
    """
    for _ in range(step):
        rho = refresh_reset_qubit(rho, epsilon_b)
        rho = compress_density_matrix(rho)
    d = rho.shape[0]
    return ptrace(rho, index=1, n1=2, n2=int(d / 2))
