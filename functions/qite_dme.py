"""
qite_dme.py

Module implementing DB-QITE with Density Matrix Exponentiation (DME) 
via Group Commutator Iteration (GCI) for a single qubit.
"""

__all__ = [
    "DME",
    "QITE_GCI_dme",
    "QITE_dme_select_M",
    "initial_psi",
    "ideal_DME",
    "H", "H_sh", "eigs", "target_energy"
]

import numpy as np
from scipy.linalg import expm
from qibo import hamiltonians
from qibo.symbols import Z

# Constants and Hamiltonian setup
L = 1
H_sh = hamiltonians.SymbolicHamiltonian(Z(0), nqubits=1)
H = np.array([[-1.-0.j, -0.-0.j],
              [-0.-0.j,  1.-0.j]])
eigs = np.array([-1., 1.])
target_energy = -1

def DME(instruction: np.ndarray, data: np.ndarray, t: float, M: int = 100) -> np.ndarray:
    """
    Approximate exp(i t [instruction, .]) applied to data using Trotterized DME.

    Args:
        instruction (np.ndarray): Operator to exponentiate.
        data (np.ndarray): Density matrix to transform.
        t (float): Time parameter.
        M (int): Number of Trotter steps.

    Returns:
        np.ndarray: Updated density matrix.
    """
    _t = t / M
    c = np.cos(_t)
    s = np.sin(_t)
    for _ in range(M):
        data = c**2 * data + s**2 * instruction + 1j * s * c * (instruction @ data - data @ instruction)
    return data

def QITE_GCI_dme(psi: np.ndarray, s: float, M: int = 100) -> np.ndarray:
    """
    Run a QITE-GCI evolution using DME for fixed steps.

    Args:
        psi (np.ndarray): Initial state vector.
        s (float): Time step.
        M (int): Number of DME steps.

    Returns:
        np.ndarray: Updated density matrix.
    """
    QITE_GCI_dme.label = r"DME"
    QITE_GCI_dme.marker = '^'
    PSI = psi @ psi.T.conj()
    U_t = expm(1j * s * H)
    return DME(U_t @ PSI @ U_t.conj().T, PSI, s, M=M)

def QITE_dme_select_M(M: int = 100):
    """
    Return a QITE-GCI function with fixed DME step count.

    Args:
        M (int): Number of DME steps.

    Returns:
        Callable: QITE-GCI function.
    """
    def QITE_GCI_dme(psi: np.ndarray, s: float, M: int = M) -> np.ndarray:
        QITE_GCI_dme.label = r"DME"
        QITE_GCI_dme.marker = '^'
        PSI = psi @ psi.T.conj()
        U_t = expm(1j * s * H)
        return DME(U_t @ PSI @ U_t.conj().T, PSI, s, M=M)
    return QITE_GCI_dme

def initial_psi(psi: np.ndarray, s: float) -> np.ndarray:
    """
    Return the initial state unchanged.

    Args:
        psi (np.ndarray): Initial state vector.
        s (float): Time step (unused).

    Returns:
        np.ndarray: The input state.
    """
    initial_psi.label = "Initial state"
    initial_psi.marker = 'o'
    return psi

def ideal_DME(psi: np.ndarray, s: float) -> np.ndarray:
    """
    Apply ideal DME evolution via full exponentiation.

    Args:
        psi (np.ndarray): Initial state vector.
        s (float): Time step.

    Returns:
        np.ndarray: Evolved state.
    """
    PSI = psi @ psi.T.conj()
    U_t = expm(1j * s * H)
    instruction = U_t @ PSI @ U_t.conj().T
    return expm(1j * s * instruction) @ PSI @ expm(-1j * s * instruction)
