"""
dme.py

Implementations of the density matrix exponentiation (DME) channel, which
applies ``exp(i t rho) sigma exp(-i t rho)`` to a data state `sigma` using
copies of an instruction state `rho`, without knowing `rho`.

All functions take ``(data, instruction, t, M)``: the data and instruction
2x2 density matrices, the total duration `t`, and the number of Trotter steps
`M` (instruction copies consumed). The approximate implementations return
``(final_data, history)`` where `history` holds the data state after each of
the `M` steps. They compute the same channel in different ways so they can be
checked against each other (the paper's Fig. S2).
"""

__all__ = [
    "dme_expm", "dme_qibo", "dme_channel", "dme_channel_equivalent", "ptrace"
]

import numpy as np
from scipy.linalg import expm
from qibo.quantum_info.linalg_operations import partial_trace
from qibo.gates import SWAP

def dme_expm(data, instruction, t, M=None):
    """
    The exact evolution ``exp(i t instruction) data exp(-i t instruction)``,
    i.e. the M -> infinity limit that DME approximates. The reference for the
    other implementations.

    Parameters:
        data (np.ndarray): 2x2 data density matrix.
        instruction (np.ndarray): 2x2 instruction density matrix (the Hamiltonian).
        t (float): Duration.
        M: Ignored; accepted so the signature matches the other implementations.

    Returns:
        np.ndarray: The evolved 2x2 data density matrix (not a tuple).
    """
    V = expm(1j * t * instruction)
    return V @ data @ V.conj().T

def dme_qibo(data, instruction, t, M):
    """
    DME built from qibo's SWAP gate matrix and qibo's partial trace.

    Same channel as `dme_channel`; useful to cross-check the numpy
    implementation against qibo.

    Parameters:
        data (np.ndarray): 2x2 data density matrix.
        instruction (np.ndarray): 2x2 instruction density matrix.
        t (float): Total duration, split into `M` steps of ``t / M``.
        M (int): Number of Trotter steps.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: The final data state, and the data
        state after each step.
    """
    rho_list = []
    _t = t / M
    # Not qibo's `matrix_exponentiation`: in qibo 0.3.x it returns expm(phase * M)
    # without the factor of i, i.e. a non-unitary operator.
    u = expm(1j * _t * SWAP(0, 1).matrix())
    dme_state = np.kron(data, instruction)
    for _ in range(M):
        dme_state = u @ dme_state @ u.T.conj()
        data = partial_trace(dme_state, [1])
        rho_list.append(data)
        dme_state = np.kron(data, instruction)
    return data, rho_list

swap = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)

def ptrace(matrix, index=1, n1=2, n2=2):
    """
    Partial trace of a bipartite ``(n1 * n2) x (n1 * n2)`` matrix.

    Parameters:
        matrix (np.ndarray): Operator on subsystems 1 (dimension `n1`) and 2
            (dimension `n2`).
        index (int): Which subsystem to trace **out**: 1 traces out subsystem 2
            and keeps subsystem 1; 2 traces out subsystem 1 and keeps subsystem 2.
        n1, n2 (int): Dimensions of the two subsystems.

    Returns:
        np.ndarray: The reduced operator.
    """
    current_tensor = matrix.reshape([n1, n2, n1, n2])
    if index == 1:
        return np.trace(current_tensor, axis1=1, axis2=3)
    elif index == 2:
        return np.trace(current_tensor, axis1=0, axis2=2)
    else:
        print('Error')

def dme_channel(data, instruction, t, M):
    """
    DME as on hardware: apply a partial SWAP ``exp(i (t / M) SWAP)`` to
    ``data (x) instruction``, trace out the instruction, and repeat `M` times
    with a fresh instruction copy.

    Parameters:
        data (np.ndarray): 2x2 data density matrix.
        instruction (np.ndarray): 2x2 instruction density matrix.
        t (float): Total duration, split into `M` steps of ``t / M``.
        M (int): Number of Trotter steps (instruction copies consumed).

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: The final data state, and the data
        state after each step.

    Example:
        >>> import numpy as np
        >>> from dbac.dme import dme_channel, dme_expm
        >>> data = np.array([[1, 0], [0, 0]], dtype=complex)
        >>> instruction = np.full((2, 2), 0.5, dtype=complex)
        >>> exact = dme_expm(data, instruction, 1.0)
        >>> [round(float(np.linalg.norm(exact - dme_channel(data, instruction, 1.0, M)[0])), 4)
        ...  for M in (1, 10, 100)]   # error shrinks like 1/M
        [0.5969, 0.0757, 0.0079]
    """
    rho_list = []
    _t = t / M
    u = expm(1j * _t * swap)
    dme_state = np.kron(data, instruction)
    for _ in range(M):
        dme_state = u @ dme_state @ u.T.conj()
        data = ptrace(dme_state)
        rho_list.append(data)
        dme_state = np.kron(data, instruction)
    return data, rho_list

def dme_channel_equivalent(data, instruction, t, M):
    """
    DME using the closed form of one channel step,
    ``cos^2 data + sin^2 instruction + i sin cos [instruction, data]``
    (with angle ``t / M``), which avoids building two-qubit matrices.

    Parameters:
        data (np.ndarray): 2x2 data density matrix.
        instruction (np.ndarray): 2x2 instruction density matrix.
        t (float): Total duration, split into `M` steps of ``t / M``.
        M (int): Number of Trotter steps.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: The final data state, and the data
        state after each step.
    """
    rho_list = []
    _t = t / M 
    c = np.cos(_t)
    s = np.sin(_t)
    for _ in range(M): 
        data = c**2 * data + s**2 * instruction + 1j * s * c * (instruction @ data - data @ instruction)
        rho_list.append(data)
    return data, rho_list
