__all__ = [
    "dme_expm", "dme_qibo", "dme_channel", "dme_channel_equivalent", "ptrace"
]

import numpy as np
from scipy.linalg import expm
from qibo.quantum_info.linalg_operations import matrix_exponentiation, partial_trace
from qibo.gates import SWAP

def dme_expm(data, instruction, t, M=None):
    """Exact unitary evolution using matrix exponential."""
    V = expm(1j * t * instruction)
    return V @ data @ V.conj().T

def dme_qibo(data, instruction, t, M):
    """Density matrix exponentiation using Qibo's SWAP gate."""
    rho_list = []
    _t = t / M
    u = matrix_exponentiation(-_t, SWAP(0, 1).matrix())
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
    """Partial trace over one subsystem."""
    current_tensor = matrix.reshape([n1, n2, n1, n2])
    if index == 1:
        return np.trace(current_tensor, axis1=1, axis2=3)
    elif index == 2:
        return np.trace(current_tensor, axis1=0, axis2=2)
    else:
        print('Error')

def dme_channel(data, instruction, t, M):
    """DME using SWAP evolution and numpy partial trace."""
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
    """Approximate DME using analytical channel formula."""
    rho_list = []
    _t = t / M 
    c = np.cos(_t)
    s = np.sin(_t)
    for _ in range(M): 
        data = c**2 * data + s**2 * instruction + 1j * s * c * (instruction @ data - data @ instruction)
        rho_list.append(data)
    return data, rho_list
