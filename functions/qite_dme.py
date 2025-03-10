import numpy as np
from scipy.linalg import expm
from qibo import hamiltonians
from qibo.symbols import Z

L = 1
H_sh = hamiltonians.SymbolicHamiltonian(Z(0), nqubits = 1)
H = - H_sh.matrix
eigs = H_sh.eigenvalues()
target_energy = min(eigs)

def DME(instruction,data,t,M=100): 
    _t = t/M 
    c = np.cos(_t)
    s = np.sin(_t)
    for i in range(M): 
        data = c**2 * data + s**2 * instruction + 1j * s * c * (instruction @ data - data @ instruction)
        # print(data)
    return data

def QITE_GCI_dme(psi, s, M=100):
    QITE_GCI_dme.label=r"DME"
    QITE_GCI_dme.marker = '^'
    PSI = psi@psi.T.conj()
    U_t = expm(1j*s*H)
    state = DME( U_t @ PSI @ U_t.conj().T, PSI, s, M = M)
    return state

def QITE_dme_select_M(M=100):
    def QITE_GCI_dme(psi, s, M=M):
        QITE_GCI_dme.label=r"DME"
        QITE_GCI_dme.marker = '^'
        PSI = psi@psi.T.conj()
        U_t = expm(1j*s*H)
        state = DME( U_t @ PSI @ U_t.conj().T, PSI, s, M = M)
        return state
    return QITE_GCI_dme

def initial_psi(psi, s):
    initial_psi.label="Initial state"
    initial_psi.marker = 'o'
    return psi