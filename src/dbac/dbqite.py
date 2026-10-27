"""
dbqite.py

One step of double-bracket quantum imaginary-time evolution (DB-QITE), the
cooling step of DBAC, for a single qubit with Hamiltonian H = -Z.

A step maps the current state psi to exp(i s rho') psi exp(-i s rho'), where
``rho' = exp(i s H) |psi><psi| exp(-i s H)`` is a rotated copy of psi used as the
instruction of density matrix exponentiation (DME). All step functions share
the signature `step(psi, s) -> density matrix`, so they can be passed to
`dbac.gci.GCI` to run several steps.
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
#: The same Hamiltonian as a qibo ``SymbolicHamiltonian``, for qibo-circuit states.
H_sh = hamiltonians.SymbolicHamiltonian(Z(0), nqubits=1)
#: The Hamiltonian H = -Z = diag(-1, 1) whose ground state ``|0>`` DBAC cools towards.
H = np.array([[-1.-0.j, -0.-0.j],
              [-0.-0.j,  1.-0.j]])
#: Eigenvalues of `H`: the ground-state energy -1 and the excited-state energy +1.
eigs = np.array([-1., 1.])
#: Energy of the target (ground) state ``|0>``.
target_energy = -1

def _as_density_matrix(psi: np.ndarray) -> np.ndarray:
    """Return `psi` as a density matrix: a 2x2 input is taken to already be one
    (e.g. the output of a previous DB-QITE step, or a mixed state), a (2, 1)
    state vector is turned into |psi><psi|."""
    if psi.ndim == 2 and psi.shape[0] == psi.shape[1]:
        return psi
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T

def DME(instruction: np.ndarray, data: np.ndarray, t: float, M: int = 100) -> np.ndarray:
    """
    Apply exp(i t instruction) to `data` with Trotterized density matrix
    exponentiation, without ever exponentiating `instruction` itself.

    Each of the `M` steps is the DME channel for duration t/M with a fresh copy
    of `instruction`, written in closed form:
    ``cos^2 data + sin^2 instruction + i sin cos [instruction, data]``.
    The result approaches ``exp(i t instruction) data exp(-i t instruction)``
    with an error of order t^2 / M.

    Note the argument order: the instruction comes first, unlike the functions
    in `dbac.dme`.

    Args:
        instruction (np.ndarray): 2x2 density matrix used as the Hamiltonian
            (the instruction state).
        data (np.ndarray): 2x2 density matrix to transform (the data state).
        t (float): Total DME duration.
        M (int): Number of Trotter steps, i.e. instruction copies consumed.

    Returns:
        np.ndarray: The transformed 2x2 data density matrix.

    Example:
        >>> import numpy as np
        >>> from dbac.dbqite import DME
        >>> zero = np.array([[1, 0], [0, 0]], dtype=complex)    # data |0><0|
        >>> plus = np.full((2, 2), 0.5, dtype=complex)          # instruction |+><+|
        >>> # t = pi/2 with one step is a full SWAP: the data becomes the instruction
        >>> DME(plus, zero, t=np.pi / 2, M=1).real.round(3)
        array([[0.5, 0.5],
               [0.5, 0.5]])
    """
    _t = t / M
    c = np.cos(_t)
    s = np.sin(_t)
    for _ in range(M):
        data = c**2 * data + s**2 * instruction + 1j * s * c * (instruction @ data - data @ instruction)
    return data

def QITE_GCI_dme(psi: np.ndarray, s: float, M: int = 100) -> np.ndarray:
    """
    One DB-QITE step with Trotterized DME, using `M` = 100 DME steps by default.

    Equivalent to ``QITE_dme_select_M(M)(psi, s)``; prefer `QITE_dme_select_M`
    when passing a step function with a specific `M` to `dbac.gci.GCI`.

    Args:
        psi (np.ndarray): Current state, a (2, 1) state vector or 2x2 density matrix.
        s (float): Step duration.
        M (int): Number of DME steps.

    Returns:
        np.ndarray: The 2x2 density matrix after the step.
    """
    QITE_GCI_dme.label = r"DME"
    QITE_GCI_dme.marker = '^'
    PSI = _as_density_matrix(psi)
    U_t = expm(1j * s * H)
    return DME(U_t @ PSI @ U_t.conj().T, PSI, s, M=M)

def QITE_dme_select_M(M: int = 100):
    """
    Build a DB-QITE step function that uses `M` DME (Trotter) steps.

    The returned function ``step(psi, s)`` performs one DBAC step of duration
    `s` on `psi`: it rotates a copy of the state about H to get the
    instruction ``exp(i s H) psi exp(-i s H)``, then applies it to `psi` with
    `DME` using `M` steps. `M` = 1 and `M` = 2 correspond to circuits A and B
    of the paper's experiment.

    Args:
        M (int): Number of DME steps per DBAC step.

    Returns:
        Callable[[np.ndarray, float], np.ndarray]: ``step(psi, s)``, which takes
        a (2, 1) state vector or 2x2 density matrix and returns the 2x2 density
        matrix after the step.

    Example:
        >>> import numpy as np
        >>> from dbac.dbqite import QITE_dme_select_M, H
        >>> from dbac.utils import rx_sweep
        >>> psi = rx_sweep([0.5])[0]            # energy 0, halfway to |1>
        >>> rho = QITE_dme_select_M(1)(psi, np.pi / 4)
        >>> round(float(np.trace(rho @ H).real), 3)
        -0.5
    """
    def QITE_GCI_dme(psi: np.ndarray, s: float, M: int = M) -> np.ndarray:
        QITE_GCI_dme.label = r"DME"
        QITE_GCI_dme.marker = '^'
        PSI = _as_density_matrix(psi)
        U_t = expm(1j * s * H)
        return DME(U_t @ PSI @ U_t.conj().T, PSI, s, M=M)
    return QITE_GCI_dme

def initial_psi(psi: np.ndarray, s: float) -> np.ndarray:
    """
    A step function that does nothing: returns `psi` unchanged.

    Useful as the "no cooling" baseline, since it has the same signature as the
    other step functions and can be passed to `dbac.gci.GCI`.

    Args:
        psi (np.ndarray): Current state.
        s (float): Step duration (unused).

    Returns:
        np.ndarray: The input state.
    """
    initial_psi.label = "Initial state"
    initial_psi.marker = 'o'
    return psi

def ideal_DME(psi: np.ndarray, s: float) -> np.ndarray:
    """
    One DB-QITE step with the reflection applied exactly (the limit M -> infinity).

    Same as ``QITE_dme_select_M(M)(psi, s)`` for very large `M`, but computes
    ``exp(i s rho')`` by matrix exponential instead of DME. For this step the
    paper gives the energy afterwards in closed form,
    ``E0 - 2 sin^2(s) (1 - E0^2) [(1 - cos s) E0 + cos s]``.

    Args:
        psi (np.ndarray): Current state, a (2, 1) state vector or 2x2 density matrix.
        s (float): Step duration.

    Returns:
        np.ndarray: The 2x2 density matrix after the step.

    Example:
        >>> import numpy as np
        >>> from dbac.dbqite import ideal_DME, H
        >>> from dbac.utils import rx_sweep
        >>> rho = ideal_DME(rx_sweep([0.5])[0], np.pi / 4)
        >>> round(float(np.trace(rho @ H).real), 3)   # -1/sqrt(2)
        -0.707
    """
    PSI = _as_density_matrix(psi)
    U_t = expm(1j * s * H)
    instruction = U_t @ PSI @ U_t.conj().T
    V = expm(1j * s * instruction)
    return V @ PSI @ V.conj().T
