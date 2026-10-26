"""
utils.py

Small state-preparation and fidelity helpers shared by the DME/DBAC/HBAC
scripts. These used to be copy-pasted into nearly every notebook with slightly
different names (`fidelity_with_0`, `apply_depolarising_channel`,
`apply_depolarizing_noise`, ...) -- they are collected here once instead.
"""

__all__ = [
    "state_0",
    "fidelity_with_0",
    "random_density_matrix",
    "thermal_state",
    "apply_depolarizing_noise",
    "rx_sweep",
]

import numpy as np
from scipy.linalg import expm

#: Reference ground state ``|0>`` (flat vector ``[1, 0]``) that DBAC/HBAC try to reset towards.
state_0 = np.array([1, 0], dtype=complex)


def fidelity_with_0(state, reference=state_0):
    """
    Fidelity ``<reference| rho |reference>`` of a state with a reference pure
    state, by default the ground state ``|0>``.

    For a qubit, fidelity and energy carry the same information:
    ``E = 1 - 2 F`` for the Hamiltonian H = -Z.

    Parameters:
        state (np.ndarray): A state vector (any shape with 2 entries, e.g. (2, 1))
            or a 2x2 density matrix.
        reference (np.ndarray): Flat reference state vector. Defaults to `state_0`.

    Returns:
        float: Fidelity between 0 (orthogonal) and 1 (identical).

    Example:
        >>> import numpy as np
        >>> from dbac.utils import fidelity_with_0
        >>> plus = np.array([[1], [1]]) / np.sqrt(2)
        >>> round(fidelity_with_0(plus), 3)
        0.5
    """
    state = np.asarray(state)
    is_density_matrix = state.ndim == 2 and state.shape[0] == state.shape[1] and state.shape[0] > 1
    if is_density_matrix:
        rho = state
    else:
        psi = state.reshape(-1, 1)
        rho = psi @ psi.conj().T
    return float(np.real(reference.conj().T @ rho @ reference))


def random_density_matrix(dim):
    """
    Draw a random full-rank density matrix of dimension `dim` (Hilbert-Schmidt
    measure: ``A A^dagger / Tr`` with complex Gaussian ``A``).

    Parameters:
        dim (int): Dimension, e.g. 2 for one qubit.

    Returns:
        np.ndarray: A ``dim x dim`` density matrix.
    """
    a = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    return rho


def thermal_state(epsilon):
    """
    Diagonal thermal (Gibbs) state of a single qubit,
    ``diag(e^epsilon, e^-epsilon) / Z``.

    Note this uses a Boltzmann factor `epsilon`, while `dbac.hbac` parametrizes
    its bath by the polarization ``epsilon_b`` with populations ``(1 +- epsilon_b) / 2``.

    Parameters:
        epsilon (float): Inverse temperature times energy gap; 0 gives the
            maximally mixed state, large values approach ``|0><0|``.

    Returns:
        np.ndarray: 2x2 diagonal density matrix.
    """
    z = np.exp(epsilon) + np.exp(-epsilon)
    return np.array([[np.exp(epsilon) / z, 0], [0, np.exp(-epsilon) / z]])


def apply_depolarizing_noise(rho, p=0.5):
    """
    Depolarize a density matrix: ``(1 - p) rho + p I / d``.

    Parameters:
        rho (np.ndarray): ``d x d`` density matrix (any number of qubits).
        p (float): Depolarizing probability, from 0 (unchanged) to 1 (maximally mixed).

    Returns:
        np.ndarray: The depolarized density matrix.
    """
    d = rho.shape[0]
    return (1 - p) * rho + (p / d) * np.eye(d)


def rx_sweep(theta_fraction, start=np.array([[1], [0]], dtype=complex)):
    """
    Rotate `start` by ``exp(-i theta X)`` for a sweep of angles, with
    ``theta = theta_fraction * pi/2``. This is the initial-state sweep used
    throughout the package.

    `theta` is the state-vector angle: for ``start = |0>`` the result is
    ``cos(theta)|0> - i sin(theta)|1>``. This is the paper's convention
    ``R_X(theta) = exp(-i theta X)``, so a fraction of 1 reaches ``|1>``. In the
    common convention ``R_X(phi) = exp(-i phi X / 2)`` it is ``phi = 2 theta``.

    Parameters:
        theta_fraction (sequence[float]): Angles as fractions of pi/2.
        start (np.ndarray): (2, 1) state vector to rotate. Defaults to ``|0>``;
            pass ``np.array([[0], [1]], dtype=complex)`` to sweep away from ``|1>``.

    Returns:
        np.ndarray: Array of shape ``(len(theta_fraction), 2, 1)``, one state
        vector per angle.

    Example:
        >>> from dbac.utils import rx_sweep, fidelity_with_0
        >>> psis = rx_sweep([0, 0.5, 1])           # |0>, halfway, |1>
        >>> [round(fidelity_with_0(p), 3) for p in psis]
        [1.0, 0.5, 0.0]
    """
    sigma_x = np.array([[0, 1], [1, 0]])
    return np.array([expm(-1j * np.pi / 2 * sigma_x * t) @ start for t in theta_fraction])
