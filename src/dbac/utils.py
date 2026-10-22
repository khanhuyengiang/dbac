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

# Reference ground state |0><0| that DBAC/HBAC try to reset towards.
state_0 = np.array([1, 0], dtype=complex)


def fidelity_with_0(state, reference=state_0):
    """
    Fidelity <reference| rho |reference> of a pure state vector or density
    matrix with the reference state (defaults to |0>).
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
    """Draw a Haar-random density matrix of the given dimension."""
    a = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    return rho


def thermal_state(epsilon):
    """Diagonal thermal (Gibbs) state of a single qubit with polarization epsilon."""
    z = np.exp(epsilon) + np.exp(-epsilon)
    return np.array([[np.exp(epsilon) / z, 0], [0, np.exp(-epsilon) / z]])


def apply_depolarizing_noise(rho, p=0.5):
    """Mix a density matrix with the maximally mixed state: (1-p) rho + p I/d."""
    d = rho.shape[0]
    return (1 - p) * rho + (p / d) * np.eye(d)


def rx_sweep(theta_fraction, start=np.array([[1], [0]], dtype=complex)):
    """
    Apply exp(-i * theta * X) to `start`, with theta = theta_fraction * pi/2,
    for each value in theta_fraction. This is the initial-state sweep used
    throughout the DBAC notebooks. Note theta is the state-vector angle
    (cos(theta)|0> - i sin(theta)|1> for start=|0>), i.e. half the Bloch-sphere
    rotation angle: in the standard convention this is R_X(2 * theta).

    Returns an array of shape (len(theta_fraction), 2, 1).
    """
    sigma_x = np.array([[0, 1], [1, 0]])
    return np.array([expm(-1j * np.pi / 2 * sigma_x * t) @ start for t in theta_fraction])
