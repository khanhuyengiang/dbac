"""Tests for `dbac.utils`: state preparation, fidelity and noise helpers."""

import numpy as np
import pytest

from dbac.dbqite import H
from dbac.utils import (
    apply_depolarizing_noise, fidelity_with_0, random_density_matrix, rx_sweep, thermal_state,
)

ONE = np.array([[0], [1]], dtype=complex)


def test_rx_sweep_shape_and_endpoints():
    psis = rx_sweep([0, 0.5, 1])
    assert psis.shape == (3, 2, 1)
    assert fidelity_with_0(psis[0]) == pytest.approx(1)
    assert fidelity_with_0(psis[1]) == pytest.approx(0.5)
    assert fidelity_with_0(psis[2]) == pytest.approx(0)


def test_rx_sweep_uses_state_vector_angle():
    theta = 0.3 * np.pi / 2
    psi = rx_sweep([0.3])[0]
    np.testing.assert_allclose(psi.ravel(), [np.cos(theta), -1j * np.sin(theta)])


def test_rx_sweep_from_one():
    psis = rx_sweep([0, 1], start=ONE)
    assert fidelity_with_0(psis[0]) == pytest.approx(0)
    assert fidelity_with_0(psis[1]) == pytest.approx(1)


@pytest.mark.parametrize("fraction", [0, 0.2, 0.5, 0.9, 1])
def test_energy_is_one_minus_twice_fidelity(fraction):
    psi = rx_sweep([fraction])[0]
    E = (psi.conj().T @ H @ psi).item().real
    assert E == pytest.approx(1 - 2 * fidelity_with_0(psi))
    assert E == pytest.approx(-np.cos(2 * fraction * np.pi / 2))


def test_fidelity_accepts_vector_and_density_matrix():
    psi = rx_sweep([0.37])[0]
    assert fidelity_with_0(psi) == pytest.approx(fidelity_with_0(psi @ psi.conj().T))
    assert fidelity_with_0(psi.ravel()) == pytest.approx(fidelity_with_0(psi))


def test_fidelity_custom_reference():
    plus = np.array([1, 1]) / np.sqrt(2)
    assert fidelity_with_0(plus[:, None], reference=plus) == pytest.approx(1)


def test_depolarizing_noise_limits():
    psi = rx_sweep([0.3])[0]
    rho = psi @ psi.conj().T
    np.testing.assert_allclose(apply_depolarizing_noise(rho, p=0), rho)
    np.testing.assert_allclose(apply_depolarizing_noise(rho, p=1), np.eye(2) / 2)
    np.testing.assert_allclose(apply_depolarizing_noise(np.kron(rho, rho), p=1), np.eye(4) / 4)


def test_thermal_state():
    np.testing.assert_allclose(thermal_state(0), np.eye(2) / 2)
    rho = thermal_state(1.3)
    assert np.trace(rho) == pytest.approx(1)
    assert rho[0, 0] > rho[1, 1]
    assert rho[0, 0] / rho[1, 1] == pytest.approx(np.exp(2 * 1.3))


@pytest.mark.parametrize("dim", [2, 4])
def test_random_density_matrix_is_valid(dim):
    rho = random_density_matrix(dim)
    assert rho.shape == (dim, dim)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    assert np.trace(rho).real == pytest.approx(1)
    assert np.linalg.eigvalsh(rho).min() > -1e-12
