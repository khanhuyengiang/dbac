"""Sanity checks for the physics core of `dbac`."""

import numpy as np
import pytest

from dbac.dme import dme_expm, dme_channel, dme_channel_equivalent, dme_qibo
from dbac.dbqite import QITE_dme_select_M, ideal_DME
from dbac.gci import GCI
from dbac.hbac import hbac
from dbac.utils import apply_depolarizing_noise, fidelity_with_0, rx_sweep


def random_pure_dm(rng):
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


@pytest.mark.parametrize("dme", [dme_channel, dme_channel_equivalent, dme_qibo])
def test_dme_converges_to_exact(dme):
    rng = np.random.default_rng(0)
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    exact = dme_expm(data, instruction, 1.0)
    errors = [np.linalg.norm(exact - dme(data, instruction, 1.0, M)[0]) for M in (10, 100, 1000)]
    # Trotter error is O(t^2 / M).
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3


@pytest.mark.parametrize("M, k", [(1, 1), (2, 1), (1, 2), (2, 3)])
def test_dbqite_preserves_trace(M, k):
    psi = rx_sweep([0.5])[0]
    final = GCI(QITE_dme_select_M(M), psi, None, [0.7] * k).execute_gci()[9]
    assert np.trace(final).real == pytest.approx(1.0)


def test_dbqite_mixed_input_preserves_trace():
    psi = rx_sweep([0.3])[0]
    rho = apply_depolarizing_noise(psi @ psi.conj().T, p=0.3)
    assert np.trace(QITE_dme_select_M(1)(rho, 0.7)).real == pytest.approx(1.0)


def test_dbqite_vector_and_density_matrix_inputs_agree():
    psi = rx_sweep([0.4])[0]
    rho = psi @ psi.conj().T
    np.testing.assert_allclose(QITE_dme_select_M(2)(psi, 0.6), QITE_dme_select_M(2)(rho, 0.6))
    np.testing.assert_allclose(ideal_DME(psi, 0.6), ideal_DME(rho, 0.6))


def test_ideal_dbqite_cools():
    psi = rx_sweep([0.5])[0]
    final = GCI(ideal_DME, psi, None, [0.7, 0.5]).execute_gci()[9]
    assert fidelity_with_0(final) > fidelity_with_0(psi)


def test_gci_returns_variance():
    psi = rx_sweep([0.3])[0]
    energies, _, variances, *_ = GCI(QITE_dme_select_M(1), psi, None, [0.7]).execute_gci()
    # Z^2 = I, so Var(Z) = 1 - <Z>^2.
    for E, var in zip(energies, variances):
        assert var == pytest.approx(1 - np.real(E) ** 2)


def test_hbac_returns_first_qubit():
    rho = np.diag([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]).astype(complex)
    out = hbac(rho, epsilon_b=0.8, step=2)
    assert out.shape == (2, 2)
    assert np.trace(out).real == pytest.approx(1.0)
