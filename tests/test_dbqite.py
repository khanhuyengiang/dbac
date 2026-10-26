"""Tests for `dbac.dbqite`: one DBAC (DB-QITE) step."""

import numpy as np
import pytest

from dbac.dbqite import QITE_GCI_dme, QITE_dme_select_M, H, eigs, ideal_DME, initial_psi, target_energy
from dbac.utils import fidelity_with_0, rx_sweep

ZERO = np.array([[1], [0]], dtype=complex)
ONE = np.array([[0], [1]], dtype=complex)


def energy(rho):
    return np.trace(rho @ H).real


def closed_form_energy(E0, s):
    """Eq. 3 of the paper: energy after one ideal DBAC step."""
    return E0 - 2 * np.sin(s) ** 2 * (1 - E0 ** 2) * ((1 - np.cos(s)) * E0 + np.cos(s))


def test_hamiltonian_constants():
    np.testing.assert_allclose(H, -np.diag([1, -1]))
    np.testing.assert_allclose(np.linalg.eigvalsh(H), eigs)
    assert target_energy == eigs.min()


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("s", [0.3, np.pi / 4, 1.1])
def test_ideal_step_matches_closed_form(fraction, s):
    psi = rx_sweep([fraction])[0]
    E0 = (psi.conj().T @ H @ psi).item().real
    assert energy(ideal_DME(psi, s)) == pytest.approx(closed_form_energy(E0, s))


@pytest.mark.parametrize("step", [QITE_dme_select_M(1), QITE_dme_select_M(3), ideal_DME])
@pytest.mark.parametrize("pole, E", [(ZERO, -1), (ONE, 1)])
def test_poles_are_fixed_points(step, pole, E):
    assert energy(step(pole, np.pi / 4)) == pytest.approx(E)


@pytest.mark.parametrize("M", [1, 2, 5])
def test_step_cools_every_intermediate_state(M):
    for psi in rx_sweep(np.linspace(0.05, 0.95, 10)):
        assert fidelity_with_0(QITE_dme_select_M(M)(psi, np.pi / 4)) > fidelity_with_0(psi)


def test_more_dme_steps_approach_ideal():
    psi = rx_sweep([0.5])[0]
    ideal = energy(ideal_DME(psi, np.pi / 4))
    gaps = [abs(energy(QITE_dme_select_M(M)(psi, np.pi / 4)) - ideal) for M in (1, 10, 100)]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-2


def test_documented_values_for_half_state():
    psi = rx_sweep([0.5])[0]
    assert energy(QITE_dme_select_M(1)(psi, np.pi / 4)) == pytest.approx(-0.5)
    assert energy(ideal_DME(psi, np.pi / 4)) == pytest.approx(-1 / np.sqrt(2))


def test_zero_duration_does_nothing():
    psi = rx_sweep([0.4])[0]
    rho = psi @ psi.conj().T
    np.testing.assert_allclose(QITE_dme_select_M(2)(psi, 0.0), rho, atol=1e-12)
    np.testing.assert_allclose(ideal_DME(psi, 0.0), rho, atol=1e-12)


def test_ideal_step_keeps_pure_states_pure():
    rho = ideal_DME(rx_sweep([0.6])[0], 0.9)
    assert np.trace(rho @ rho).real == pytest.approx(1)


def test_initial_psi_is_identity():
    psi = rx_sweep([0.3])[0]
    assert initial_psi(psi, 0.5) is psi


def test_default_step_uses_100_dme_steps():
    psi = rx_sweep([0.45])[0]
    np.testing.assert_allclose(QITE_GCI_dme(psi, 0.7), QITE_dme_select_M(100)(psi, 0.7))


def test_step_functions_carry_plot_labels():
    step = QITE_dme_select_M(1)
    step(rx_sweep([0.2])[0], 0.5)
    assert step.label == "DME"
