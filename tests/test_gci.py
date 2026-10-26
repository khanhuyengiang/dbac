"""Tests for `dbac.gci`: running the DBAC recursion and reading its output."""

import numpy as np
import pytest

from dbac.dbqite import QITE_dme_select_M, H, ideal_DME
from dbac.gci import GCI, max_excitation
from dbac.utils import fidelity_with_0, rx_sweep


def test_output_has_one_entry_per_step_plus_initial():
    psi = rx_sweep([0.5])[0]
    results = GCI(QITE_dme_select_M(1), psi, None, [0.7, 0.5, 0.3]).execute_gci()
    assert len(results) == 11
    energies, s_mins, variances, durations = results[:4]
    states, fidelities = results[8], results[10]
    for seq in (energies, s_mins, variances, durations, states, fidelities):
        assert len(seq) == 4


def test_documented_fidelities():
    psi = rx_sweep([0.5])[0]
    fidelities = GCI(QITE_dme_select_M(1), psi, None, [np.pi / 4] * 2).execute_gci()[-1]
    np.testing.assert_allclose(fidelities, [0.5, 0.75, 0.875])


def test_schedule_as_numpy_array_keeps_leading_zero():
    # Regression: `[0] + np.array(...)` used to broadcast instead of prepending.
    psi = rx_sweep([0.5])[0]
    s_mins = GCI(QITE_dme_select_M(1), psi, None, np.array([0.7, 0.5])).execute_gci()[1]
    np.testing.assert_allclose(s_mins, [0, 0.7, 0.5])


def test_durations_are_cumulative_squares():
    psi = rx_sweep([0.5])[0]
    durations = GCI(QITE_dme_select_M(1), psi, None, [0.7, 0.5]).execute_gci()[3]
    np.testing.assert_allclose(durations, [0, 0.49, 0.74])


def test_final_state_and_fidelities_are_consistent():
    psi = rx_sweep([0.6])[0]
    results = GCI(QITE_dme_select_M(2), psi, None, [0.8, 0.6]).execute_gci()
    energies, states, final, fidelities = results[0], results[8], results[9], results[10]
    assert final is states[-1]
    for E, rho, F in zip(energies, states, fidelities):
        assert F == pytest.approx(fidelity_with_0(rho))
        assert np.real(E) == pytest.approx(1 - 2 * F)


def test_schedule_matches_manual_steps():
    psi = rx_sweep([0.6])[0]
    step = QITE_dme_select_M(1)
    final = GCI(step, psi, None, [0.8, 0.4]).execute_gci()[9]
    np.testing.assert_allclose(final, step(step(psi, 0.8), 0.4))


def test_recursion_keeps_cooling():
    psi = rx_sweep([0.7])[0]
    fidelities = GCI(QITE_dme_select_M(1), psi, None, [np.pi / 4] * 4).execute_gci()[-1]
    assert all(b > a for a, b in zip(fidelities, fidelities[1:]))


def test_line_search_picks_lowest_energy_candidate():
    psi = rx_sweep([0.6])[0]
    candidates = list(np.linspace(0.1, 1.5, 15))
    s_chosen = GCI(ideal_DME, psi, 1, candidates).execute_gci()[1][1]
    energies = [np.trace(ideal_DME(psi, s) @ H).real for s in candidates]
    assert s_chosen == candidates[int(np.argmin(energies))]


def test_line_search_warns_at_edge_of_interval():
    psi = rx_sweep([0.6])[0]
    with pytest.warns(UserWarning, match="edge of interval"):
        GCI(ideal_DME, psi, 1, [0.01, 0.02, 0.03]).execute_gci()


def test_max_excitation():
    x, F = [0.1, 0.2, 0.3, 0.4], [0.5, 0.7, 0.9, 0.95]
    assert max_excitation(x, F, 0.9) == 0.3
    assert max_excitation(x, F, 0.69) == 0.2      # found once the tolerance has grown
    assert max_excitation(x, F, 0.1) is None      # out of reach
