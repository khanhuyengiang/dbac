"""Tests for `dbac.hbac`: the heat-bath algorithmic cooling baseline."""

import numpy as np
import pytest

from dbac.dme import ptrace
from dbac.hbac import compress_density_matrix, hbac, refresh_reset_qubit
from dbac.utils import fidelity_with_0, random_density_matrix, rx_sweep

MIXED3 = np.eye(8, dtype=complex) / 8


def test_refresh_sets_last_qubit_to_bath():
    rho = random_density_matrix(8)
    out = refresh_reset_qubit(rho, 0.6)
    np.testing.assert_allclose(ptrace(out, index=2, n1=4, n2=2), np.diag([0.8, 0.2]), atol=1e-12)
    # the other qubits are untouched
    np.testing.assert_allclose(ptrace(out, index=1, n1=4, n2=2), ptrace(rho, index=1, n1=4, n2=2),
                               atol=1e-12)


def test_compress_sorts_populations_and_drops_coherence():
    rho = random_density_matrix(4)
    out = compress_density_matrix(rho)
    np.testing.assert_allclose(out, np.diag(np.diag(out)))
    populations = np.diag(out).real
    assert list(populations) == sorted(populations, reverse=True)
    np.testing.assert_allclose(sorted(populations), sorted(np.diag(rho).real))


def test_compress_rejects_non_hermitian():
    with pytest.raises(AssertionError):
        compress_density_matrix(np.array([[1, 1], [0, 0]], dtype=complex))


def test_hbac_beats_bath_and_saturates():
    fidelities = [fidelity_with_0(hbac(MIXED3, epsilon_b=0.8, step=k)) for k in (1, 3, 6, 20, 40)]
    assert fidelities[0] == pytest.approx(0.9)
    assert all(b >= a - 1e-12 for a, b in zip(fidelities, fidelities[1:]))
    # three-qubit HBAC limit: polarization 2 eps / (1 + eps^2)
    assert fidelities[-1] == pytest.approx((1 + 1.6 / 1.64) / 2, abs=1e-3)


def test_hbac_on_pure_copies_returns_a_qubit_state():
    psi = rx_sweep([0.5])[0]
    r = psi @ psi.conj().T
    out = hbac(np.kron(np.kron(r, r), r), epsilon_b=0.8, step=2)
    assert out.shape == (2, 2)
    assert np.trace(out).real == pytest.approx(1)
