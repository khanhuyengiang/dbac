"""Tests for `dbac.dme` and `dbac.dbqite.DME`: the density matrix exponentiation channel."""

import numpy as np
import pytest

from conftest import random_pure_dm
from dbac.dbqite import DME
from dbac.dme import dme_channel, dme_channel_equivalent, dme_expm, dme_qibo, ptrace

IMPLEMENTATIONS = [dme_channel, dme_channel_equivalent, dme_qibo]


def test_ptrace_of_product_state(rng):
    a, b = random_pure_dm(rng), random_pure_dm(rng)
    np.testing.assert_allclose(ptrace(np.kron(a, b), index=1), a, atol=1e-12)
    np.testing.assert_allclose(ptrace(np.kron(a, b), index=2), b, atol=1e-12)


def test_ptrace_unequal_dimensions(rng):
    a = random_pure_dm(rng)
    b = np.eye(4) / 4
    np.testing.assert_allclose(ptrace(np.kron(a, b), index=1, n1=2, n2=4), a, atol=1e-12)


@pytest.mark.parametrize("M", [1, 3, 10])
def test_implementations_agree(rng, M):
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    reference = dme_channel(data, instruction, 0.8, M)[0]
    for dme in IMPLEMENTATIONS[1:]:
        np.testing.assert_allclose(dme(data, instruction, 0.8, M)[0], reference, atol=1e-12)


@pytest.mark.parametrize("dme", IMPLEMENTATIONS)
def test_history_has_one_state_per_step(rng, dme):
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    final, history = dme(data, instruction, 1.0, 7)
    assert len(history) == 7
    np.testing.assert_allclose(history[-1], final)


@pytest.mark.parametrize("dme", IMPLEMENTATIONS)
def test_channel_output_is_a_density_matrix(rng, dme):
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    out = dme(data, instruction, 1.2, 4)[0]
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)
    assert np.trace(out).real == pytest.approx(1)
    assert np.linalg.eigvalsh(out).min() > -1e-12


@pytest.mark.parametrize("dme", IMPLEMENTATIONS)
def test_full_swap_hands_over_the_instruction(rng, dme):
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    np.testing.assert_allclose(dme(data, instruction, np.pi / 2, 1)[0], instruction, atol=1e-12)


def test_trotter_error_scales_as_one_over_M(rng):
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    exact = dme_expm(data, instruction, 1.0)
    e100 = np.linalg.norm(exact - dme_channel(data, instruction, 1.0, 100)[0])
    e1000 = np.linalg.norm(exact - dme_channel(data, instruction, 1.0, 1000)[0])
    assert e100 / e1000 == pytest.approx(10, rel=0.05)


def test_dme_expm_is_unitary_conjugation(rng):
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    out = dme_expm(data, instruction, 0.9)
    assert np.trace(out @ out).real == pytest.approx(1)   # stays pure
    np.testing.assert_allclose(dme_expm(data, instruction, 0.0), data)


def test_commuting_states_are_unchanged():
    zero = np.diag([1, 0]).astype(complex)
    for dme in IMPLEMENTATIONS:
        np.testing.assert_allclose(dme(zero, zero, 0.7, 3)[0], zero, atol=1e-12)


def test_dbqite_DME_takes_instruction_first(rng):
    data, instruction = random_pure_dm(rng), random_pure_dm(rng)
    np.testing.assert_allclose(DME(instruction, data, 0.8, 5),
                               dme_channel_equivalent(data, instruction, 0.8, 5)[0])
