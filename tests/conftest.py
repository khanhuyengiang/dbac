"""Shared pytest setup: a non-interactive matplotlib backend and figure cleanup."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_pure_dm(rng):
    """Random single-qubit pure state as a density matrix."""
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())
