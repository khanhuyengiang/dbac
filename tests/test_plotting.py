"""Smoke tests for `dbac.plotting`: every figure builds without errors."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dbac.dbqite import QITE_dme_select_M, H, ideal_DME
from dbac.gci import GCI
from dbac.plotting import (
    DBAC_CIRCUITS, draw_circuit, draw_dbac_circuit, figure_size, plot_bloch_trajectory,
    plot_dme_channel_norms, plot_energy_vs_theta, plot_fidelity_fidelity, plot_gci_results,
    plot_ptm, plot_single_energy_curve, use_paper_style,
)
from dbac.utils import rx_sweep

FRACTIONS = np.linspace(0, 1, 20)
THETA = FRACTIONS * np.pi / 2
PSIS = rx_sweep(FRACTIONS)
E_INITIAL = [(p.conj().T @ H @ p).item().real for p in PSIS]
E_AFTER = [np.trace(QITE_dme_select_M(1)(p, np.pi / 4) @ H).real for p in PSIS]


def test_use_paper_style_without_tex():
    with plt.rc_context():
        use_paper_style(use_tex=False, font_size=9, figsize=figure_size("PRD", "onecol"))
        assert plt.rcParams["text.usetex"] is False
        assert plt.rcParams["font.size"] == 9


def test_figure_size_aspect():
    width, height = figure_size("PRD", "twocol", aspect=2)
    assert width == pytest.approx(2 * height)


@pytest.mark.parametrize("name", sorted(DBAC_CIRCUITS))
def test_draw_dbac_circuit(name):
    ax = draw_dbac_circuit(name)
    spec = DBAC_CIRCUITS[name]
    assert len(ax.patches) == len(spec["ops"])   # one box per gate


def test_draw_circuit_on_given_axis():
    _, ax = plt.subplots()
    spec = {"n": 1, "data": 0, "barriers": [], "ops": [(0, [0], "X")]}
    assert draw_circuit(spec, ax=ax, data_label=False) is ax


def test_circuit_specs_are_consistent():
    for spec in DBAC_CIRCUITS.values():
        assert 0 <= spec["data"] < spec["n"]
        for _, qubits, _ in spec["ops"]:
            assert all(0 <= q < spec["n"] for q in qubits)


def test_single_energy_curve_with_experiment(tmp_path):
    path = tmp_path / "curve.png"
    ax = plot_single_energy_curve(THETA, E_INITIAL, "M=1", E_AFTER, "#1a1a8d", "o",
                                  experimental=(THETA[::4], E_AFTER[::4]), save_path=path)
    assert path.exists()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Initial state", "Analytical", "Experiment"]


def test_energy_and_fidelity_overviews():
    plot_energy_vs_theta(THETA, E_INITIAL, {"a": E_AFTER, "b": E_INITIAL})
    F = [(1 - E) / 2 for E in E_AFTER]
    plot_fidelity_fidelity(FRACTIONS, FRACTIONS, {"a": F})


def test_gci_results_and_bloch():
    psi = rx_sweep([0.6])[0]
    res = GCI(QITE_dme_select_M(1), psi, None, [np.pi / 4] * 2).execute_gci()
    plot_gci_results(res, GCI(ideal_DME, psi, None, [np.pi / 4] * 2).execute_gci(), target_energy=-1)
    plot_gci_results(res, target_energy=-1, mode="Fidelity")
    plot_bloch_trajectory(res[8], ["0", "1", "2"], target_fidelity=0.9)


def test_ptm_and_dme_norm_plots():
    plot_ptm(np.eye(4), ["I", "X", "Y", "Z"])
    plot_dme_channel_norms({"a": [0.5, 0.2, 0.1], "b": [0.4, 0.2, 0.1]}, show_final_value=True)
