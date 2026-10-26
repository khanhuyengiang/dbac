# dbac.plotting

Every figure in the paper and the notebooks is drawn by a function in this package. All
functions are re-exported at the top level, so `from dbac.plotting import plot_gci_results`
works for every function on this page. Functions that create a figure accept `save_path=` to
save it.

See [Plotting](../guide/plotting.md) for examples of the most common plots.

```{eval-rst}
.. currentmodule:: dbac.plotting

.. autosummary::

   style.use_paper_style
   style.figure_size
   cooling.plot_single_energy_curve
   cooling.plot_energy_vs_theta
   cooling.plot_fidelity_fidelity
   cooling.plot_energy_grid
   cooling.plot_ideal_dme_summary
   cooling.plot_optimal_s_vs_M
   cooling.plot_max_excitation_vs_k
   cooling.plot_hbac_fidelity_fidelity
   circuits.draw_dbac_circuit
   circuits.draw_circuit
   gci.plot_gci_results
   bloch.plot_bloch_trajectory
   bloch.plot_bloch_sphere
   bloch.plot_bloch_vector
   bloch.density_matrix_to_bloch
   bloch.add_3d_outline
   ptm.plot_ptm
   ptm.plot_ptm_sweep
   ptm.plot_process_matrix
   dme.plot_dme_channel_norms
```

## Style

Paper-ready fonts, line widths and figure sizes.

```{eval-rst}
.. currentmodule:: dbac.plotting.style

.. autofunction:: use_paper_style

.. autofunction:: figure_size

.. autodata:: JOURNAL_WIDTHS
   :no-value:
```

## Cooling curves

Energy or fidelity after DBAC (or HBAC) against the initial state: the paper's main plots.

```{eval-rst}
.. currentmodule:: dbac.plotting.cooling

.. autofunction:: plot_single_energy_curve

.. autofunction:: plot_energy_vs_theta

.. autofunction:: plot_fidelity_fidelity

.. autofunction:: plot_energy_grid

.. autofunction:: plot_ideal_dme_summary

.. autofunction:: plot_optimal_s_vs_M

.. autofunction:: plot_max_excitation_vs_k

.. autofunction:: plot_hbac_fidelity_fidelity

.. autodata:: DEFAULT_COLORS
   :no-value:

.. autodata:: DEFAULT_MARKERS
   :no-value:
```

## Circuits

Drawings of the hardware circuits, without a qiskit or LaTeX dependency.

```{eval-rst}
.. currentmodule:: dbac.plotting.circuits

.. autofunction:: draw_dbac_circuit

.. autofunction:: draw_circuit

.. autodata:: DBAC_CIRCUITS
   :no-value:
```

## Recursion output

```{eval-rst}
.. currentmodule:: dbac.plotting.gci

.. autofunction:: plot_gci_results
```

## Bloch sphere

```{eval-rst}
.. currentmodule:: dbac.plotting.bloch

.. autofunction:: plot_bloch_trajectory

.. autofunction:: plot_bloch_sphere

.. autofunction:: plot_bloch_vector

.. autofunction:: density_matrix_to_bloch

.. autofunction:: add_3d_outline
```

## Process matrices

Pauli-transfer matrices from process tomography, as heatmaps.

```{eval-rst}
.. currentmodule:: dbac.plotting.ptm

.. autofunction:: plot_ptm

.. autofunction:: plot_ptm_sweep

.. autofunction:: plot_process_matrix

.. autodata:: pauli_labels
   :no-value:

.. autodata:: pauli_labels_2q
   :no-value:
```

## DME accuracy

```{eval-rst}
.. currentmodule:: dbac.plotting.dme

.. autofunction:: plot_dme_channel_norms
```
