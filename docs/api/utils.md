# dbac.utils

Helpers for preparing single-qubit states and measuring how cold they are. Use `rx_sweep` to
build the initial states of a sweep and `fidelity_with_0` to score the result; the noise
helpers build mixed inputs.

See [States and conventions](../guide/conventions.md) for the conventions behind these
functions, in particular the angle convention of `rx_sweep`.

```{eval-rst}
.. currentmodule:: dbac.utils

.. autosummary::

   rx_sweep
   fidelity_with_0
   state_0
   apply_depolarizing_noise
   thermal_state
   random_density_matrix
```

## Initial states

```{eval-rst}
.. autofunction:: rx_sweep

.. autodata:: state_0
   :no-value:
```

## Measuring cooling

```{eval-rst}
.. autofunction:: fidelity_with_0
```

## Mixed states

```{eval-rst}
.. autofunction:: apply_depolarizing_noise

.. autofunction:: thermal_state

.. autofunction:: random_density_matrix
```
