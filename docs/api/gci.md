# dbac.gci

Runs the DBAC recursion. `GCI` (group commutator iteration) applies a step function from
`dbac.dbqite` again and again, each time to the previous output, and records the energy,
fidelity and state after every step.

See [Running the recursion](../guide/recursion.md) for both modes of `GCI` and a guide to its
output.

```{eval-rst}
.. currentmodule:: dbac.gci

.. autosummary::

   GCI
   GCI.execute_gci
   max_excitation
```

## Running the recursion

```{eval-rst}
.. autoclass:: GCI
   :members: execute_gci, compute_moments, fidelity_with_0
   :member-order: bysource
```

## Summarizing a cooling curve

```{eval-rst}
.. autofunction:: max_excitation
```
