# dbac.dme

Density matrix exponentiation (DME): applying $e^{it\rho}$ to a data state $\sigma$ using
copies of $\rho$ instead of knowledge of $\rho$. This is the subroutine inside every DBAC step.

The module offers the same channel four ways, so the implementations can be checked against
each other. All take `(data, instruction, t, M)`: the data and instruction density matrices, the
total duration, and the number of Trotter steps (instruction copies used).

See [Density matrix exponentiation](../guide/dme.md) for how DME works and which
implementation to pick.

:::{note}
DBAC steps do not call this module: they use `dbac.dbqite.DME`, which takes the arguments in
the order `(instruction, data, t, M)`.
:::

```{eval-rst}
.. currentmodule:: dbac.dme

.. autosummary::

   dme_channel
   dme_channel_equivalent
   dme_qibo
   dme_expm
   ptrace
```

## Approximate DME (Trotterized)

```{eval-rst}
.. autofunction:: dme_channel

.. autofunction:: dme_channel_equivalent

.. autofunction:: dme_qibo
```

## Exact reference

```{eval-rst}
.. autofunction:: dme_expm
```

## Helpers

```{eval-rst}
.. autofunction:: ptrace
```
