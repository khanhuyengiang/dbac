# dbac.hbac

Heat-bath algorithmic cooling (HBAC), the standard entropy-based protocol, implemented
minimally so DBAC can be compared with it. Each round resets the last qubit to a thermal state
and sorts the populations of the joint state; `hbac` runs several rounds and returns the first
qubit.

See [HBAC baseline](../guide/hbac.md) for how HBAC differs from DBAC and a side-by-side comparison.

```{eval-rst}
.. currentmodule:: dbac.hbac

.. autosummary::

   hbac
   refresh_reset_qubit
   compress_density_matrix
```

## Running HBAC

```{eval-rst}
.. autofunction:: hbac
```

## The two parts of a round

```{eval-rst}
.. autofunction:: refresh_reset_qubit

.. autofunction:: compress_density_matrix
```
