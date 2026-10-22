# dbac

Code accompanying [*Double-Bracket Algorithmic Cooling*](https://arxiv.org/abs/2510.00302).

DBAC removes *coherence* from a pure qubit state by simulating quantum imaginary-time
evolution with recursive double-bracket steps. Each step consumes a copy of the state as a
**quantum instruction** through **density matrix exponentiation (DME)**, so cooling
performance is set by how many copies are available.

The `dbac` package provides DME implementations, the DB-QITE/GCI recursion, the HBAC
baseline, and all plotting code used for the paper figures.

```{toctree}
:maxdepth: 2

installation
quickstart
figures
api/index
```
