# dbac.dbqite

One step of DBAC, called a *DB-QITE step* (double-bracket quantum imaginary-time evolution) in
the code. A step takes the current state of the data qubit and a duration $s$ and returns a
colder state:

$$
\rho \mapsto e^{is\rho'}\,\rho\,e^{-is\rho'}, \qquad \rho' = e^{is\hat H}\rho\, e^{-is\hat H},
\qquad \hat H = -Z .
$$

All step functions share the signature `step(psi, s) -> density matrix`, so any of them can be
handed to `dbac.gci.GCI` to run several steps. They differ only in how $e^{is\rho'}$ is applied.

See [One DBAC step](../guide/dbac-step.md) for a walkthrough with plots.

```{eval-rst}
.. currentmodule:: dbac.dbqite

.. autosummary::

   QITE_dme_select_M
   ideal_DME
   initial_psi
   QITE_GCI_dme
   DME
   H
   target_energy
```

## Step functions

```{eval-rst}
.. autofunction:: QITE_dme_select_M

.. autofunction:: ideal_DME

.. autofunction:: initial_psi

.. autofunction:: QITE_GCI_dme
```

## DME inside the step

```{eval-rst}
.. autofunction:: DME
```

## The Hamiltonian

```{eval-rst}
.. autodata:: H
   :no-value:

.. autodata:: eigs
   :no-value:

.. autodata:: target_energy
   :no-value:

.. autodata:: H_sh
   :no-value:
```
