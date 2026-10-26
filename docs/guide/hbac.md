---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# HBAC baseline

Heat-bath algorithmic cooling (HBAC) is the standard algorithmic-cooling protocol. The package
includes a minimal implementation in `dbac.hbac` so that DBAC can be compared with it on the
same axes. This page explains what HBAC does differently and how to run it.

## Entropy versus coherence

HBAC cools by moving **entropy**. It treats the qubits as a classical mixture of $|0\rangle$ and
$|1\rangle$, sorts the populations so that the target qubit holds the most polarized ones, and
dumps the excess entropy into a *reset qubit* that is re-thermalized with a cold bath every
round.

DBAC solves a different problem: the qubits are in a **pure** state with no entropy at all, and
the task is to remove their **coherence**. HBAC cannot do this directly. Its sorting step only
looks at the populations, so it first has to throw the coherence away, which turns the pure
state into a mixed one with *more* entropy.

| | HBAC | DBAC |
|---|---|---|
| removes | entropy | coherence |
| input | mixed (thermal) states | copies of a pure state |
| extra resource | a reset qubit coupled to a cold bath | instruction copies of the state |
| operations | population sorting, multi-qubit gates (Toffolis) | partial SWAPs, single-qubit rotations |
| measurements | none | none |

## One round of HBAC

Each round has two parts:

1. **refresh**: replace the last qubit with a thermal state of polarization $\epsilon_b$,
   i.e. populations $(1\pm\epsilon_b)/2$ (`refresh_reset_qubit`);
2. **compress**: reorder the diagonal of the joint density matrix in decreasing order, which is
   what the optimal sorting unitary achieves; coherences are dropped (`compress_density_matrix`).

`hbac(rho, epsilon_b, step)` runs `step` rounds on an $n$-qubit density matrix and returns the
reduced state of the **first** qubit, the one being cooled. The **last** qubit is the reset qubit.

```{code-cell} ipython3
import numpy as np
from dbac.hbac import hbac
from dbac.utils import rx_sweep, fidelity_with_0

psi = rx_sweep([0.5])[0]                 # the same theta = pi/4 state used in the DBAC guide
rho = psi @ psi.conj().T
rho3 = np.kron(np.kron(rho, rho), rho)   # three qubits, all in |psi>

for rounds in (1, 3, 6, 20):
    target = hbac(rho3, epsilon_b=0.8, step=rounds)
    print(f"{rounds:2d} round(s): F = {fidelity_with_0(target):.3f}")
```

Repeated rounds push the target past the bath's own value $(1+\epsilon_b)/2 = 0.9$, but only up
to a limit set by the bath polarization and the number of qubits (about $0.988$ here). Like
DBAC, HBAC cannot reach $|0\rangle$ exactly with finite resources.

:::{note}
`hbac` parametrizes the bath by its polarization $\epsilon_b$, with populations
$(1\pm\epsilon_b)/2$. `dbac.utils.thermal_state(epsilon)` instead uses a Boltzmann factor,
$\mathrm{diag}(e^{\epsilon}, e^{-\epsilon})/Z$. They are different parametrizations of the same
family of states.
:::

## Comparing with DBAC

The same sweep over initial states as in the DBAC guide, now through three-qubit HBAC:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from dbac.gci import GCI
from dbac.dbqite import QITE_dme_select_M

psis = rx_sweep(np.linspace(0, 1, 40), start=np.array([[0], [1]], dtype=complex))
F_initial = [fidelity_with_0(p) for p in psis]

def three_copies(p):
    r = p @ p.conj().T
    return np.kron(np.kron(r, r), r)

F_hbac = [fidelity_with_0(hbac(three_copies(p), epsilon_b=0.8, step=6)) for p in psis]
F_dbac = [GCI(QITE_dme_select_M(2), p, None, [np.pi / 4]).execute_gci()[-1][-1] for p in psis]

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(F_initial, F_initial, ":", label="initial state")
ax.plot(F_initial, F_hbac, color="#006f7a", label="HBAC, 3 qubits, 6 rounds")
ax.plot(F_initial, F_dbac, color="#2c0084", label="DBAC, 3 qubits ($k=1$, $M=2$)")
ax.set_xlabel("fidelity of initial state with $|0\\rangle$")
ax.set_ylabel("fidelity after cooling")
ax.legend();
```

HBAC ends up colder here, but for a different reason. Every round it pulls in a fresh qubit from
a cold bath ($\epsilon_b = 0.8$ means the reset qubit is already 90% in $|0\rangle$), and that
bath does the cooling: the result barely depends on the input state. DBAC has no bath. Its only
resource is copies of the same unknown state, and what it removes is that state's coherence, the
quantity HBAC throws away in its sorting step. The two curves answer different questions.

The comparison that matters in practice is cost. In the paper's experiment HBAC is far more expensive: the three-qubit sorting needs three
Toffoli gates (about two dozen CZ gates), while a DBAC step had a process fidelity comparable to
a single CZ gate. `notebooks/hbac.ipynb` extends this comparison to depolarized inputs.

## Next

The last guide page shows the [plotting helpers](plotting.md) used to make every figure.
