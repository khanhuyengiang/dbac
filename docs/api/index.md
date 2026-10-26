# API reference

The package has two halves: the physics in `dbac`, and the plotting in `dbac.plotting`. The
physics modules never import matplotlib.

If you are new to the package, the [guide](../guide/conventions.md) explains each module with
runnable examples. These pages list every public function and its arguments.

| module | what it is for | guide |
|---|---|---|
| [`dbac.utils`](utils.md) | building initial states, fidelity, noise | [States and conventions](../guide/conventions.md) |
| [`dbac.dme`](dme.md) | density matrix exponentiation, four implementations | [Density matrix exponentiation](../guide/dme.md) |
| [`dbac.dbqite`](dbqite.md) | one DBAC step, the Hamiltonian | [One DBAC step](../guide/dbac-step.md) |
| [`dbac.gci`](gci.md) | running several steps, reading the results | [Running the recursion](../guide/recursion.md) |
| [`dbac.hbac`](hbac.md) | heat-bath algorithmic cooling, for comparison | [HBAC baseline](../guide/hbac.md) |
| [`dbac.plotting`](plotting.md) | every figure: cooling curves, circuits, Bloch sphere, PTMs | [Plotting](../guide/plotting.md) |

## How the pieces fit together

A typical calculation touches the modules in this order:

1. `dbac.utils.rx_sweep` builds the initial states;
2. `dbac.dbqite.QITE_dme_select_M` makes a DBAC step function, which uses
   [DME](dme.md) internally;
3. `dbac.gci.GCI` runs that step $k$ times and records energy and fidelity;
4. `dbac.plotting` draws the result.

```python
import numpy as np
from dbac.utils import rx_sweep
from dbac.dbqite import QITE_dme_select_M
from dbac.gci import GCI

psi = rx_sweep([0.5])[0]                                    # 1. initial state
step = QITE_dme_select_M(1)                                 # 2. step with M = 1
results = GCI(step, psi, None, [np.pi / 4] * 2).execute_gci()  # 3. k = 2 steps
print(results[-1])                                          # fidelity after 0, 1, 2 steps
```
