---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Quick start

This page cools one qubit with DBAC in a few lines, then explains what each line did. It
assumes `dbac` is [installed](installation.md).

## Cool a qubit

```{code-cell} ipython3
import numpy as np
from dbac.utils import rx_sweep
from dbac.dbqite import QITE_dme_select_M
from dbac.gci import GCI

psi = rx_sweep([0.5])[0]                  # a qubit halfway between |0> and |1>
step = QITE_dme_select_M(M=1)             # a DBAC step with one DME step
results = GCI(step, psi, None, [np.pi / 4, np.pi / 4]).execute_gci()

for k, F in enumerate(results[-1]):
    print(f"after {k} DBAC steps: fidelity with |0> = {F:.3f}")
```

The qubit started with a 50% overlap with the ground state $|0\rangle$. Two DBAC steps raised
that to 87.5%, without measuring the qubit or knowing its state.

## What just happened

**`rx_sweep([0.5])`** built the initial state $\cos\theta|0\rangle - i\sin\theta|1\rangle$ with
$\theta = 0.5 \cdot \pi/2$. It takes a list because it is usually used to sweep many angles at
once. → [States and conventions](guide/conventions.md)

**`QITE_dme_select_M(M=1)`** created a *step function*: one round of cooling. A step rotates a
copy of the state and uses it as a *quantum instruction*, applied to the qubit through
density matrix exponentiation (DME) with $M$ steps. → [One DBAC step](guide/dbac-step.md),
[Density matrix exponentiation](guide/dme.md)

**`GCI(step, psi, None, [s1, s2])`** ran the step twice, with durations $s_1 = s_2 = \pi/4$,
each time on the output of the previous step. `execute_gci()` returned a tuple of results;
`results[-1]` holds the fidelity after each step. → [Running the recursion](guide/recursion.md)

## Try changing

- **more DME steps**: `QITE_dme_select_M(M=2)` uses two instruction copies per step and cools more;
- **more DBAC steps**: a longer schedule `[np.pi / 4] * 4` keeps cooling, with diminishing returns;
- **a hotter state**: `rx_sweep([0.9])` starts closer to $|1\rangle$. Note that $|1\rangle$ itself
  (`rx_sweep([1])`) cannot be cooled at all: it is an excited eigenstate.

```{code-cell} ipython3
for M in (1, 2):
    results = GCI(QITE_dme_select_M(M), psi, None, [np.pi / 4] * 4).execute_gci()
    print(f"M = {M}:", np.round(results[-1], 3))
```

## Next steps

- The [front page](index.ipynb) shows the same calculation for the paper's three hardware
  circuits, next to the measured data.
- The guide, starting with [States and conventions](guide/conventions.md), explains each
  module in depth.
- [Reproducing the figures](figures.md) lists the scripts that regenerate every figure of the paper.
