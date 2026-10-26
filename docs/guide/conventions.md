---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# States and conventions

Everything in `dbac` acts on a **single qubit**: the *data qubit* we want to cool. This page
fixes the conventions that the rest of the package uses: the Hamiltonian, what "energy" and
"fidelity" mean, and how initial states are parametrized. If a number from `dbac` ever
looks surprising, the answer is usually on this page.

## The Hamiltonian and the target state

DBAC cools towards the ground state of

$$
\hat H = -Z = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}.
$$

So the target is $|0\rangle$ with energy $-1$, and the most excited state is $|1\rangle$ with
energy $+1$. The matrix is available as `dbac.dbqite.H`:

```{code-cell} ipython3
import numpy as np
from dbac.dbqite import H, eigs, target_energy

print(H.real)
print("eigenvalues:", eigs, "  target energy:", target_energy)
```

## Two numbers that measure cooling

For a state $\rho$ of the data qubit:

| quantity | formula | cold | hot |
|---|---|---|---|
| energy | $E = \mathrm{Tr}(\rho \hat H)$ | $-1$ at $\lvert 0\rangle$ | $+1$ at $\lvert 1\rangle$ |
| fidelity with the ground state | $F = \langle 0\rvert\rho\lvert 0\rangle$ | $1$ | $0$ |

They carry the same information for a qubit, since $E = 1 - 2F$. The paper's main figure
uses energy; the supplementary figures mostly use fidelity.

```{code-cell} ipython3
from dbac.utils import fidelity_with_0, state_0

plus = np.array([[1], [1]]) / np.sqrt(2)   # |+>, halfway between |0> and |1>
rho = plus @ plus.conj().T

print(f"F = {fidelity_with_0(plus):.3f}")    # works with a state vector...
print(f"F = {fidelity_with_0(rho):.3f}")     # ...or a density matrix
print(f"E = {np.trace(rho @ H).real:.3f}")
```

## State vectors and density matrices

Functions accept either form wherever it makes sense:

- a **state vector** is a column of shape `(2, 1)`, for example `np.array([[1], [0]])` for $|0\rangle$;
- a **density matrix** has shape `(2, 2)`.

DBAC steps always *return* a density matrix, because the DME subroutine is a channel: it
traces out the instruction qubit, so its output is in general mixed.

:::{note}
`dbac.utils.state_0` is the flat vector `[1, 0]`, used as the reference state inside
`fidelity_with_0`. When you build a state to *evolve*, use the column shape `(2, 1)` instead.
:::

## Sweeping the initial state

The paper studies how well DBAC cools as a function of how far the initial state is from
$|0\rangle$. Every sweep in the package uses `rx_sweep`, which rotates a starting state by
$e^{-i\theta X}$:

```{code-cell} ipython3
from dbac.utils import rx_sweep

fractions = [0, 0.25, 0.5, 0.75, 1]        # theta = fraction * pi/2
for f, psi in zip(fractions, rx_sweep(fractions)):
    E = round((psi.conj().T @ H @ psi).item().real, 6) + 0.0   # + 0.0 turns -0.0 into 0.0
    print(f"theta = {f:4.2f} * pi/2   F = {fidelity_with_0(psi):.3f}   E = {E:+.3f}")
```

:::{important}
`rx_sweep` takes the **fraction** of $\pi/2$, not the angle itself, and the angle is the
*state-vector* angle: $e^{-i\theta X}|0\rangle = \cos\theta|0\rangle - i\sin\theta|1\rangle$.
This matches the paper's convention $R_X(\theta) = e^{-i\theta X}$, so $\theta = \pi/2$ reaches
$|1\rangle$. In the more common convention $R_X(\phi) = e^{-i\phi X/2}$, it is $\phi = 2\theta$.
:::

The energy along this sweep is $E_0(\theta) = -\cos 2\theta$, the dotted *Initial state*
curve on the [front page](../index.ipynb).

## Mixed and thermal states

Two helpers build mixed inputs, which are useful for comparing DBAC with entropy-based
cooling:

```{code-cell} ipython3
from dbac.utils import apply_depolarizing_noise, thermal_state

print(apply_depolarizing_noise(rho, p=0.3).real)   # (1 - p) rho + p I/2
print(thermal_state(0.5))                           # diag(e^eps, e^-eps) / Z
```

## Next

With the conventions fixed, the next page explains the subroutine that does all the work:
[density matrix exponentiation](dme.md).
