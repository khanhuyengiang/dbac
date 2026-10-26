"""
GCI: Group Commutator Iteration, the driver that runs the DBAC recursion.

`GCI` applies a step function (from `dbac.dbqite`) repeatedly, feeding each
output state back in as the next input, and records energy, fidelity and the
states along the way. Plotting of the results produced by
`GCI.execute_gci` lives in `dbac.plotting.gci.plot_gci_results`, so this module has
no matplotlib dependency.
"""

__all__ = [
    "GCI",
    "max_excitation"
]

import warnings

from qibo import hamiltonians, Circuit
from qibo.symbols import Z
import numpy as np

class GCI:
    """
    Run the DBAC recursion: apply a step function `k` times, each time to the
    output of the previous step, and track energy and fidelity.

    The step durations are chosen in one of two modes, selected by `nmb_steps`:

    - **fixed schedule** (``nmb_steps=None``): `times_dbr` lists the duration
      of each step, ``[s_1, ..., s_k]``.
    - **greedy line search** (``nmb_steps=k``): `times_dbr` lists candidate
      durations; every step uses the candidate that gives the lowest energy.
      A warning is emitted if the best candidate is at the edge of the list.

    Nothing is computed until `execute_gci` is called.

    Parameters:
        gci_step (Callable): Step function ``step(psi, s) -> state``, e.g.
            ``dbac.dbqite.QITE_dme_select_M(1)`` or ``dbac.dbqite.ideal_DME``.
        state_of_recursion (np.ndarray): Initial state, a (2, 1) state vector
            or 2x2 density matrix (or a qibo ``Circuit``).
        nmb_steps (int or None): Number of steps for the line-search mode, or
            None for the fixed-schedule mode.
        times_dbr (sequence[float]): The schedule (fixed mode) or the
            candidate durations (line-search mode).
        please_be_verbose (bool): Unused; kept for backwards compatibility.

    Example:
        >>> import numpy as np
        >>> from dbac.gci import GCI
        >>> from dbac.dbqite import QITE_dme_select_M
        >>> from dbac.utils import rx_sweep
        >>> psi = rx_sweep([0.5])[0]
        >>> results = GCI(QITE_dme_select_M(1), psi, None, [np.pi / 4, np.pi / 4]).execute_gci()
        >>> [round(float(F), 3) for F in results[-1]]   # fidelity with |0> after 0, 1, 2 steps
        [0.5, 0.75, 0.875]
    """
    def __init__(self, gci_step, state_of_recursion, nmb_steps, times_dbr, please_be_verbose=True):
        self.gci_step = gci_step
        self.state_of_recursion = state_of_recursion
        self.psi0 = state_of_recursion
        self.nmb_steps = nmb_steps
        self.times_dbr = times_dbr
        self.please_be_verbose = please_be_verbose
        self.states = [state_of_recursion]

        # Fixed problem setup for 1-qubit Z Hamiltonian
        self.L = 1
        self.H_sh = hamiltonians.SymbolicHamiltonian(Z(0), nqubits=1)
        self.H = np.array([[-1.-0.j, -0.-0.j],
                           [-0.-0.j,  1.-0.j]])
        self.eigs = np.array([-1., 1.])
        self.target_energy = -1

        # Initial tracking
        self.results = []
        self.fidelities = []
        self.s_mins = [0]
        self.state_0 = np.array([1, 0], dtype=complex)
        self.fidelities.append(self.fidelity_with_0(state_of_recursion))
        self.results.append(self.compute_moments(state_of_recursion))

        if isinstance(state_of_recursion, Circuit):
            self.circuit_depths = [0]
            self.circuits = [state_of_recursion]

    def fidelity_with_0(self, state):
        """Fidelity of a state vector or density matrix with ``|0>``."""
        shape = np.shape(state)
        if shape[0] == shape[1]:  # density matrix
            return np.real(np.conj(self.state_0).T @ state @ self.state_0)
        else:  # state vector
            dm = state @ state.T.conj()
            return np.real(np.conj(self.state_0).T @ dm @ self.state_0)

    def compute_moments(self, psi):
        """
        Energy moments of a state with respect to H = -Z.

        Returns:
            tuple: ``(E, E2, variance)`` with ``E = <H>``, ``E2 = <H^2>`` and
            ``variance = E2 - E^2``. For a density matrix the values are
            complex-typed (take ``np.real``); for a qibo circuit only ``E`` is
            computed.
        """
        if isinstance(psi, Circuit):
            return self.H_sh.expectation(psi(self.psi0.T[0]).state()), None, None
        shape = np.shape(psi)
        if shape[0] == shape[1]:  # density matrix
            E = np.trace(psi @ self.H)
            S = np.trace(psi @ self.H @ self.H)
        else:  # state vector
            E = (psi.conj().T @ self.H @ psi)[0, 0].real
            S = (psi.conj().T @ self.H @ self.H @ psi)[0, 0].real
        return E, S, S - E**2

    def execute_gci(self):
        """
        Run the recursion and return everything it recorded.

        Every list in the result has one entry for the initial state followed
        by one entry per step.

        Returns:
            tuple: An 11-tuple, best read from both ends (``results[0]`` for
            energies, ``results[-1]`` for fidelities, ``results[-2]`` for the
            final state):

            0. energies: ``Tr(rho H)`` after each step (take ``np.real``).
            1. step sizes: the durations used, starting with 0.
            2. variances: energy variance after each step.
            3. durations: cumulative evolution time, the sum of ``s_j ** 2``.
            4. label: display label of the step function.
            5. marker: matplotlib marker of the step function.
            6. circuit depths: for qibo-circuit states only, else None.
            7. circuits: for qibo-circuit states only, else None.
            8. states: the state after each step.
            9. final state: the last entry of `states`.
            10. fidelities: fidelity with ``|0>`` after each step.
        """
        if self.nmb_steps is None:
            for s in self.times_dbr:
                self.state_of_recursion = self.gci_step(self.state_of_recursion, s)
                self.results.append(self.compute_moments(self.state_of_recursion))
                self.fidelities.append(self.fidelity_with_0(self.state_of_recursion))
                self.states.append(self.state_of_recursion)
            self.s_mins = [0] + list(self.times_dbr)
        else:
            for k in range(self.nmb_steps):
                test_es = [self.compute_moments(self.gci_step(self.state_of_recursion, dt))[0] for dt in self.times_dbr]
                ind = np.argmin(test_es)
                if ind == 0 or ind == len(self.times_dbr) - 1:
                    warnings.warn(f"{k} {ind} {self.times_dbr[ind]} --- Warning edge of interval reached")
                s_min = self.times_dbr[ind]
                self.state_of_recursion = self.gci_step(self.state_of_recursion, s_min)
                self.results.append(self.compute_moments(self.state_of_recursion))
                self.fidelities.append(self.fidelity_with_0(self.state_of_recursion))
                self.s_mins.append(s_min)
                self.states.append(self.state_of_recursion)
                if isinstance(self.state_of_recursion, Circuit):
                    self.circuit_depths.append(len(self.state_of_recursion.queue) * 3)
                    self.circuits.append(self.state_of_recursion)

        energies = [r[0] for r in self.results]
        variances = [r[2] for r in self.results]

        # Add default display labels if not already set
        if not hasattr(self.gci_step, 'exponent'):
            self.gci_step.exponent = 2
        if not hasattr(self.gci_step, 'label'):
            self.gci_step.label = ''
        if not hasattr(self.gci_step, 'marker'):
            self.gci_step.marker = '-'

        durations = (
            [sum(s**self.gci_step.exponent for s in self.s_mins[1:x]) for x in range(1, len(self.s_mins) + 1)]
            if self.gci_step.exponent > 0 else self.s_mins
        )

        return (
            energies, self.s_mins, variances,
            durations,
            self.gci_step.label, self.gci_step.marker,
            self.circuit_depths if isinstance(self.state_of_recursion, Circuit) else None,
            self.circuits if isinstance(self.state_of_recursion, Circuit) else None,
            self.states,
            self.state_of_recursion,
            self.fidelities
        )

def max_excitation(thetas, fidelity, target_fidelity, initial_tolerance=1e-5, max_tolerance=0.2, step=1e-3):
    """
    Find where a cooling curve first reaches a target fidelity.

    Scans ``(thetas[i], fidelity[i])`` in order and returns the first
    ``thetas[i]`` whose fidelity is within a tolerance of `target_fidelity`.
    The tolerance starts at `initial_tolerance` and grows by `step` until a
    match is found or it exceeds `max_tolerance`. With a sweep ordered from
    hot to cold, the result is the hottest initial state that is still cooled
    to the target: everything colder ends up at least as cold.

    Parameters:
        thetas (sequence[float]): x-values of the curve, e.g. the initial
            fidelities or angles of a sweep.
        fidelity (sequence[float]): Final fidelity for each x-value.
        target_fidelity (float): Fidelity to look for.
        initial_tolerance (float): Starting tolerance.
        max_tolerance (float): Largest tolerance tried before giving up.
        step (float): Increment of the tolerance.

    Returns:
        float or None: The matching x-value, or None if no point is within
        `max_tolerance` of the target.

    Example:
        >>> from dbac.gci import max_excitation
        >>> max_excitation([0.1, 0.2, 0.3, 0.4], [0.5, 0.7, 0.9, 0.95], target_fidelity=0.9)
        0.3
    """
    tolerance = initial_tolerance
    while tolerance <= max_tolerance:
        for xi, yi in zip(thetas, fidelity):
            if abs(yi - target_fidelity) <= tolerance:
                return xi
        tolerance += step
    return None
