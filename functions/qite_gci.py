"""
GCI: Group Commutator Iteration framework for quantum state evolution and optimization.

This module provides a class-based structure to execute and analyze QITE-like procedures
using energy minimization and fidelity tracking. Plotting of the results produced by
`GCI.execute_gci` lives in `visualization.gci_plots.plot_gci_results`, so this module has
no matplotlib dependency.
"""

__all__ = [
    "GCI",
    "max_excitation"
]

from qibo import hamiltonians, Circuit
from qibo.symbols import Z
import numpy as np

class GCI:
    """
    Class for executing and analyzing Group Commutator Iteration (GCI) steps
    with energy and fidelity tracking.
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
        """Compute fidelity with respect to |0⟩ state."""
        shape = np.shape(state)
        if shape[0] == shape[1]:  # density matrix
            return np.real(np.conj(self.state_0).T @ state @ self.state_0)
        else:  # state vector
            dm = state @ state.T.conj()
            return np.real(np.conj(self.state_0).T @ dm @ self.state_0)

    def compute_moments(self, psi):
        """Compute energy, energy squared, and variance for a given state."""
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
        Run the GCI loop either for a fixed number of steps or over predefined step sizes.
        Tracks energy, fidelity, and (if applicable) circuit depth.
        """
        if self.nmb_steps is None:
            for s in self.times_dbr:
                self.state_of_recursion = self.gci_step(self.state_of_recursion, s)
                self.results.append(self.compute_moments(self.state_of_recursion))
                self.fidelities.append(self.fidelity_with_0(self.state_of_recursion))
                self.states.append(self.state_of_recursion)
            self.s_mins = [0] + self.times_dbr
        else:
            for k in range(self.nmb_steps):
                test_es = [self.compute_moments(self.gci_step(self.state_of_recursion, dt))[0] for dt in self.times_dbr]
                ind = np.argmin(test_es)
                if ind == 0 or ind == len(self.times_dbr):
                    print(k, ind, self.times_dbr[ind], "--- Warning edge of interval reached")
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
        variances = [r[1] for r in self.results]

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
    Find the theta corresponding to a target fidelity within increasing tolerance.

    Args:
        thetas (list or np.ndarray): Step sizes.
        fidelity (list or np.ndarray): Fidelity values.
        target_fidelity (float): Desired fidelity.
        initial_tolerance (float): Starting tolerance.
        max_tolerance (float): Maximum allowed tolerance.
        step (float): Step size for increasing tolerance.

    Returns:
        float or None: Theta value if match found, else None.
    """
    tolerance = initial_tolerance
    while tolerance <= max_tolerance:
        for xi, yi in zip(thetas, fidelity):
            if abs(yi - target_fidelity) <= tolerance:
                return xi
        tolerance += step
    return None
