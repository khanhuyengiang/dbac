
from qibo import hamiltonians, Circuit
from qibo.symbols import Z
import numpy as np
import matplotlib.pyplot as plt


class GCI:
    def __init__(self, gci_step, state_of_recursion, nmb_steps, times_dbr, please_be_verbose=True):
        self.gci_step = gci_step
        self.state_of_recursion = state_of_recursion
        self.psi0 = state_of_recursion
        self.nmb_steps = nmb_steps
        self.times_dbr = times_dbr
        self.please_be_verbose = please_be_verbose
        self.states = [state_of_recursion]
        
        # Fixed parts
        self.L = 1
        self.H_sh = hamiltonians.SymbolicHamiltonian(Z(0), nqubits=1)
        self.H = np.array([[-1.-0.j, -0.-0.j],
                            [-0.-0.j,  1.-0.j]])
        self.eigs = np.array([-1.,  1.])
        self.target_energy = -1

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
        shape = np.shape(state)
        if shape[0] == shape[1]:
            return np.real(np.conj(self.state_0).T @ state @ self.state_0)
        else:
            dm = state @ state.T.conj()
            return np.real(np.conj(self.state_0).T @ dm @ self.state_0)
    
    def compute_moments(self, psi):
        if isinstance(psi, Circuit):
            return self.H_sh.expectation(psi(self.psi0.T[0]).state()), None, None
        shape = np.shape(psi)
        if shape[0] == shape[1]:
            E = np.trace(psi @ self.H)
            S = np.trace(psi @ self.H @ self.H)
        else:
            E = (psi.conj().T @ self.H @ psi)[0, 0].real
            S = (psi.conj().T @ self.H @ self.H @ psi)[0, 0].real
        return E, S, S - E**2
    
    def execute_gci(self):
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
        if not hasattr(self.gci_step, 'exponent'):
            self.gci_step.exponent = 2
        if not hasattr(self.gci_step, 'label'):
            self.gci_step.label = ''
        if not hasattr(self.gci_step, 'marker'):
            self.gci_step.marker = '-'
        
        return (
            energies, self.s_mins, variances,
            [sum(s**self.gci_step.exponent for s in self.s_mins[1:x]) for x in range(1, len(self.s_mins) + 1)]
            if self.gci_step.exponent > 0 else self.s_mins,
            self.gci_step.label, self.gci_step.marker,
            self.circuit_depths if isinstance(self.state_of_recursion, Circuit) else None,
            self.circuits if isinstance(self.state_of_recursion, Circuit) else None,
            self.states,
            self.state_of_recursion,
            self.fidelities
        )
    
    def plot_GCI_results(self, *results, save_path=None, mode='Default'):
        if mode == 'Default':
            for result in results:
                if result[5] == '-':
                    plt.plot(result[3], result[0], label=result[4], linestyle=result[5])        
                else:
                    plt.plot(result[3], result[0], label=result[4], marker=result[5])
            plt.ylabel('Energy')
            plt.axline((0, self.target_energy), (results[0][3][-1], self.target_energy), linestyle=":")

        elif mode == 'Fidelity':
            for result in results:
                if result[5] == '-':
                    plt.plot(result[3], result[-1], label=result[4], linestyle=result[5])        
                else:
                    plt.plot(result[3], result[-1], label=result[4], marker=result[5])
            plt.ylabel('Fidelity')
            plt.axline((0, 1), (results[0][3][-1], 1), linestyle=":")

        plt.xlabel('Evolution duration')
        plt.grid()
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()