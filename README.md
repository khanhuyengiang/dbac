# dme_sizzle_dbqite
This is the repository for the paper **Double-bracket quantum algorithm for qubit reset implemented with multicolor-driven superconducting qubits**

*Dynamic quantum algorithms are quantum algorithms whose operation on data registers depends on the quantum states of additional instruction registers. Density matrix exponentiation (DME) is a canonical example of a dynamic quantum algorithm. It uses multiple copies of an instruction state to apply unitary evolution to a data state, where the instruction density matrix acts as the Hamiltonian. Until now, DME has been experimentally demonstrated only for special cases of single-qubit quantum instructions and resorted to recycling of instruction qubits. In this work, we present an experimental demonstration of DME in a small-scale device but without recycling quantum instructions. Using a new fabrication of superconducting qubits with the newly proposed Stark induced ZZ by level excursions (siZZle) gate, we implement DME for a single data register with up to four instruction rounds involving neighboring qubits. Going beyond DME, we propose a new dynamic quantum algorithm for qubit reset, which uses DME as a subroutine in a recursive way. We advocate that dynamic quantum algorithms can be used as a benchmarking protocol for small-scale quantum chips.*

In this repo, there will be:
- DME
- DB-QITE
- circuit draw

View `jupyter_notebooks` for DME and DB-QITE notebooks. View `functions` for basic function and classes.