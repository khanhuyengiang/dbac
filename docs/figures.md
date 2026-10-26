# Reproducing the figures

Every figure of the paper is produced by a script in `scripts/`, named after the figure. Run a
script from the repository root and it writes its PDFs to `figures/`:

```bash
python scripts/figure_4.py
```

The notebooks in `notebooks/` walk through the same calculations with explanations. They only
save PDFs when `SAVE_PLOTS = True` in `notebooks/config.py` (off by default).

:::{tip}
The scripts use the paper style, which renders labels with LaTeX. Without a LaTeX installation,
see [Installation](installation.md#latex-for-paper-style-figures).
:::

## Main text

### Fig. 4: cooling on hardware

```bash
python scripts/figure_4.py
```

Energy of the data qubit after DBAC against the initialization angle, for the three hardware
circuits A ($k=1$, $M=1$), B ($k=1$, $M=2$) and C ($k=2$, $M=1$), with the measured data
overlaid. Writes `F4_Circuit_A_n_2.pdf`, `F4_Circuit_B_n_3.pdf` and `F4_Circuit_C_n_4.pdf`.

The [front page](index.ipynb) of this documentation is the narrated version of this script,
including the circuit diagrams.

## Supplementary material

### Fig. S2: accuracy of DME

```bash
python scripts/sup_figure_S2.py
```

Distance between Trotterized DME and the exact evolution as the number of Trotter steps $M$
grows, comparing the implementations in `dbac.dme`. Writes `S2_DME_norm.pdf`.
→ [Density matrix exponentiation](guide/dme.md)

### Fig. S4: DBAC with ideal DME

```bash
python scripts/sup_figure_S4.py
```

DBAC in the limit $M\to\infty$ (`dbac.dbqite.ideal_DME`), sweeping the recursion depth
$k = 1, \dots, 6$: final fidelity against initial fidelity, the hottest state that still reaches
a target fidelity, and the optimized step durations. Writes `S4_QITE_ideal_DME_merged.pdf`.
Notebook: `notebooks/sup_figure_S4.ipynb`. → [Running the recursion](guide/recursion.md)

This script also owns the table of optimized durations `notebooks/optimal_s_Mk.npy` used by
the other scripts (`load_or_fit_optimal_s_Mk`), and refits it if the file is missing.

### Fig. S5: the recursion on the Bloch sphere

```bash
python scripts/sup_figure_S5.py
```

The states of the DBAC recursion drawn on the Bloch sphere, for three initial states that are
close to, halfway from and far from $|0\rangle$. Writes `S5_bloch_good.pdf`, `S5_bloch_mid.pdf`
and `S5_bloch_bad.pdf`. The module also provides `compare_pure_vs_mixed`, which runs DBAC on
depolarized inputs.

### Fig. S6: choosing the step duration

```bash
python scripts/sup_figure_S6.py
```

Final fidelity after one DBAC step as a function of the initial angle and the step duration
$s$, for $M=1$ and $M=2$, with the optimal durations marked. Writes `S6_M1k1_s_sweep.pdf` and
`S6_M2k1_s_sweep.pdf`. → [One DBAC step](guide/dbac-step.md)

## Other notebooks

| notebook | what it shows |
|---|---|
| `notebooks/dme_ptm_plot.ipynb` | Pauli-transfer matrices of the $ZZ$ interaction, the DME unitary and hardware circuits A and B |
| `notebooks/hbac.ipynb` | heat-bath algorithmic cooling on three qubits, for pure and depolarized inputs → [HBAC baseline](guide/hbac.md) |
| `notebooks/draw_circuit.ipynb` | LaTeX (qcircuit) drawings of the DME, DBAC and HBAC circuits, made with qiskit |
