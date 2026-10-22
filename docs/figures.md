# Reproducing the figures

Every script writes its PDFs to `figures/` at the repository root. Notebooks save
their plots only when `SAVE_PLOTS = True` in `notebooks/config.py` (off by default).

| Script | Output |
|---|---|
| `scripts/figure_4.py` | `F4_Circuit_A_n_2.pdf`, `F4_Circuit_B_n_3.pdf`, `F4_Circuit_C_n_4.pdf` (Fig. 4) |
| `scripts/sup_figure_S2.py` | `S2_DME_norm.pdf` |
| `scripts/sup_figure_S4.py` | `S4_QITE_ideal_DME_merged.pdf` |
| `scripts/sup_figure_S5.py` | `S5_bloch_good.pdf`, `S5_bloch_mid.pdf`, `S5_bloch_bad.pdf` |
| `scripts/sup_figure_S6.py` | `S6_M1k1_s_sweep.pdf`, `S6_M2k1_s_sweep.pdf` |
