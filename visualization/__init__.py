"""
visualization

All plotting code for the DBAC project lives here, separated from the
physics/computation in `functions`. Submodules:

- style: shared matplotlib figure-size and rcParams helpers.
- bloch_sphere: Bloch-sphere rendering of single-qubit states and trajectories.
- process_matrix: Pauli-transfer-matrix and process-matrix heatmaps.
- gci_plots: energy/fidelity vs. evolution-duration plots for `GCI.execute_gci`.
- dbac_plots: fidelity-fidelity and energy-vs-theta cooling-performance plots.
- dme_plots: DME channel implementation benchmark plots.
"""

from .style import *  # noqa: F401,F403
from .bloch_sphere import *  # noqa: F401,F403
from .process_matrix import *  # noqa: F401,F403
from .gci_plots import *  # noqa: F401,F403
from .dbac_plots import *  # noqa: F401,F403
from .dme_plots import *  # noqa: F401,F403
