"""
dbac.plotting

All plotting code for the DBAC project lives here, separated from the
physics/computation in `dbac`. Submodules:

- style: shared matplotlib figure-size and rcParams helpers.
- bloch: Bloch-sphere rendering of single-qubit states and trajectories.
- ptm: Pauli-transfer-matrix and process-matrix heatmaps.
- gci: energy/fidelity vs. evolution-duration plots for `GCI.execute_gci`.
- cooling: fidelity-fidelity and energy-vs-theta cooling-performance plots.
- dme: DME channel implementation benchmark plots.
"""

from .style import *  # noqa: F401,F403
from .bloch import *  # noqa: F401,F403
from .ptm import *  # noqa: F401,F403
from .gci import *  # noqa: F401,F403
from .cooling import *  # noqa: F401,F403
from .dme import *  # noqa: F401,F403
