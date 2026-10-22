"""dbac: physics and numerics for double-bracket algorithmic cooling (plotting lives in `dbac.plotting`)."""

import warnings

import numpy as np

__version__ = "0.1.0"

# Silence noisy-but-harmless warnings: complex values being cast to real when
# reading off energies/fidelities, and GCI's "edge of interval" step-size notice.
warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)
warnings.filterwarnings("ignore", message=".*edge of interval reached.*")
