"""Shared settings for the notebooks."""

from pathlib import Path

# Set to True to also save every plot as a PDF into `figures/`.
SAVE_PLOTS = False

PLOTS_DIR = Path(__file__).resolve().parents[1] / "figures"


def plot_path(name):
    """Return the PDF path for figure `name` if SAVE_PLOTS is on, else None
    (which the plotting functions treat as "don't save")."""
    if not SAVE_PLOTS:
        return None
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR / name
