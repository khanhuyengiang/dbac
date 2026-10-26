"""
style.py

Shared matplotlib figure-size and rcParams helpers, so every script produces
paper-ready figures with a consistent look instead of redefining the same
rcParams block in each notebook.
"""

__all__ = ["JOURNAL_WIDTHS", "figure_size", "use_paper_style"]

import matplotlib.pyplot as plt

_PT = 1.0 / 72.27  # points -> inches (standard LaTeX point)

#: Column widths in inches, ``JOURNAL_WIDTHS[journal][column]``.
JOURNAL_WIDTHS = {
    "PRD": {"onecol": 246.0 * _PT, "twocol": 510.0 * _PT},
    "CQG": {"onecol": 374.0 * _PT},  # CQG is single column
}


def figure_size(journal="PRD", column="onecol", aspect=5.5 / 3.5):
    """Return (width, height) in inches for a figure sized to fit `journal`."""
    width = JOURNAL_WIDTHS[journal][column]
    return width, width / aspect


def use_paper_style(use_tex=True, font_size=10, figsize=None):
    """Apply the rcParams used for the figures in the DBAC paper."""
    params = {
        "font.size": font_size,
        "savefig.dpi": 75,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 0.3,
        "legend.handletextpad": 0.1,
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": "Times",
        "figure.autolayout": True,
    }
    if figsize is not None:
        params["figure.figsize"] = list(figsize)
    plt.rcParams.update(params)
