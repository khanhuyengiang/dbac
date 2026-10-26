import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dbac

project = "dbac"
author = "Khanh Uyen Giang et al."
release = dbac.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["qibo", "qutip"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
myst_enable_extensions = ["dollarmath", "amsmath"]
html_theme = "furo"
html_baseurl = "https://dbac.khanhuyengiang.com/"
exclude_patterns = ["_build"]
