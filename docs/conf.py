import os
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
    "myst_nb",
]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["qibo", "qutip"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3
html_theme = "furo"
html_title = "dbac"
# House style shared with khanhuyengiang.com; see the header of _static/academic.css.
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600"
    "&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap",
    "academic.css",
]
pygments_style = "lovelace"
pygments_dark_style = "dracula"
# Furo's default sidebar with a "back to khanhuyengiang.com" link above the brand.
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "sidebar/home-link.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}
html_theme_options = {
    "source_repository": "https://github.com/khanhuyengiang/dbac/",
    "source_branch": "main",
    "source_directory": "docs/",
}
add_module_names = False
html_baseurl = "https://dbac.khanhuyengiang.com/"
# The front page (index.ipynb) is committed with its outputs; guide pages opt in to running
# their examples at build time with `mystnb: execution_mode: force` in their front matter.
nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_output_stderr = "remove"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
# Keep qibo's "Using numpy backend" log line out of executed outputs (inherited by the kernels).
os.environ.setdefault("QIBO_LOG_LEVEL", "3")
