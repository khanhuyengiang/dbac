# Installation

`dbac` is a pure-Python package. It needs Python 3.10 or newer and installs its dependencies
(numpy, scipy, matplotlib, qibo, qutip) automatically.

## Install from source

Clone the repository and install it into a virtual environment:

```bash
git clone https://github.com/khanhuyengiang/dbac
cd dbac
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The `-e` (editable) flag means changes to the source take effect without reinstalling.

## Optional extras

Extras add the tools for the notebooks, the documentation, and the tests:

| command | adds | needed for |
|---|---|---|
| `pip install -e ".[notebooks]"` | Jupyter | running the notebooks in `notebooks/` |
| `pip install -e ".[docs]"` | Sphinx, Furo, MyST-NB | building this documentation |
| `pip install -e ".[test]"` | pytest | running the test suite |

Combine them with commas, for example `pip install -e ".[notebooks,test]"`.

## Check that it works

```bash
python -c "import dbac; print(dbac.__version__)"
pytest                                   # with the [test] extra; also runs the docstring examples
```

## LaTeX for paper-style figures

The scripts and notebooks render figure labels with LaTeX, as in the paper. This needs a LaTeX
installation (for example `texlive` on Linux or MacTeX on macOS). Without one, switch it off
before plotting:

```python
from dbac.plotting import use_paper_style

use_paper_style(use_tex=False)
```

## Building the documentation

With the `[docs]` extra installed:

```bash
make -C docs html
```

The site is written to `docs/_build/html`; open `index.html` in a browser. Guide pages run
their examples while building, so a broken example fails the build. The front page is the
notebook `docs/index.ipynb` and is shown with its saved outputs: re-run it in Jupyter after
changing the code it uses.
