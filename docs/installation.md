# Installation

```bash
git clone https://github.com/khanhuyengiang/dbac && cd dbac
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # package only
pip install -e ".[notebooks]"    # + Jupyter
pip install -e ".[docs]"         # + Sphinx, to build these docs (`make -C docs apidoc html`)
```

Figures with LaTeX labels need a local LaTeX install, or call
`dbac.plotting.use_paper_style(use_tex=False)`.
