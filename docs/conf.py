"""Sphinx configuration."""
project = "Fermi Contours"
author = "Pablo Piskunow"
copyright = "2022, Pablo Piskunow"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
