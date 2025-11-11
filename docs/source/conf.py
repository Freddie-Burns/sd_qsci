# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
# Make sd-qsci package importable by Sphinx (src/ layout)
sys.path.insert(0, os.path.abspath('../../src'))

project = 'sd-qsci'
copyright = '2025, Freddie Burns'
author = 'Freddie Burns'

version = '0.1'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",     # parses NumPy style
    "sphinx.ext.viewcode",
    "myst_parser",             # allow Markdown in docs
]

templates_path = ['_templates']
exclude_patterns = []

# Mock imports for packages not available in the build environment.
# Uncomment these if building docs without installing heavy dependencies:
# autodoc_mock_imports = [
#     "numpy",
#     "pyscf",
#     "scipy",
#     "ffsim",
#     "qiskit",
#     "qiskit_aer",
# ]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Autodoc / typing / autosummary
autodoc_typehints = "description"  # put type hints into the doc body
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True

# Napoleon config for NumPy style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Theme
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "style_external_links": True,
}
html_baseurl = "https://freddie-burns.github.io/sd_qsci/"
