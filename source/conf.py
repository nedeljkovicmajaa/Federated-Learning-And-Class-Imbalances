# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Federated Learning and Class Imbalances'
copyright = '2025, Marija Nedeljkovic'
author = 'Marija Nedeljkovic'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically document Python docstrings
    'sphinx.ext.napoleon',  # Google-style docstrings
    'sphinx.ext.viewcode',  # Include links to the source code
    'nbsphinx',  # Allows including Jupyter Notebooks
]

templates_path = ['_templates']
exclude_patterns = []

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = ['custom.css']
