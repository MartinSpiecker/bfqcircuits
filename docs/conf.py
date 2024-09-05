import os
import sys

# Get the absolute path of the directory containing conf.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Append the project root directory to sys.path, relative to the source directory
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../src')))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Black Forest Quantum Circuits'
copyright = '2024, Martin Spiecker'
author = 'Martin Spiecker'
version = 'v0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", 'nbsphinx', 'sphinxcontrib.apidoc']

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  #pydata_sphinx_theme
html_static_path = []  # "_static

nbsphinx_execute = 'never'

apidoc_module_dir = '../src/bfqcircuits'
apidoc_output_dir = 'api'
apidoc_toc_file = False
apidoc_module_first = True
apidoc_separate_modules = False

autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "mpl_toolkits", "schemdraw"]
