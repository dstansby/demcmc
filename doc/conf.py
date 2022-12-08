# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "demcmc"
copyright = "2022, David Stansby"
author = "David Stansby"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    # "sphinx_gallery.gen_gallery",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples/*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# -- Extension configuration
"""sphinx_gallery_conf = {
    "examples_dirs": "./examples",  # path to your example scripts
    "gallery_dirs": "./_auto_examples",  # path to where to save gallery generated output
    "download_all_examples": False,
}"""
numpydoc_show_class_members = False
automodapi_toctreedirnm = "_api"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

numpydoc_validation_checks = {
    "all",
    "SA01",  # Allow omitting See Also section
    "EX01",  # Allow omitting Examples section
    "ES01",  # Allow omitting extended summary section
}
