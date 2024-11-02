# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'finitewave'
copyright = '2024, Timur Nezlobinsky, Arstanbek Okenov'
author = 'Timur Nezlobinsky, Arstanbek Okenov'
release = '0.8.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    'sphinx_gallery.gen_gallery'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for sphinx_copybutton -------------------------------------------
copybutton_prompt_text = ">>> "  # Text to ignore in code blocks
copybutton_prompt_is_regexp = True  # Allows regex for more complex patterns

# -- Options for sphinx_gallery ----------------------------------------------
sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery
}
