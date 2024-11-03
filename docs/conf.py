# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import pyvista

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sys.path.insert(0, os.path.abspath('..'))

pyvista.OFF_SCREEN = True
pyvista.set_plot_theme("document")
pyvista.BUILDING_GALLERY = True

project = 'finitewave'
copyright = '2024, Timur Nezlobinsky, Arstanbek Okenov'
author = 'Timur Nezlobinsky, Arstanbek Okenov'
release = '0.8.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',    # Adds links to highlighted source code
    'sphinx.ext.intersphinx',
    "sphinx_copybutton",
    'sphinx_gallery.gen_gallery',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 3,
    "show_prev_next": True,
    "icon_links": [
        {"name": "Home Page", "url": "https://networkx.org", "icon": "fas fa-home"},
        {
            "name": "GitHub",
            "url": "https://github.com/TiNezlobinsky/Finitewave/",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "header_links_before_dropdown": 3,
    "show_version_warning_banner": True,
}

# -- Options for sphinx_copybutton -------------------------------------------
copybutton_prompt_text = r">>> |\$ "  # Text to ignore in code blocks
copybutton_prompt_is_regexp = True  # Allows regex for more complex patterns

# -- Options for sphinx_gallery ----------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery
    'image_scrapers': ('matplotlib', 'pyvista'),
    "subsection_order": ExplicitOrder(
        [
            "../examples/2D",
            "../examples/3D",
        ]
    ),
    'within_subsection_order': FileNameSortKey,
}

# -- Options for autodoc -----------------------------------------------------
autosummary_generate = True

# -- Options for autodoc ----------------------------------------------------
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Disable showing the full path to modules
add_module_names = False

# -- Options for intersphinx -------------------------------------------------
modindex_common_prefix = ["finitewave"]
