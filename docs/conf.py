
import os
import sys
import pyvista as pv

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sys.path.insert(0, os.path.abspath('..'))

pv.OFF_SCREEN = True
pv.set_plot_theme("document")
pv.BUILDING_GALLERY = True

project = 'finitewave'
copyright = '2024, Timur Nezlobinsky, Arstanbek Okenov'
author = 'Timur Nezlobinsky, Arstanbek Okenov'
release = '0.8.0'
version = '0.8.0'

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

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 2,
    "show_prev_next": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TiNezlobinsky/Finitewave/",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "header_links_before_dropdown": 7,
    "show_version_warning_banner": True,
}

copybutton_prompt_text = r">>> |\$ "  # Text to ignore in code blocks
copybutton_prompt_is_regexp = True  # Allows regex for more complex patterns

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

# autodoc options
autosummary_generate = True
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# disable showing the full path to modules
add_module_names = False

# intersphinx options
modindex_common_prefix = ["finitewave."]
