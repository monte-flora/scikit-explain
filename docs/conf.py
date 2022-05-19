# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from os.path import dirname
import os
import sys
import shutil 
import sphinx_rtd_theme

#current_dir = os.getcwd()
#path = dirname(dirname(current_dir))
#sys.path.append(path)
#sys.path.append(os.path.join(path, 'skexplain'))
#sys.path.insert(0, os.path.abspath('../../'))

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('sphinxext'))
sys.path.insert(0, os.path.abspath('../tutorial_notebooks'))

# make copy of notebooks in docs folder, as they must be here for sphinx to
# pick them up properly.
NOTEBOOKS_DIR = os.path.abspath('notebooks')

if os.path.exists(NOTEBOOKS_DIR):
    import warnings
    warnings.warn('example_notebooks directory exists, replacing...')
    shutil.rmtree(NOTEBOOKS_DIR)
    
shutil.copytree(os.path.abspath('../tutorial_notebooks'), NOTEBOOKS_DIR)

# -- Project information -----------------------------------------------------

project = 'Scikit-Explain'
copyright = '2021, Montgomery Flora; Shawn Handler'
author = 'Montgomery Flora; Shawn Handler'

# The full version, including alpha/beta/rc tags
release = 'v0.0.7'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon',
              "sphinx.ext.intersphinx",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              'sphinx_rtd_theme',
              "nbsphinx",
             ]

# Taken from the SHAP documentation. 
autodoc_default_options = {
    'members': True,
    'inherited-members': True
}

autosummary_generate = True
numpydoc_show_class_members = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = 'latest'
# The full version, including alpha/beta/rc tags.
release = 'latest'


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

html_tile=project

html_theme_options = {
    #'canonical_url': '',
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#343131',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}



# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

#if not on_rtd:  # only import and set the theme if we're building docs locally
#   import stanford_theme
#    html_theme = 'stanford_theme'
#    html_theme_path = [stanford_theme.get_html_theme_path()]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']