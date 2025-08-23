# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'DEEPDATAMININGLEARNING'
copyright = '2025, Kaikai Liu'
author = 'Kaikai Liu'

release = '1.0'
version = '1.0.0'

# -- General configuration

import sys
import os

# Add the _extensions directory to the Python path
sys.path.insert(0, os.path.abspath('../_extensions'))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    "nbsphinx",
    "nbsphinx_link",
    'mcreference'  # Custom extension for handling mcreference tags
]

# Configure nbsphinx for fast processing without execution
nbsphinx_execute = 'never'  # Never execute notebooks, just render existing outputs
nbsphinx_allow_errors = True  # Allow notebooks with errors to be processed
nbsphinx_timeout = 30  # Shorter timeout
nbsphinx_codecell_lexer = 'none'  # Disable syntax highlighting for speed

# Exclude problematic notebooks that cause long processing times
exclude_patterns = [
    'notebooks/CMPE-Langchain.ipynb',
    'notebooks/CMPE-LangchainMac.ipynb', 
    'notebooks/CMPE-LangchainNVIDIA.ipynb',
    'notebooks/CMPE-LangchainNVIDIA2.ipynb',
    'notebooks/CMPE-Pinecone.ipynb'
]

# MyST parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "strikethrough",
    "substitution",
    "tasklist"
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'furo' # Modern, clean theme similar to Material Design
# html_theme = 'sphinx_immaterial' #'sphinx_rtd_theme' #'bizstyle' #'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
