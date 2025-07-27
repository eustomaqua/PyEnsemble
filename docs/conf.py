# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import solar_theme
# import sphinx_pdj_theme
# import sphinx_readable_theme


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyFairness'
copyright = '2025, eustomaqua'
author = 'eustomaqua'
release = '0.2.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'conestack'
# html_theme = 'python_docs_theme'
# html_theme = 'readable'
# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme = 'sphinx_pdj_theme'
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
# html_theme = 'karma_sphinx_theme'
# extensions.append("sphinx_wagtail_theme")
# html_theme = 'sphinx_wagtail_theme'
html_theme = 'sphinxdoc'    # 'nature'
# html_theme = 'solar_theme'
# html_theme_path = [solar_theme.theme_path]
# html_theme = 'sphinx_nefertiti'

# html_theme = "sphinx_rtd_theme"
# extensions = ['recommonmark', 'sphinx_markdown_tables']
# html_theme = 'alabaster'
html_static_path = ['_static']
