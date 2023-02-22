# coding: utf-8

import sys
import os


thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, "_extensions"))
sys.path.insert(0, os.path.dirname(thisdir))

import scinum as sn


project = sn.__name__
author = sn.__author__
copyright = sn.__copyright__
copyright = copyright[10:] if copyright.startswith("Copyright ") else copyright
copyright = copyright.split(",", 1)[0]
version = sn.__version__[:sn.__version__.index(".", 2)]
release = sn.__version__
language = "en"

templates_path = ["_templates"]
html_static_path = ["_static"]
master_doc = "index"
source_suffix = ".rst"
exclude_patterns = []
pygments_style = "sphinx"
add_module_names = False

html_title = "{} v{}".format(project, version)
html_logo = "../assets/logo.png"
html_favicon = "../assets/favicon.ico"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "logo_only": True,
    "show_navbar_depth": 2,
    "repository_url": "https://github.com/riga/scinum",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "autodocsumm",
    "myst_parser",
    "sphinx_lfs_content",
    "pydomain_patch",
]

autodoc_member_order = "bysource"


def setup(app):
    app.add_css_file("styles.css")
