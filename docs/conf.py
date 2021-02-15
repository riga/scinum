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
html_logo = "../logo.png"
html_sidebars = {"**": [
    "about.html",
    "localtoc.html",
    "searchbox.html",
]}
html_theme = "alabaster"
html_theme_options = {
    "github_user": "riga",
    "github_repo": "scinum",
    "travis_button": True,
    "fixed_sidebar": True,
}

extensions = ["sphinx.ext.autodoc", "pydomain_patch"]

autodoc_member_order = "bysource"


def setup(app):
    app.add_css_file("styles.css")
