# -*- coding: utf-8 -*-


import sys
import os
import shlex


sys.path.insert(0, os.path.abspath(".."))
import scinum as sn


project = "scinum"
author = sn.__author__
copyright = sn.__copyright__
version = sn.__version__
release = sn.__version__


templates_path = ["_templates"]
html_static_path = ["_static"]
master_doc = "index"
source_suffix = ".rst"


exclude_patterns = []
pygments_style = "sphinx"
html_logo = "../logo.png"
html_theme = "alabaster"
html_sidebars = {"**": [
    "about.html",
    "localtoc.html",
    "searchbox.html"]
}
html_theme_options = {
    "github_user": "riga",
    "github_repo": "scinum",
    "travis_button": True
}


extensions = [
    "sphinx.ext.autodoc"
]
