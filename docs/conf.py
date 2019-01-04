# -*- coding: utf-8 -*-


import sys
import os


sys.path.insert(0, os.path.abspath(".."))
import scinum as sn


project = "scinum"
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
add_module_names = True

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
    "travis_button": True,
    "fixed_sidebar": True
}

extensions = [
    "sphinx.ext.autodoc"
]
