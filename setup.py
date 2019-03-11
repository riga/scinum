# coding: utf-8


import os
from setuptools import setup

import scinum as sn


this_dir = os.path.dirname(os.path.abspath(__file__))


keywords = [
    "scientific", "numbers", "error", "systematics", "propagation",
]


classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
]


# read the readme file
with open(os.path.join(this_dir, "README.md"), "r") as f:
    long_description = f.read()


# load installation requirements
with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    install_requires = [line.strip() for line in f.readlines() if line.strip()]


setup(
    name=sn.__name__,
    version=sn.__version__,
    author=sn.__author__,
    author_email=sn.__email__,
    description=sn.__doc__.strip().split("\n")[0].strip(),
    license=sn.__license__,
    url=sn.__contact__,
    keywords=keywords,
    classifiers=classifiers,
    long_description=long_description,
    install_requires=install_requires,
    python_requires=">=2.7",
    zip_safe=False,
    py_modules=[sn.__name__],
)
