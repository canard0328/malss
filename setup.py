# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

PACKAGE = "malss"
NAME = "malss"
DESCRIPTION = "MALSS: MAchine Learning Support System for beginners"
AUTHOR = __import__(PACKAGE).__author__
AUTHOR_EMAIL = ""
URL = ""
VERSION = __import__(PACKAGE).__version__
LICENSE = __import__(PACKAGE).__license__

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read("README.rst"),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
    keywords='machine learning support system',
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib']
)
