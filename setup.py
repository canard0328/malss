# -*- coding: utf-8 -*-

import sys
from distutils.version import LooseVersion

try:
    from setuptools import setup, find_packages
except ImportError:
    print 'setuptools is required.'
    sys.exit()

if sys.version_info < (2, 7):
    print 'python >= 2.7 is required.'
    sys.exit()

try:
    import numpy
    import scipy
    import sklearn
    import matplotlib
    import pandas
    import jinja2
except ImportError as inst:
    print inst
    sys.exit()

if LooseVersion(numpy.__version__) < LooseVersion('1.6.1'):
    print 'numpy >= 1.6.1 is required'
    sys.exit()

if LooseVersion(scipy.__version__) < LooseVersion('0.9'):
    print 'scipy >= 0.9 is required'
    sys.exit()

if LooseVersion(sklearn.__version__) < LooseVersion('0.14'):
    print 'sklearn >= 0.14 is required'
    sys.exit()

if LooseVersion(matplotlib.__version__) < LooseVersion('1.3'):
    print 'matplotlib >= 1.3 is required'
    sys.exit()

if LooseVersion(pandas.__version__) < LooseVersion('0.13'):
    print 'pandas >= 0.13 is required'
    sys.exit()

if LooseVersion(jinja2.__version__) < LooseVersion('2.7'):
    print 'jinja2 >= 2.7 is required'
    sys.exit()


PACKAGE = "malss"
NAME = "malss"
DESCRIPTION = "MALSS: MAchine Learning Support System"
AUTHOR = __import__(PACKAGE).__author__
AUTHOR_EMAIL = "https://github.com/canard0328/malss/"
URL = "https://github.com/canard0328/malss/"
VERSION = __import__(PACKAGE).__version__
LICENSE = __import__(PACKAGE).__license__

with open('README.rst') as file:
    long_description = file.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    packages=["malss"],
    include_package_data=True,
    package_data={"malss": ["template/report.html.tmp"]},
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
    keywords='machine learning support system'
)
