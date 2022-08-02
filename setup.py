# -*- coding: utf-8 -*-

import sys 
from distutils.version import LooseVersion


try:
    from setuptools import setup, find_packages
except ImportError:
    print('setuptools is required.')
    sys.exit()

if sys.version_info.major == 2:
    print('python >= 3.9 is required.')
    sys.exit()
elif sys.version_info.major == 3 and sys.version_info < (3, 9):
    print('python >= 3.9 is required.')
    sys.exit()

try:
    import numpy
    import scipy
except ImportError as inst:
    print(inst)
    sys.exit()
import sklearn
import matplotlib
import pandas
import jinja2

if LooseVersion(numpy.__version__) < LooseVersion('1.21.2'):
    raise ImportError('numpy >= 1.21.2 is required')

if LooseVersion(scipy.__version__) < LooseVersion('1.7.1'):
    raise ImportError('scipy >= 1.7.1 is required')

if LooseVersion(sklearn.__version__) < LooseVersion('1.1.1'):
    raise ImportError('sklearn >= 1.1.1 is required')

if LooseVersion(matplotlib.__version__) < LooseVersion('3.4.3'):
    raise ImportError('matplotlib >= 3.4.3 is required')

if LooseVersion(pandas.__version__) < LooseVersion('1.3.3'):
    raise ImportError('pandas >= 1.3.3 is required')

if LooseVersion(jinja2.__version__) < LooseVersion('3.1.2'):
    raise ImportError('jinja2 >= 3.1.2 is required')


PACKAGE = "malss"
NAME = "malss"
DESCRIPTION = "MALSS: MAchine Learning Support System"
AUTHOR = __import__(PACKAGE).__author__
AUTHOR_EMAIL = "malss@malss.com"
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
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    python_requires='>=3.9',
    #packages=["malss", "malss.app"],
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.1.1',
        'matplotlib>=3.4.3',
        'pandas>=1.3.3',
        'jinja2>=3.1.2'
        ],
    include_package_data=True,
    package_data={"malss": ["template/*.tmp",
                            "static/*.png",
                            "app/static/*.txt",
                            "app/static/*.png"]},
    classifiers=[
        "Development Status :: 4 - Beta",
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
