# -*- coding: utf-8 -*-

import sys 
from distutils.version import LooseVersion


try:
    from setuptools import setup, find_packages
except ImportError:
    print('setuptools is required.')
    sys.exit()

if sys.version_info.major == 2:
    print('python >= 3.6 is required.')
    sys.exit()
elif sys.version_info.major == 3 and sys.version_info < (3, 6):
    print('python >= 3.6 is required.')
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

if LooseVersion(numpy.__version__) < LooseVersion('1.10.2'):
    raise ImportError('numpy >= 1.10.2 is required')

if LooseVersion(scipy.__version__) < LooseVersion('0.16.1'):
    raise ImportError('scipy >= 0.16.1 is required')

if LooseVersion(sklearn.__version__) < LooseVersion('0.20'):
    raise ImportError('sklearn >= 0.20 is required')

if LooseVersion(matplotlib.__version__) < LooseVersion('1.5.1'):
    raise ImportError('matplotlib >= 1.5.1 is required')

if LooseVersion(pandas.__version__) < LooseVersion('0.14.1'):
    raise ImportError('pandas >= 0.14.1 is required')

if LooseVersion(jinja2.__version__) < LooseVersion('2.8'):
    raise ImportError('jinja2 >= 2.8 is required')


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
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    python_requires='>=3.6',
    #packages=["malss", "malss.app"],
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=0.20',
        'matplotlib>=1.5.1',
        'pandas>=0.14.1',
        'jinja2>=2.8'
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
