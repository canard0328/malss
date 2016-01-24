# -*- coding: utf-8 -*-

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

__author__ = 'Ryota KAMOSHIDA'
__version__ = '1.0.0'
__license__ = 'MIT License: http://www.opensource.org/licenses/mit-license.php'

__all__ = ['algorithm', 'data', 'malss']

# high level interface
import os
print(os.getcwd())
from malss import MALSS
