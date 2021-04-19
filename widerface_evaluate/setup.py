"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""
import paddle
from distutils.core import setup
from distutils.core import Extension
from Cython.Build import cythonize
import numpy
package = Extension('bbox', ['box_overlaps.pyx'], include_dirs=[numpy.
    get_include()])
setup(ext_modules=cythonize([package]))
