#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sls_detector_tools, python and c++...
"""
import os
import subprocess
from setuptools import setup, Extension, find_packages
import numpy.distutils.misc_util as du
import cppyy
__version__ = '0.0.1'
out = subprocess.run(['root-config', '--cflags', '--glibs', '--ldflags'], 
                     stdout = subprocess.PIPE)
args = out.stdout.decode().strip('\n').split()

c_ext = Extension("_sls_cmodule",
                  sources = ["src/sls_cmodule.cpp", "src/fit_tgraph.cpp"],
                  libraries = ['Core', 'MathCore', 'Hist','cppyy3_10'],
                  library_dirs = [os.path.join(os.environ['ROOTSYS'])],
                  include_dirs=[*du.get_numpy_include_dirs(),],
                  extra_compile_args = args #+ ['-fsanitize=address'],
                #   extra_link_args = ['-fsanitize=address'],
                  )
                  

setup(
    name='sls_detector_tools',
    version=__version__,
    author='Erik Frojdh',
    author_email='erik.frojdh@psi.ch',
    url='https://github.com/something/',
    description='Some test',
    ext_modules=[c_ext],
    long_description='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    zip_safe=False,
)
