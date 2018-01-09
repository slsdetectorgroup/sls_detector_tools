#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:17:59 2017

@author: erikfrojdh
"""

from setuptools import setup, Extension, find_packages
#import sys
#import setuptools
import numpy.distutils.misc_util
__version__ = '0.0.1'


c_ext = Extension("_sls_cmodule",
                  sources = ["src/sls_cmodule.cpp", "src/fit_tgraph.cpp"], 
                  libraries = ['stdc++', 'Core', 'MathCore', 'Hist'],
                  library_dirs = ['/opt/root/5.XX/root/lib/'])
                  
c_ext.extra_compile_args = ['-std=c++11 `root-config --cflags --glibs`']

c_ext.language = 'c++'

setup(
    name='sls_detector_tools',
    version=__version__,
    author='Erik Frojdh',
    author_email='erik.frojdh@psi.ch',
    url='https://github.com/something/',
    description='Some test',
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    long_description='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    zip_safe=False,
)
