#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:17:59 2017

@author: erikfrojdh
"""

from setuptools import setup, Extension, find_packages
import sys
import setuptools

__version__ = '0.0.1'

setup(
    name='sls_detector_tools',
    version=__version__,
    author='Erik Frojdh',
    author_email='erik.frojdh@psi.ch',
    url='https://github.com/something/',
    description='Some test',
    long_description='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    zip_safe=False,
)