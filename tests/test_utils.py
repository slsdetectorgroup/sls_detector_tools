#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:32:10 2018

@author: l_frojdh
"""
import numpy as np
from sls_detector_tools import utils

def test_rebin_shape():
    image = np.zeros((256,256))
    out = utils.rebin_image(image, 2)
    assert out.shape == (128,128)
    
def test_rebin_values():
    image = np.ones((256,256))
    out = utils.rebin_image(image, 2)
    assert (out==4).sum()==(128*128)
    
def test_rebin_values3():
    image = np.ones((256,256))
    out = utils.rebin_image(image, 3)
    assert (out==9).sum()==(85*85)