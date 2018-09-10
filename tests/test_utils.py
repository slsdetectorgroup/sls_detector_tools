#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:32:10 2018

@author: l_frojdh
"""
import pytest

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
    
    
    
def test_sum_array_small_array():
    arr = np.array((1,3,5,2,8,7,9))
    assert all(utils.sum_array(arr, 1) == arr)
    assert all(utils.sum_array(arr, 2) == np.array((4,7,15)))
    assert all(utils.sum_array(arr, 3) == np.array((9,17)))
    assert all(utils.sum_array(arr, 4) == np.array((11)))
    assert all(utils.sum_array(arr, 5) == np.array((19)))
    assert all(utils.sum_array(arr, 6) == np.array((26)))
    assert utils.sum_array(arr, 7) == arr.sum()
    
def test_sum_array_large_array():
    arr = np.linspace(0,500,1000)
    assert utils.sum_array(arr,2).sum() == pytest.approx( arr.sum(), 1e-5)