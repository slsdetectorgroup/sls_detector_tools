#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:42:50 2018

@author: l_frojdh
"""

import numpy as np


def load_file(fname, n_frames):
    f = open(fname, 'rb')
    image = np.zeros((512,1024,n_frames))
    for i in range(n_frames):
        f.read(48) #frame header TODO! inspect
        image[:,:,i] = np.fromfile(f, dtype = np.uint16, count = 1024*512).reshape((512,1024))
        
    return image