#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 18:01:56 2019

@author: l_frojdh
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sls_detector_tools.plot import imshow
from sls_detector_tools.io import load_frame, load_file, read_frame_header, read_header

os.chdir('/home/l_frojdh/out')

det_id = 'T101'
bias = 150

image = load_frame(f'{det_id}_CuXRF_{bias}V',0)

#T107_CuXRF_20V_d0_0.raw

#a = image[:,512:528].copy()
#image[:,512:528] = image[:,0:16]
##image[:,0:16] = 0
ax, im = imshow(image)
im.set_clim(0,1000)
ax.set_title(f'{det_id} CuXRF {bias}V')
plt.savefig(f'{det_id}_CuXRF_{bias}V')