# -*- coding: utf-8 -*-
"""
Script to calibrate an EIGER module using the big X-ray box



Erik Frojdh
"""

#import ROOT

#import sys

#Temporary paths for dev
#sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector')
#sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')

#python
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()
sns.set()
sns.set_context('talk', font_scale = 1.2)


#sls_detector
import sls_detector_tools.config as cfg
from sls_detector_tools import calibration
from sls_detector import Detector, Eiger
from sls_detector_tools import XrayBox, xrf_shutter_open
from sls_detector_tools.plot import imshow
import sls_detector_tools.mask as mask
from sls_detector_tools.io import write_trimbit_file



cfg.verbose = True
cfg.nmod = 36
cfg.geometry = '9M'
cfg.calibration.type = 'XRF'

#Configuration for the calibration script
cfg.det_id = 'Eiger9M'
cfg.calibration.gain = 'gain6'
cfg.calibration.target = 'Ge'
cfg.path.data = os.path.join( '/external_pool/eiger_data/2018/calibration', 
                             cfg.det_id, cfg.calibration.gain)


from sls_detector_tools.function import scurve

#data 
with np.load(os.path.join(cfg.path.data, calibration.get_data_fname())) as f:
    data0 = f['data']
    x0 = f['x']

fit_result0 = np.load(os.path.join(cfg.path.data, calibration.get_fit_fname()))

p = (2583,1248)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,9))
pixels =[(2583,1248), (1659,1145)]
for p in pixels:
    par = fit_result0[p]
    ax1.plot(x0, data0[p]/data0[p][-1]*1000, 'o')
    ax1.plot(x0, scurve(x0, *par))
    
    par1 =[result[p][i] for i in range(6)]
    ax2.plot(x, data[p], 'o')
    ax2.plot(x, scurve(x, *par1))