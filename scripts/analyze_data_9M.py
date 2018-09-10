# -*- coding: utf-8 -*-
"""
Script to calibrate an EIGER module using the big X-ray box



Erik Frojdh
"""
import seaborn as sns
import ROOT

import sys

##Temporary paths for dev
#sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector')
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')

#python
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
sns.set()
sns.set_context('talk', font_scale = 1.2)

#
##sls_detector
import sls_detector_tools.config as cfg
from sls_detector_tools import calibration
##from sls_detector import Detector, Eiger
##from sls_detector_tools import XrayBox, xrf_shutter_open
#from sls_detector_tools.plot import imshow
#
##Current Eiger calibration plan
#"""
#Name      vcmp       target
#gain0     1500       In 3.3keV
#gain2     1200       Ti 4.5keV
#gain3     1200       Cr 5.4keV
#gain4     1200       Fe 6.4keV
#gain5     1200       Cu 8.0keV
#gain6     1200       Ge 9.9keV
#gain7     1200       Zr
#gain8                Mo
#gain9                Ag
#gain10               Sn
#
#"""
#
#cfg.verbose = True
#cfg.nmod = 2
#cfg.geometry = '9M'
#cfg.calibration.type = 'XRF'
#
##Configuration for the calibration script
cfg.det_id = 'Eiger9M'
cfg.calibration.gain = 'gain5'
cfg.calibration.target = 'Cu'
cfg.path.data = os.path.join( '/mnt/disk1/calibration', 
                             cfg.det_id, cfg.calibration.gain)

os.chdir(cfg.path.data)

with np.load( calibration.get_vrf_fname()) as f:
    data = f['data']
    x = f['x']
cfg.nmod = 36
cfg.geometry = '9M'
vrf = calibration._fit_and_plot_vrf_data(data, x, cfg.Eiger9M.hostname )

#Write vrf to text file    
with open('vrf.txt', 'w') as f:
    for i, v in enumerate(vrf): 
        f.write( './sls_detector_put {:d}:vrf {:d}\n'.format(i, v) )
