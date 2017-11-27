# -*- coding: utf-8 -*-
"""
Script to calibrate an EIGER module using the big X-ray box



Erik Frojdh
"""

import ROOT

import sys

#Temporary paths for dev
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector')
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')

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
from sls_detector import Detector
from sls_detector_tools import XrayBox


#Current Eiger calibration plan
"""
Name      vcmp       target
gain0     1500       In 3.3keV
gain2     1200       Ti 4.5keV
gain3     1200       Cr 5.4keV
gain4     1200       Fe 6.4keV
gain5     1200       Cu 8.0keV
gain6     1200       Ge 9.9keV
gain7     1200       Zr
gain8                Mo
gain9                Ag
gain10               Sn

"""

cfg.verbose = True
cfg.nmod = 2
cfg.geometry = '500k'
cfg.calibration.type = 'XRF'

#Configuration for the calibration script
cfg.det_id = 'T45-API'
cfg.calibration.gain = 'gain5'
cfg.calibration.target = 'Cu'
cfg.path.data = os.path.join( '/mnt/disk1/calibration/', 
                             cfg.det_id, cfg.calibration.gain)


#Record the measurement in a log file
logger = logging.getLogger()
cfg.path.log = cfg.path.data
cfg.set_log('default_file.log', stream = False, level = logging.INFO)


#-------------------------------------------------------------Xray box control
#box = DummyBox()
box = XrayBox()
box.unlock()
box.HV( True )

#--------------------------------------------Setup for taking calibration data
d = Detector()
#calibration.setup_detector(d)
##
##data, x = calibration._vrf_scan(d)
#vrf = calibration.do_vrf_scan(d, box)
#cfg.calibration.threshold = 1200
#vrf = calibration.do_vrf_scan(d, box, start = 2500, stop = 3950, step = 30)
#d.set_dac('0:vrf', vrf[0]) 
#d.set_dac('1:vrf', vrf[1])
#
#
##---------------------------------Initial threshold disperion and trimbit scan
#thrange = (0,2000)
#cfg.calibration.exptime = 10
#data, x = calibration.do_scurve(d, box, step = 40, thrange = thrange)
##cfg.calibration.plot = False
#fit_result = calibration.do_scurve_fit( thrange = thrange )
#calibration.do_trimbit_scan(d, box, step = 2)
#calibration.find_and_write_trimbits( None, tau = 200 )
##
##
###---------------------------------------------Load trimbits and verify trimming
#calibration.load_trim(d)
#cfg.calibration.run_id = 1
#data, x = calibration.do_scurve(d, box, step = 40,thrange = thrange)
#fit_result = calibration.do_scurve_fit( thrange = thrange)
#calibration.rewrite_calibration_files(d, tau = 200)
###
####Generate a mask with pixels too far from center
###a = calibration.generate_mask()
###np.save( os.path.join(cfg.path.data, 'mask'), a )
#
#
##--------------------------------------------Global scurve for vcmp calibration
##calibration.take_global_calibration_data(d, box, thrange = thrange)
#calibration.per_chip_global_calibration()
#cfg.top, cfg.bottom = d.get_hostname()
#data = calibration.generate_calibration_report(thrange = thrange)
