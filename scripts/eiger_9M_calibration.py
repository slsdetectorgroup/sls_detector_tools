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
from sls_detector_tools import XrayBox, xrf_shutter_open, DummyBox
from sls_detector_tools.plot import imshow
import sls_detector_tools.mask as mask
from sls_detector_tools.io import write_trimbit_file

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
cfg.nmod = 36
cfg.geometry = '9M'
cfg.calibration.type = 'XRF'

#Configuration for the calibration script
cfg.det_id = 'Eiger9M'
cfg.calibration.gain = 'gain4'
cfg.calibration.target = 'Fe'
cfg.path.data = os.path.join( '/external_pool/eiger_data/2018/calibration/', 
                             cfg.det_id,'vtr2500', cfg.calibration.gain)


#Record the measurement in a log file
logger = logging.getLogger()
cfg.path.log = cfg.path.data
cfg.set_log('default_file.log', stream = False, level = logging.INFO)
cfg.calibration.run_id = 0

cfg.calibration.vtr = 2500

#-------------------------------------------------------------Xray box control
#box = XrayBox()
#box.unlock()
#box.HV =  True

b = XrayBox()
box = DummyBox()

#--------------------------------------------Setup for taking calibration data
d = Eiger()
#calibration.setup_detector(d)
#cfg.calibration.vrf_scan_exptime = 1
#vrf, t, cts = calibration.do_vrf_scan(d, box)
#d.dacs.vrf = vrf
##
#cfg.calibration.exptime = 200
##data, x = calibration.do_scurve(d, box)
##fit_result = calibration.do_scurve_fit_scaled()
##np.save(os.path.join(cfg.path.data, calibration.get_fit_fname()), fit_result)
##data, x = calibration.do_trimbit_scan(d, box)
######
#######-------------------------------------------------------------------TRIMBITS
##tb, target, data,x, result = calibration.find_and_write_trimbits_scaled()
##np.save(os.path.join(cfg.path.data, calibration.get_trimbit_fname()), tb)
#######
######
##dacs = d.dacs.get_asarray()
##dacs = np.vstack((dacs, np.zeros(36)))
#####def write_trimbit_file( fname, data, dacs, ndacs = 18 ):
#####
##os.chdir(cfg.path.data)
##host = d.hostname
##for i, hm in enumerate(mask.detector[cfg.geometry].halfmodule):
##    fn = '{}.sn{}'.format(calibration.get_trimbit_fname(),host[i][3:])
##    write_trimbit_file( fn, tb[hm], dacs[:,i] )
####
#calibration.load_trimbits(d)
#cfg.calibration.run_id = 1
#data, x = calibration.do_scurve(d, box)
#fit_result = calibration.do_scurve_fit_scaled()
#np.save(os.path.join(cfg.path.data, calibration.get_fit_fname()), fit_result)
#tb, target, data,x, result = calibration.find_and_write_trimbits_scaled()
#

