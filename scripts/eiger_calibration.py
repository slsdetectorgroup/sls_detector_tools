
# -*- coding: utf-8 -*-
"""
Script to calibrate an EIGER module using the big X-ray box.
"""
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
from sls_detector import Detector, Eiger
from sls_detector_tools import XrayBox, xrf_shutter_open, DummyBox
from sls_detector_tools.plot import imshow
from sls_detector_tools.io import write_trimbit_file
from sls_detector_tools import mask

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

#Configuration for the calibration script
cfg.geometry = '500k' #quad, 500k, 2M, 9M
cfg.calibration.type = 'XRF' #Sets function to fit etc.
cfg.det_id = 'T45'
cfg.calibration.gain = 'gain5'
cfg.calibration.target = 'Cu'
#cfg.path.data = os.path.join('/mnt/local_sw_raid/eiger_data/trash',
#                             cfg.det_id, cfg.calibration.gain)

cfg.calibration.run_id = 0

#Record the measurement in a log file
logger = logging.getLogger()
cfg.path.log = cfg.path.data
cfg.set_log('default_file.log', stream = False, level = logging.INFO)


#-------------------------------------------------------------Xray box control
box = XrayBox()  #XrayBox or DummyBox
#box.unlock()
#box.HV =  True
print(box.current)




#--------------------------------------------Setup for taking calibration data
#d = Eiger()
#calibration.setup_detector(d)
#
#
#vrf, t, cts = calibration.do_vrf_scan(d, box)
#d.dacs.vrf = vrf
#cfg.calibration.exptime = t
#
##
##
##data, x = calibration.do_scurve(d, box)
#fit_result = calibration.do_scurve_fit_scaled()
#np.save(os.path.join(cfg.path.data, calibration.get_fit_fname()), fit_result)
##out = calibration.find_mean_and_set_vcmp(d, fit_result)
#data, x = calibration.do_trimbit_scan(d, box)
#tb, target, data,x, result = calibration.find_and_write_trimbits_scaled()
#
#dacs = d.dacs.get_asarray()
#dacs = np.vstack((dacs, np.zeros(2)))
#
#os.chdir(cfg.path.data)
#host = d.hostname
#for i, hm in enumerate(mask.detector[cfg.geometry].halfmodule):
#    fn = '{}.sn{}'.format(calibration.get_trimbit_fname(),host[i][3:])
#    write_trimbit_file( fn, tb[hm], dacs[:,i] )
##
#calibration.load_trimbits(d)


#cfg.calibration.run_id = 1
#data, x = calibration.do_scurve(d, box)
#calibration.do_scurve_fit()
#data, x = calibration.take_global_calibration_data(d, box)
#calibration.per_chip_global_calibration()
#
#cfg.top = d.hostname[0]
#cfg.bottom = d.hostname[1]
#calibration.generate_calibration_report()

#with np.load(os.path.join(cfg.path.data, calibration.get_tbdata_fname())) as f:
#    data = f['data']
#    x = f['x']
#
#
#calibration._plot_trimbit_scan(data,x)
