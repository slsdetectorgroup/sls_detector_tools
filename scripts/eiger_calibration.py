
# -*- coding: utf-8 -*-
"""
Script to calibrate an EIGER module using the big X-ray box.
"""
import ROOT
import sys


import os
import logging
import numpy as np

#sls_detector
# sys.path.append('/home/l_msdetect/erik/sls_detector_tools')
import sls_detector_tools.config as cfg
from sls_detector_tools import calibration
from sls_detector_tools.plot import imshow
from sls_detector_tools.io import write_trimbit_file
from sls_detector_tools import mask

from slsdet import Eiger
from slsdetbox import BigXrayBox

import matplotlib.pyplot as plt
import seaborn as sns
#plt.ion()
sns.set()
sns.set_context('talk', font_scale = 1.2)

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
cfg.geometry = '9M' #250k, 500k, 2M, 9M
cfg.calibration.type = 'XRF' #Sets function to fit etc.
cfg.det_id = '9M'
cfg.calibration.gain = 'gain2'
cfg.calibration.target = 'Ti'
cfg.calibration.energy = 4.5
cfg.path.data = os.path.join('/mnt/ssd/calibration/',
                             cfg.det_id, cfg.calibration.gain)

cfg.calibration.run_id = 0

#Record the measurement in a log file
logger = logging.getLogger()
cfg.path.log = cfg.path.data
cfg.set_log('default_file.log', stream = False, level = logging.INFO)


# #-------------------------------------------------------------Xray box control
box = BigXrayBox()
cfg.calibration.threshold = 1200
cfg.calibration.vrf_scan_exptime = 0.1
cfg.calibration.vtr = 2400

# #--------------------------------------------Setup for taking calibration data
d = Eiger()
calibration.setup_detector(d)
d.parallel = False


d.dacs.vtrim = cfg.calibration.vtr
d.vthreshold = cfg.calibration.threshold

vrpreamp, t, cts = calibration.do_vrf_scan(d, box, start = 2700, stop = 3800)
#vrpreamp = calibration.load_vrpreamp()
d.dacs.vrpreamp = vrpreamp
# cfg.calibration.exptime = t
cfg.calibration.exptime = 300

logger.info(f'vpreamp: {d.dacs.vrpreamp}')
logger.info(f'exptime: {cfg.calibration.exptime }s')
logger.info(f'vtrim: {d.dacs.vtrim}')

# # # # calibration.load_trimbits(d)

data, x = calibration.do_scurve(d, box)
fit_result = calibration.do_scurve_fit_scaled()
# fit_result = calibration.load_fit_result()

data, x = calibration.do_trimbit_scan(d, box)
tb, target, data,x, result = calibration.find_and_write_trimbits_scaled(d)
calibration.load_trimbits(d)


#cfg.calibration.run_id = 1
#data, x = calibration.do_scurve(d, box)
#fit_result = calibration.do_scurve_fit_scaled()


