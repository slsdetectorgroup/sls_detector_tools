
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()
sns.set()
sns.set_context('talk', font_scale = 1.2)

#sls_detector
import sls_detector_tools.config as cfg
from sls_detector_tools import calibration
from sls_detector_tools.plot import imshow, chip_histograms


#Configuration for the calibration script
cfg.geometry = '250k' #quad, 500k, 2M, 9M
cfg.calibration.type = 'XRF' #Sets function to fit etc.
cfg.det_id = 'EM3'
cfg.calibration.gain = 'gain3'
cfg.calibration.target = 'Cr'
#cfg.path.data = os.path.join('/mnt/local_sw_raid/eiger_data/trash',
#                             cfg.det_id, cfg.calibration.gain)

cfg.calibration.run_id = 0

#plot vcmp
with np.load(os.path.join(cfg.path.data, calibration.get_data_fname())) as f:
    data = f['data']
    x = f['x']

fit_result = np.load(os.path.join(cfg.path.data, calibration.get_fit_fname()))


calibration._plot_scurve(data, x)
mean, std, lines = chip_histograms( fit_result['mu'] )
plt.xlabel('Vcmp [DAC LSB]')
plt.ylabel('Number of Pixels')

#Load trimbit data
with np.load(os.path.join(cfg.path.data, calibration.get_tbdata_fname())) as f:
    tbdata = f['data']
    tbx = f['x']
    
calibration._plot_trimbit_scan(tbdata, tbx)

with np.load(os.path.join(cfg.path.data, calibration.get_trimbit_fname()+'.npz')) as f:
    tb = f['trimbits']
  
    
ax, im = imshow(tb)
ax.set_title('Trimbit map')
a = calibration._plot_trimbit_histogram(tb)

