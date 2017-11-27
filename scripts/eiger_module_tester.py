import numpy as np
import os
import sys

#Temporary paths for dev
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector')
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')

from sls_detector import Detector
#from sls_detector_tools.io import load_frame, save_txt, load_txt, load_file
from sls_detector_tools import eiger_tests
#import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

#import time 
import seaborn as sns
#from sls_detector_tools.plot import *
sns.set_context('talk')

import sls_detector_tools.config as cfg

#Initialize the system
#d = SlsDetector(cfg = 'pc1875_tests.cfg')
cfg.verbose = False
cfg.debug = False
d = Detector()


##Setup
cfg.det_id = 'T06-retest'
path = os.path.join( cfg.path.test, cfg.det_id )


cfg.geometry= '500k'

try:
	os.mkdir( path )
except OSError:
	pass

print("\n --- EIGER Module testing ---")
print("Test path:", path)
print("Tmp data path:", d.file_path)
print("module:", cfg.det_id)


cfg.nmod = 2
d.dacs.iodelay = 660
d.dacs.rxb_lb = 1100
d.dacs.rxb_rb = 1100

### RX bias test to find operation point
out = eiger_tests.rx_bias(d, clk = 'Full Speed', npulse = 10)
#out = eiger_tests.rx_bias(d, clk = 'Half Speed', npulse = 10)
#
###################
#out = eiger_tests.io_delay(d, clk = 'Full Speed', plot = True)
#out = eiger_tests.io_delay(d, clk = 'Half Speed', plot = True)
#####################
##
#a = eiger_tests.analog_pulses(d)
#
######################
#a = eiger_tests.counter(d, clk = 'Half Speed')
#tests.counter(name, d, clk = 1)
#####################
#data = tests.overflow(name, d)
########
#tests.generate_report(path)

#plt.ioff()
#
#def plot_lines(x, lines):
#    fig = plt.figure()
#    ax = plt.subplot(1,1,1)
#    for i in range(8):
#        ax.plot(x, lines[i], 'o-')
#    return fig, ax
