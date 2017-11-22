import numpy as np
import os
import sys

#Temporary paths for dev
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector')
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')

from sls_detector import Detector
#from sls_detector_tools.io import load_frame, save_txt, load_txt, load_file
from sls_detector_tools import module_tests
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
#

cfg.nmod = 2
d.dacs.iodelay = 660

### RX bias test to find operation point
out = module_tests.rx_bias(d, clk = 'Full Speed', npulse = 10, plot = True)
#out = module_tests.rx_bias(d, clk = 'Half Speed', npulse = 10, plot = True)
######
#########Set rx bias and iodelay
#d.set_dac('0:rxb_lb', 1100)
#d.set_dac('0:rxb_rb', 1100)
#d.set_dac('1:rxb_lb', 1100)
#d.set_dac('1:rxb_rb', 1100)
#d.set_dac('iodelay', 660)
#############
##################
#tests.io_delay(name, d, clk = 0, plot = True)
#tests.io_delay(name, d, clk = 1, plot = True)
#####################
##
#tests.analog_pulses(name, d, plot = True)
#
######################
#tests.counter(name, d, clk = 0)
#tests.counter(name, d, clk = 1)
#####################
#data = tests.overflow(name, d)
########
#tests.generate_report(path)
