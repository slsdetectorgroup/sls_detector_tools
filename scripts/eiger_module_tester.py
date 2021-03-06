# -*- coding: utf-8 -*-

#Python imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()
sns.set_context('paper')

#Temporary paths for dev
#sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector')
#sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')

from sls_detector import Eiger
from sls_detector_tools import eiger_tests
import sls_detector_tools.config as cfg


cfg.verbose = False
cfg.debug = False
d = Eiger()


##Setup
cfg.det_id = 'T107'
cfg.geometry= '500k'
path = os.path.join( cfg.path.test, cfg.det_id )


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
out = eiger_tests.rx_bias(d, clk = 'Half Speed', npulse = 10)
####
######################
out = eiger_tests.io_delay(d, clk = 'Full Speed')
out = eiger_tests.io_delay(d, clk = 'Half Speed')
#########################
######
a = eiger_tests.analog_pulses(d)
###
########################
a = eiger_tests.counter(d, clk = 'Half Speed')
a = eiger_tests.counter(d, clk = 'Full Speed')
##tests.counter(name, d, clk = 1)
########################
data = eiger_tests.overflow(d)
#########
a = eiger_tests.generate_report(path)
#
#plt.ioff()
#
#def plot_lines(x, lines):
#    fig = plt.figure()
#    ax = plt.subplot(1,1,1)
#    for i in range(8):
#        ax.plot(x, lines[i], 'o-')
#    return fig, ax
