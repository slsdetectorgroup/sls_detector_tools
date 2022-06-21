#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:06:55 2018

@author: l_frojdh
"""

import time
from slsdet import Eiger
from sls_detector_tools import ZmqReceiver
from sls_detector_tools.plot import imshow
import sls_detector_tools.config as cfg
import matplotlib.pyplot as plt
plt.ion()

cfg.geometry = '1M'

d = Eiger()
d.rx_zmqstream = False
time.sleep(0.1)
d.rx_zmqstream = True
# d.file_write = False



receiver =  ZmqReceiver(d)

# d.acquire()
image = receiver.get_frame()
ax, im = imshow(image)
im.set_clim(0,2000)