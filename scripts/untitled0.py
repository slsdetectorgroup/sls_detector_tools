#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:06:55 2018

@author: l_frojdh
"""

import time
from sls_detector import Eiger
from sls_detector_tools import ZmqReceiver
from sls_detector_tools.plot import imshow

d = Eiger()
d.rx_datastream = False
time.sleep(0.1)
d.rx_datastream = True
d.file_write = False



receiver =  ZmqReceiver(d)

d.acq()
image = receiver.get_frame()
ax, im = imshow(image)