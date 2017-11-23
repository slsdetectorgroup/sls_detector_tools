#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:40:47 2017

@author: l_frojdh
"""
import os
import sys
from itertools import chain
sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')
from sls_detector_tools.io import load_txt
import numpy as np
import matplotlib.pyplot as plt
from sls_detector_tools import config as cfg
os.chdir('/home/l_frojdh/code/scripts/testing')
threshold = 30000

low = []
high = []
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('_rxbias_1.txt'):
            fp = os.path.join(root, f)
#            print(fp)
            out = load_txt(fp)
            for values in out[1:]:
                it= enumerate(values)
                for j,v in it:
                    if v<threshold:
                        low.append(out[0][j])
                        break
                for j,v in it:
                    if v>threshold:
                        high.append(out[0][j])
                        break




#from sls_detector_tools import root_helper as r
#
#
#c,h = r.hist(low, xmin = 500, xmax = 1800, bins = 30)
#c.Draw()
#
#c1,h1 = r.hist(high, xmin = 500, xmax = 1800, bins = 30)
#c1.Draw()
                        
low = np.asarray(low)
high = np.asarray(high)                    
           
print('L: Mean: {:.2f} Std: {:.2f}'.format(low[low>600].mean(), low[low>600].std()))
print('H: Mean: {:.2f} Std: {:.2f}'.format(high.mean(), high.std()))         

import seaborn as sns
sns.set()
sns.set_style('white')
plt.grid(True)
import matplotlib.patches as patches
max_value = 256*256*1.1


colors = sns.color_palette()
fig = plt.figure()
ax = fig.add_subplot(111)
for interval in cfg.tests.rxb_interval['Half Speed']:
    left, width = interval
    width *= 3
    left -= width/2
    ax.add_patch(
            patches.Rectangle(
                    (left,0),
                    width,
                    max_value,
                    fill=True,
                    alpha = 0.3,
                    )
            )    
for values in out[1:]:
    ax.plot(out[0], values)
ax.plot([1100,1100],[0, max_value], '--', color = colors[2], linewidth = 3)
ax.set_ylim(0, max_value)
plt.grid(True) 
#    
#    
#plt.figure()
#for values in out[1:]:
#    it= enumerate(values)
#    for j,v in it:
#        if v<threshold:
#            print(out[0][j])
#            break
#    for j,v in it:
#        if v>threshold:
#            print(out[0][j])
#            break
#    plt.plot(out[0], values)