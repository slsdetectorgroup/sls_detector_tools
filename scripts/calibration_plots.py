
# -*- coding: utf-8 -*-
"""
Script to calibrate an EIGER module using the big X-ray box.
"""
import ROOT
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
from sls_detector_tools.function import scurve
from sls_detector_tools import mask
import sls_detector_tools.root_helper as r


def tb_histograms(trimbits):
    fig, ax = plt.subplots()
    for m in mask.chip:
        
        c,h = r.hist(trimbits[m], xmin = 0, xmax = 63, bins = 64, draw =False)
        x,y = r.getHist(h)
        ax.plot(x,y)
    return fig, ax

def scale_data(data, scale = None):
    if scale is None:
        scale =  data[:,:,-1] / 1000.
    for i in range(data.shape[2]):
        data[:,:,i] /= scale
    data[np.isnan(data)] = 0
    return data, scale

def plot_pixel(pixel):
    fig, ax = plt.subplots(1,2)
    ax[0].plot(x, data[pixel[0], pixel[1],:], 'o')
    ax[0].plot(xx, scurve(xx, *fit_result[pixel]))
    ax[0].plot(fit_result['mu'][pixel], target[pixel], 'o', color = 'red')
    
    ax[1].plot(tbx, tbdata[pixel[0], pixel[1],:], 'o')
    tmp = tuple(a for a in tb_fit[pixel])
    par = tmp[0:6]
    tbit = tmp[6]
    ax[1].plot(xx_tb, scurve(xx_tb, *par))
    ax[1].plot((0,64), (target[pixel], target[pixel]), color = 'red')
    ax[1].plot((tbit, tbit), (0,1000), color = 'red')
    return fig, ax


#Configuration for the calibration script
cfg.geometry = '500k' #quad, 500k, 2M, 9M
cfg.calibration.type = 'XRF' #Sets function to fit etc.
cfg.det_id = 'T128'
cfg.calibration.gain = 'gain2'
cfg.calibration.target = 'Ti'
cfg.path.data = os.path.join('/mnt/disk1/calibration/',
                            cfg.det_id, cfg.calibration.gain)

cfg.calibration.run_id = 0

#plot vcmp
with np.load(os.path.join(cfg.path.data, calibration.get_data_fname())) as f:
    data = f['data']
    x = f['x']

fit_result = np.load(os.path.join(cfg.path.data, calibration.get_fit_fname()))


# calibration._plot_scurve(data, x)
# mean, std, lines = chip_histograms( fit_result['mu'] )
# plt.xlabel('Vcmp [DAC LSB]')
# plt.ylabel('Number of Pixels')

#Load trimbit data
with np.load(os.path.join(cfg.path.data, calibration.get_tbdata_fname())) as f:
    tbdata = f['data']
    tbx = f['x']
    
# calibration._plot_trimbit_scan(tbdata, tbx)

with np.load(os.path.join(cfg.path.data, calibration.get_trimbit_fname()+'.npz')) as f:
    tb = f['trimbits']
    tb_fit = f['fit']
  

target = scurve( fit_result['mu'], 
                fit_result['p0'],
                fit_result['p1'],
                fit_result['mu'],
                fit_result['sigma'], 
                fit_result['A'],
                fit_result['C'])

xx = np.linspace(0,2000)
xx_tb = np.linspace(0,64)
data, scale = scale_data(data)
scale_data(tbdata, scale)

i,j = np.where(tb == 0)
pixels = [it for it in zip(i,j)]

print(f"Found {len(pixels)} pixels with value 0")

plot_pixel(pixels[30])

# a = tb_histograms(tb)

# plot_pixel((500,600))

c,h = r.hist(tb, xmin =0, xmax = 64, bins = 65)
# c.SetLogy()

# ax, im = imshow(tb)
# ax.set_title('Trimbit map')
# a = calibration._plot_trimbit_histogram(tb)

# failed = tb ==0
# ax, im = imshow(failed)
# cfg.calibration.run_id = 1

# #plot vcmp
# with np.load(os.path.join(cfg.path.data, calibration.get_data_fname())) as f:
#     data = f['data']
#     x = f['x']

# fit_result = np.load(os.path.join(cfg.path.data, calibration.get_fit_fname()))


# # calibration._plot_scurve(data, x)
# mean, std, lines = chip_histograms( fit_result['mu'] )
# plt.xlabel('Vcmp [DAC LSB]')
# plt.ylabel('Number of Pixels')

