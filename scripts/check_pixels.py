
from sls_detector_tools import calibration
from sls_detector_tools import mpfit
from sls_detector_tools.plot import imshow
import sls_detector_tools.config as cfg
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sls_detector_tools.function import scurve
import os
plt.ion()


def scale_data(data, scale = None):
    if scale is None:
        scale =  data[:,:,-1] / 1000.
    for i in range(data.shape[2]):
        data[:,:,i] /= scale
    data[np.isnan(data)] = 0
    return data, scale


cfg.geometry = '500k' #quad, 500k, 2M, 9M
cfg.calibration.type = 'XRF' #Sets function to fit etc.
cfg.det_id = 'TQ1'
cfg.calibration.gain = 'gain5'
cfg.calibration.target = 'Cu'
cfg.calibration.energy = 8
cfg.path.data = os.path.join('/home/l_msdetect/erik/quad/data',
                             cfg.det_id, cfg.calibration.gain)

path = Path(cfg.path.data)


cfg.calibration.run_id = 0

with np.load(path/calibration.get_data_fname()) as f:
    data = f['data']
    x = f['x']

data[:, 512:,:] = 0


with np.load(path/calibration.get_tbdata_fname()) as f:
    data_tb = f['data']
    x_tb = f['x']

fit_result = np.load(path/calibration.get_fit_fname())

target = scurve( fit_result['mu'], 
                fit_result['p0'],
                fit_result['p1'],
                fit_result['mu'],
                fit_result['sigma'], 
                fit_result['A'],
                fit_result['C'])


with np.load(f'{path/calibration.get_trimbit_fname()}.npz') as f:
    tb_fit = f['fit']
    tb = f['trimbits']


par = np.array([9.35, 2.08, 33.5 , 14.47, 507, 4])
result = mpfit.find_trimbits(data_tb, x_tb, target, cfg.calibration.nproc, par)
n_failed = (np.isnan(result['trimbits'])).sum()
print(f'Num failed pixels: {n_failed}')

i,j = np.where(np.isnan(result['trimbits']))
pixels = [it for it in zip(i,j)]
xx = np.linspace(0,2000)
xx_tb = np.linspace(0,64)
data, scale = scale_data(data)
scale_data(data_tb, scale)


#pixel = (100,100)

def plot_pixel(pixel):
    fig, ax = plt.subplots(1,2)
    ax[0].plot(x, data[pixel[0], pixel[1],:], 'o')
    ax[0].plot(xx, scurve(xx, *fit_result[pixel]))
    ax[0].plot(fit_result['mu'][pixel], target[pixel], 'o', color = 'red')
    
    ax[1].plot(x_tb, data_tb[pixel[0], pixel[1],:], 'o')
    tmp = tuple(a for a in tb_fit[pixel])
    par = tmp[0:6]
    tbit = tmp[6]
    ax[1].plot(xx_tb, scurve(xx_tb, *par))
    ax[1].plot((0,64), (target[pixel], target[pixel]), color = 'red')
    ax[1].plot((tbit, tbit), (0,1000), color = 'red')
    return fig, ax
