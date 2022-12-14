from slsdet import Eiger, defs
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from sls_detector_tools.receiver import ZmqReceiver
from sls_detector_tools import mask
import sls_detector_tools.config as cfg
cfg.geometry = '9M'
# energies = np.arange(6500,10000,100)
d = Eiger()
d.exptime = 1



# d.setThresholdEnergy(6400)
# d.dacs.vrpreamp[3] = 0
# d.trimval = 32

# vrpreamp = d.dacs.vrpreamp.get()
# for i in [2,3]:
#     d.dacs.vrpreamp[i] = vrpreamp[i] -12

r = ZmqReceiver(d)
d.acquire()
image = r.get_frame()

fig, ax = plt.subplots()
im = ax.imshow(image)
im.set_clim(0,100)
# data = np.zeros((1024+32, 1024))
# data[0:512,:] = tmp[0:512,:]
# data[512+32:,:] = tmp[512:,:]
# m = data > data.mean()*1.5
# data[m] = 0
# fig, ax = plt.subplots()
# im = ax.imshow(data, origin = 'lower')
# im.set_clim(0,data.mean()*1.5)

# half_modules = mask.eiger1M().halfmodule

# fig, ax = plt.subplots()
# y = data[:, 600:650].sum(axis = 1)
# ax.plot(y)
# ax.plot([0,1024], [y.mean(),y.mean()])

# ratio = y[0:512].sum()/y[512+32:].sum()
# print(f'ratio: {ratio} / {320/450}')

# m = data[0] > 3*data[0].mean()

# for frame in data:
#     frame[m] = 0

