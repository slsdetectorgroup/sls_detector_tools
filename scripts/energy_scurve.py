# from sls_detector_tools import sls_detector_tools
from slsdet import Eiger, defs
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from sls_detector_tools.receiver import ZmqReceiver
from sls_detector_tools import mask
import sls_detector_tools.config as cfg
cfg.geometry = '1M'
energies = np.arange(6500,10000,100)
d = Eiger()
d.exptime = 0.1
r = ZmqReceiver(d)
data = np.zeros((energies.size, 1024,1024))

d.setThresholdEnergy(8000)

for i,energy in enumerate(energies):
    d.setThresholdEnergy(energy, defs.STANDARD, False)
    d.acquire()
    data[i] = r.get_frame()


half_modules = mask.eiger1M().halfmodule

m = data[0] > 3*data[0].mean()

for frame in data:
    frame[m] = 0

fig, ax = plt.subplots(1,2)
for hm in half_modules:
    y = data[[slice(None, None, None), *hm]].sum(axis = 1).sum(axis = 1)
    ax[0].plot(energies, y)
    ax[1].plot(energies, -np.gradient(y))