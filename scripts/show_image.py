import hdf5maker as h5m

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from sls_detector_tools.plot import imshow
path = Path('/home/l_msdetect/erik/data/calibration/EM15/images')

image = h5m.RawFile(path/'run_master_0.raw').read()[0]


ax, im = imshow(image, cmap = 'viridis')
im.set_clim(0,18e3)
ax.set_title(f'T98 Cu XRF th=4.0keV (non connected pixels = {(image<100).sum()})')
plt.savefig(path/'T98_CuXRF')