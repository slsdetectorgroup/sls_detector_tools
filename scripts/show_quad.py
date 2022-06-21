import hdf5maker as h5m

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from sls_detector_tools.plot import imshow
path = Path('/home/l_msdetect/erik/data/calibration/EM15/images')
path = Path('/home/l_msdetect/erik/data/calibration/EM16/gain5')

# image = h5m.RawFile(path/'run_master_0.raw').read()[0]

with np.load(path/'EM16_CuXRF_gain5.npz') as f:
    tb = f['trimbits']
img = np.zeros((512,512))
img[256:, :] = tb[:,0:512]
img[0:256,:] = np.rot90(tb[:,512:],2)
# # ax, im = imshow(image)
# image = np.zeros((515,515))
# data1 = h5m.raw_data_file.RawDataFile(path/'quad2_FeXRF_150V_d0_f0_0.raw', frame_size = 256*512, dr=32, frames_per_file = [1]).read()[0]
# data2 = h5m.raw_data_file.RawDataFile(path/'quad2_FeXRF_150V_d1_f0_0.raw', frame_size = 256*512, dr=32, frames_per_file = [1]).read()[0]

# image[0:257] = data2
# image[258:] = data1
ax, im = imshow(img)
ax.set_title("Quad SIM2 - trimbits")
plt.savefig(path/'sim2_tb')