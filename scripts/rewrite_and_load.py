from sls_detector_tools.io import read_trimbit_file, write_trimbit_file

from pathlib import Path
from slsdet import Eiger
import numpy as np
import matplotlib.pyplot as plt

path = Path('/home/l_msdetect/erik/data/calibration/MS1M/gain4')
ifname = 'MS1M_FeXRF_gain4.sn053'
ofname = 'out.sn000'
tb, dacs = read_trimbit_file(path/ifname)



# tmp.flat = tmp.flat[::-1]


tb[10:20, 0:100] = 63
tb[100:120,200:500] = 63
tb[10:20, 600:1000] = 63
tmp = tb.copy()
tb[:,0:512] = tb[:,0:512][:,::-1]
tb[:,512:] = tb[:,512:][:,::-1]


fig, ax = plt.subplots()
im = ax.imshow(tmp, origin = 'lower')


write_trimbit_file(path/ofname, tb,dacs )

d = Eiger()
d.loadTrimbits((path/ofname).as_posix(), [3])
plt.show()