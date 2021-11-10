import ROOT
from ROOT import TF1
from pathlib import Path
import os
import numpy as np
from sls_detector_tools import io   
import sls_detector_tools.root_helper as r
import matplotlib.pyplot as plt
import sls_detector_tools.function as sf
plt.ion()

fun = "pol3"
pyfunc = eval(f"sf.{fun}")
target = 3300

path = Path('/home/l_msdetect/erik/T98/standard')
folders = [f for f in path.iterdir() if f.name.endswith('eV') and str(target) not in f.name]
folders = [f for f in path.iterdir() if f.name.endswith('eV')]
folders.sort(key = lambda x: int(x.name.rstrip('eV')))
energy = [int(f.name.rstrip('eV')) for f in folders]

xx = np.linspace(2900, 10000)
fig, ax = plt.subplots(figsize = (14,9))
for det in ['083', '098']:

    dacs = np.zeros((len(folders), 18), dtype = np.int64)
    tb = np.zeros((len(folders), 256, 1024), dtype = np.int32)
    energy = np.array(energy)

    for i, f in enumerate(folders):
        tb[i], dacs[i] = io.read_trimbit_file(f/f'noise.sn{det}')


    c,h = r.plot(energy, dacs[:, 2], draw = False)
    func = TF1("func", fun, 3000, 17000)
    fit = h.Fit("func", "NRSQ")
    par = [func.GetParameter(i) for i in range(func.GetNpar())]



    print(dacs[:,2])

    yy = pyfunc(xx, *par)
    vrf = func(target)
    ax.plot(energy, dacs[:, 2], 'o')
    ax.plot(xx,yy, 'r')
    ax.plot(target, vrf, 'o')
    print(f"{det} vrf: {vrf}")

    # dst = path/f'{target}eV/noise.sn{det}'
    # dst.parent.mkdir(parents=True, exist_ok=True)
    # dacs[0,2] = vrf
    # # io.write_trimbit_file(dst, tb[0], dacs[0])
    # print(dst)


ax.grid()

# from sls_detector_tools.utils import random_pixel
# pixels = random_pixel(n_pixels=10, rows=(0,256))
# fig, ax = plt.subplots()
# for pixel in pixels:
#     ax.plot(tb[:,pixel[0], pixel[1]], 'o-')