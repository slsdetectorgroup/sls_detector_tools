
from pathlib import Path
import shutil

import sls_detector_tools.config as cfg
from sls_detector_tools import calibration
import os

# def material_t

#Configuration for the calibration script
cfg.geometry = '500k' #quad, 500k, 2M, 9M
cfg.calibration.type = 'XRF' #Sets function to fit etc.
cfg.det_id = 'T128'
cfg.calibration.gain = 'gain1'
# cfg.calibration.target = 'Ti'
cfg.path.data = Path(f'/mnt/disk1/calibration/{cfg.det_id}/{cfg.calibration.gain}')

out_path = Path('/home/l_frojdh/tmp/settings/standard')

for i in range(1,9,1):
    cfg.calibration.gain = f'gain{i}'
    path = Path(f'/mnt/disk1/calibration/{cfg.det_id}/{cfg.calibration.gain}')
    
    # print(f'src: {path}')
    tbfiles = [f for f in os.listdir(path) if '.sn' in f]
    for f in tbfiles:
        
        material = f.split('_')[1][0:2]
        ending = f.split('.')[-1]
        energy = int(calibration.xrf[material]*1000)
        # print(f'{f} : {energy}')
        src = path/f
        dst = out_path/f'{energy}eV/noise.{ending}'
        print(src)
        print(dst)
        # p.mkdir(parents=True, exist_ok=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


    

    # print(f'dst: {dst}')




