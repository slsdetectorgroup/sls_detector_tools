
from pathlib import Path
import shutil

import sls_detector_tools.config as cfg
from sls_detector_tools import calibration
import os


#Configuration for the calibration script
cfg.geometry = '500k' #quad, 500k, 2M, 9M
cfg.calibration.type = 'XRF' #Sets function to fit etc.
cfg.det_id = 'T98'
# cfg.calibration.gain = 'gain1'
# cfg.calibration.target = 'Ti'
# cfg.path.data = os.path.join('/home/l_msdetect/erik/data/calibration/',
#                              cfg.det_id, cfg.calibration.gain)

out_path = Path('/home/l_msdetect/erik/T98/standard/')

for i in [8]:
    cfg.calibration.gain = f'gain{i}'
    path = Path(os.path.join('/home/l_msdetect/erik/data/calibration/',
                             cfg.det_id, cfg.calibration.gain))
    
    # print(f'src: {path}')
    tbfiles = [f for f in os.listdir(path) if '.sn' in f]
    for f in tbfiles:
        material = f.split('_')[1][0:2]
        ending = f.split('.')[-1]
        energy = int(calibration.xrf[material]*1000)
        src = path/f
        dst = out_path/f'{energy}eV/noise.{ending}'
        print(src)
        print(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


    

    # print(f'dst: {dst}')




