

from sls_detector import Eiger
import time

d = Eiger()
d.file_path = '/home/l_frojdh/data'
d.file_write = True
d.file_index = 0
d.exposure_time = 1

bias = [20, 150]
module = 'T109'
d.dynamic_range = 32

d.trimbits = 60

for b in bias:
    d.high_voltage = b
    d.file_name = f'{module}_CuXRF_{b}V'
    d.file_index = 0
    time.sleep(1)
    d.acq()

d.file_write = False