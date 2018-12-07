from .instruments import AgilentMultiMeter

try:
    from .keithley import SourceMeter
except ModuleNotFoundError:
    print('Serial is missing. Keithley is disabled')
from .xray_box import XrayBox, BigXrayBox, VacuumBox, DummyBox, xrf_shutter_open
from .load_tiff import load_tiff
from .io import load_frame

try:
    from .plot import histogram
except ModuleNotFoundError:
    print('_sls_cmodule not foound, support for compiled histograms is disabled')    

try:
    from .receiver import ZmqReceiver
except ModuleNotFoundError:
    print('zmq not found receiver disabled')    
