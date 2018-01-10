# -*- coding: utf-8 -*-
"""
The main funtions to record and fit calibration data. An example on how to 
use the functions can be found in **sls_detector_tools/calibration/trim_and
_calibrate_vrf.py**

The fitting relies on the routines in sls_cmodule

.. todo::
    Should we moved to scaled fits for all detectors?

"""
#ROOT
#import ROOT
#from ROOT import TF1
    


#Python imports
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import time

#Copy files
import shutil

#sls_detector imports
#from . import root_helper as r
from . import plot as plot
from . import utils as u
from . import config as cfg
from . import function
from . import mask
from . import io 
from . import ZmqReceiver
from . import xrf_shutter_open
from . import mpfit

#compiled
from _sls_cmodule import vrf_fit

from contextlib import contextmanager
@contextmanager
def setup_measurement(detector):
    """
    Contexmanger that is used in many of the tests. Sets up the detector 
    and returns a receiver which can be used to get images from the receiver
    zmq stream.
    
    ::
        
        clk = 'Full Speed'
        with setup_test_and_receiver(detector, clk) as receiver:
            detector.acq()
            data = receiver.get_frame()
            
        
    """
    #Setup for test
    detector.rx_datastream = False
    time.sleep(0.1)
    detector.rx_datastream = True
    detector.file_write = False
#    detector.dynamic_range = 16 
#    detector.readout_clock = clk
#    detector.exposure_time = 0.01 
    dacs = detector.dacs.get_asarray()  
    
    yield ZmqReceiver('10.1.1.100', detector.rx_zmqport)
    
    #Teardown after test
    detector.dacs.set_from_array( dacs )



xrf = {'In':3.29,'Ti': 4.5, 'Cr': 5.4, 'Fe': 6.4, 'Cu': 8.02, 'Ge': 9.9, 'Zr': 15.7,
'Mo' : 17.5, 'Ag' : 22.2, 'Sn': 25.3}
#xrf = {'In':3.29,'Ti': 4.5, 'Cr': 5.4, 'Fe': 6.4, 'Cu': 8.02, 'Ge': 9.9, 'Zr': 15.7,
#'Mo' : 17.5, 'Ag' : 22.2, 'In':24.2, 'Sn': 25.3}


def get_data_fname(run_id = None):
    """
    Get the filename of the npz file with the vcmp data. The filename
    is built from det_id, target and run_id as well as calibration.type.
    Allows for convenient access during calibration.
    
    Parameters
    ---------
    run_id: int, optional
        Used only if requesting a special run_id otherwise fetched from
        cfg.calibration.run_id
        
    Returns
    ----------
    fname: str
        filename including .npz ending
    
    Examples 
    -------
    
    ::
        
        cfg.det_id = T45
        cfg.run_id = 1
        cfg.calibration.target = Cr
        get_data_fname()
        >> T45_vcmp_CrXRF_1.npz
    
    """
    #Use the run_id specified in cfg.calibration.run_id
    if run_id is None:
        run_id = cfg.calibration.run_id
        
    if cfg.calibration.type == 'XRF':
        fname = '{:s}_vcmp_{:s}XRF_{:d}.npz'.format(cfg.det_id, cfg.calibration.target, run_id)
    elif cfg.calibration.type == 'TP':
        fname = '{:s}_vcmp_tp_{:.2f}keV_{:d}.npz'.format(cfg.det_id, cfg.calibration.energy, run_id)
    elif cfg.calibration.type == 'beam':
        fname = '{:s}_vcmp_beam_{:.2f}keV_{:d}.npz'.format(cfg.det_id,  cfg.calibration.energy, run_id)
    return fname

def get_fit_fname( run_id = None ):
    """
    Get the name of the file containing the fit result. Shares the structure 
    with the data file but ends with **_fit.npy** instead
    
    Parameters
    ---------
    run_id: int, optional
        Used only if requesting a special run_id otherwise fetched from
        cfg.calibration.run_id
    
    Returns
    ---------- 
    fname: str
            name of the fit file
    """
    fname = get_data_fname( run_id = run_id ).strip('.npz') + '_fit.npy'
    return fname


def get_tbdata_fname():
    """
    Get the name of the npz file containing the trimbit data. *Note that 
    this file does not have a run_id*
    
    
    Returns
    ---------- 
    fname: str
            Name of the file
    """
    if cfg.calibration.type == 'XRF':
        fname = '{:s}_tbscan_{:s}XRF_{:d}.npz'.format(cfg.det_id, cfg.calibration.target, cfg.calibration.run_id)
    elif cfg.calibration.type == 'TP':
        fname = '{:s}_tbscan_tp_{:.2f}keV_{:d}.npz'.format(cfg.det_id, cfg.calibration.energy, cfg.calibration.run_id)
    elif cfg.calibration.type == 'beam':
        fname = '{:s}_tbscan_beam_{:.2f}keV_{:d}.npz'.format(cfg.det_id, cfg.calibration.energy, cfg.calibration.run_id)
    return fname
   
def get_trimbit_fname():
    """
    Get the name of the trimbit files without the file ending. This is 
    because when passed by commandline to the slsDetectorSoftware this then
    loads trimbits for all modules based on their id.
    
    
    Returns
    ---------- 
    fname: str
            Name of the file
    """
    if cfg.calibration.type == 'XRF':
        fname = '{:s}_{:s}XRF_{:s}'.format(cfg.det_id, cfg.calibration.target, cfg.calibration.gain)
    elif cfg.calibration.type == 'TP':
        fname = '{:s}_tb_{:.2f}keV_{:s}'.format(cfg.det_id,  cfg.calibration.energy, cfg.calibration.gain)
    elif cfg.calibration.type == 'beam':
        fname = '{:s}_beam_{:.2f}keV_{:s}'.format(cfg.det_id, cfg.calibration.energy, cfg.calibration.gain)
    return fname  

def get_vrf_fname():
    """
    Get the name of the npz file containing the vrf scan data
    
    
    Returns
    ---------- 
    str
        Filename
            
            
    """
    if cfg.calibration.type == 'XRF':
        fname = '{:s}_vrf_{:s}XRF_{:d}.npz'.format(cfg.det_id, cfg.calibration.target, cfg.calibration.run_id)
    elif cfg.calibration.type == 'TP':
        fname = '{:s}_vrf_tp_{:.2f}keV_{:d}.npz'.format(cfg.det_id,  cfg.calibration.energy, cfg.calibration.run_id)
    elif cfg.calibration.type == 'beam':
        fname = '{:s}_vrf_beam_{:.2f}keV_{:d}.npz'.format(cfg.det_id, cfg.calibration.energy, cfg.calibration.run_id)
    return fname           

def get_halfmodule_mask():
    """
    Get the masks for all half modules in the detector using the geometry from
    cfg.geometry
    
    Returns
    -------
    list with slices
        A list containing slice objects to select each half module
    
    Raises
    -------
    NotImplementedError
        If the selected geometry is not supported
    
    Examples
    --------
    
    Select a half module from a module ::
        
        hm = get_halfmodule_mask()
        data_from_halfmodule = data[hm[0]]
    
    """
    if cfg.geometry == '500k':
        a = mask.eiger500k()
        return a.halfmodule
    elif cfg.geometry == '2M':
        a  =mask.eiger2M()
        return a.halfmodule
    elif cfg.geometry == '9M':
        a = mask.eiger9M()
        return a.halfmodule
    else:
        raise NotImplementedError('Half module mask doses not exist for the'\
                                  'selected geometry:', cfg.geometry)
        

def setup_detector(detector):
    """
    Make sure that the detector is in a correct state for calibration.
    Settings that are applied are taken from the config file:
    
    * Number of frames
    * Period
    * Exposure time
    * Dynamic range
    * clock divider
    * trimbit used during initial scurve
    * V_trim
    * Vrs
    
    """
    #Exposure
    detector.n_frames = cfg.calibration.nframes
    detector.period = cfg.calibration.period
    if cfg.calibration.type == 'TP':
        detector.exposure_time = cfg.calibration.tp_exptime
    else:
        detector.exposure_time = cfg.calibration.exptime
    
    #Data format and communication
    if cfg.calibration.type == 'TP':
        detector.dynamic_range = cfg.calibration.tp_dynamic_range 
    else:
        detector.dynamic_range = cfg.calibration.dynamic_range
        
    detector.readout_clock = cfg.calibration.speed
#    for flag in cfg.calibration.flags:
#        detector.set_flags( flag )
    
    #Trimming
    detector.trimbits = cfg.calibration.trimval
    detector.dacs.vtr = cfg.calibration.vtr
    detector.dacs.vrs = cfg.calibration.vrs
    
#    detector.set_dac('vthreshold', cfg.calibration.threshold)    
    
#    detector.set_fwrite( True )
#    detector.set_overwrite(True)
#    detector.s



def _vrf_scan(detector, start=1500, stop = 3800, step = 30):
    """
    Record a vrf scan for choosing gain before starting to trim. 
    This can be used as a stand alone function but is also used in the
    :py:meth:`sls_detector.calibration.do_vrf_scan`
    """
     #Switch to 16bit since we always scan this fast
    dr = detector.dynamic_range
    detector.dynamic_range = 16 
    detector.vthreshold = cfg.calibration.threshold  
    detector.exposure_time = cfg.calibration.vrf_scan_exptime

    vrf_array = np.arange(start, stop, step)
    print(vrf_array)
    
    _s = detector.image_size
    data = np.zeros((_s.rows, _s.cols, vrf_array.size))
    
    with setup_measurement(detector) as receiver:
        for i,v in enumerate(vrf_array):
            detector.dacs.vrf = v
            print(detector.dacs.vrf)
            detector.acq()
            data[:,:,i] = receiver.get_frame()
    
    #Reset dr
    detector.dynamic_range = dr
    return data, vrf_array

def _threshold_scan(detector, start = 0, stop = 2001, step = 40):
     #Switch to 16bit since we always scan this fast

    detector.dynamic_range = cfg.calibration.dynamic_range 
    detector.exposure_time = cfg.calibration.exptime

    threshold = np.arange(start, stop, step)

    _s = detector.image_size
    data = np.zeros((_s.rows, _s.cols, threshold.size))
    
    with setup_measurement(detector) as receiver:
        for i,th in enumerate(threshold):
            detector.vthreshold = th
            print(detector.vthreshold)
            detector.acq()
            data[:,:,i] = receiver.get_frame()
    
    return data, threshold    


def _clean_vrf_data(data):
    """
    Clean the data based on median
    """
    _pm = np.zeros((data.shape[0], data.shape[1]), dtype = np.bool)
    for i in range(data.shape[2]):
        _threshold = np.median(data[:,:,i])*3
        if _threshold < 50:
            _threshold = 50
        _pm[data[:,:,i]>_threshold] = True
    for i in range(data.shape[2]):
        data[:,:,i][_pm] = 0
    return data

def _fit_and_plot_vrf_data(data, x, hostnames):
    """
    Plot and fit
    """
    vrf = []
    
    if cfg.calibration.plot:
        colors = sns.color_palette(  n_colors = len(hostnames) )
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,7))
        xx = np.linspace(x.min(), x.max(), 300)
    
    
    halfmodule = get_halfmodule_mask()
    
    for i in range( len(halfmodule) ):
        y = data[halfmodule[i]].sum(axis = 0).sum(axis = 0)
        yd = np.gradient( y )
        center = np.argmax( yd )
        print( center )
        if center > 75:
            xmin = x[72]
            xmax = x[-1]
        else:
            xmin = x[center-4]
            xmax = x[center+3]
            print( xmin, xmax )
        
        #Graph and fit function
#        c,h = r.plot(x,yd)
#        func = TF1('func', 'gaus', xmin, xmax)
#        fit = h.Fit('func', 'SQR') 
#        
#        par = [ fit.Get().Parameter(j) for j in range(func.GetNpar()) ]
        
        par = vrf_fit(x, yd, np.array((xmin, xmax)))
        print(len(hostnames), hostnames)
        if cfg.calibration.plot:
            ax1.plot(x, y, 'o', color = colors[i], label = hostnames[i])
            ax2.plot(x, yd, 'o', color = colors[i], label = '$\mu: ${:.0f}'.format(par[1]))
            ax2.plot(xx, function.gaus(xx, *par), color = colors[i])
        
#        vrf.append( int( np.round( fit.Get().Parameter(1))) )
        vrf.append( int( np.round(par[1])) )
    
    if cfg.calibration.plot:
        ax1.legend(loc = 'upper left')
        ax1.set_ylabel('Counts [1]')
        ax1.set_xlabel('vrf [1]')
        ax2.legend(loc = 'upper left')
        ax2.set_ylabel('Counts (differential) [1]')
        ax2.set_xlabel('vrf [1]')
        ax2.set_xlim(xmin-100, xmax+100)
        fig.suptitle('{:s} vrf scan: {:s}'.format(cfg.det_id, cfg.calibration.target))
        plt.tight_layout()
        plt.savefig( os.path.join(cfg.path.data, get_vrf_fname().strip('.npz')) ) 
    
    return vrf

def do_vrf_scan(detector, xraybox, pixelmask = None, 
                start = 1500, 
                stop = 3800, 
                step = 30):
    """
    Does a vrf scan and fits the differential of the scurve for each halfmodule
    in the detector system. 
    
    .. todo::
        Check the multi module system support
        
    Parameters
    ----------
    detector: SlsDetector
        The detector that should be scanned
    xraybox: XrayBox or DummyBox
        Used for selecting the right target and controlling the shutter
    pixelmask: np_array(bool), optional
        Numpy array of bools of the same size and one frame or None to disable
    start: int, optional
        start value of the scan
    stop: int, optional 
        end value of the scan
    step: int, optional 
        stepsize
    
    Returns
    -------
    vrf: list
        list of vrf values for each half module
    t: float
        Suggested exposure time for the scurve
        
        
    Examples
    ---------
    
    ::
        
        vrf, t = calibration.do_vrf_scan(d, box)
        
        
    .. image:: _static/vrf_scan.png
        
        
    """   

    #data taking in the xray box        
    with xrf_shutter_open(xraybox, cfg.calibration.target):
        data,x = _vrf_scan(detector, start, stop, step)



    #Set pixels that are True in the mask to zero for all scan steps
    if pixelmask is not None:
        for i in range( data.shape[2] ):
            data[:,:,i][pixelmask] = 0
   
    data = _clean_vrf_data(data)
    vrf = _fit_and_plot_vrf_data(data, x, detector.hostname)
    
    #Save vrf?
    cts = [data[:,:,np.argmin(np.abs(x-vrf[i]))].mean() for i in range(detector.n_modules)]
    t = (1000*cfg.calibration.vrf_scan_exptime)/min([data[:,:,np.argmin(np.abs(x-vrf[i]))].mean() for i in range(detector.n_modules)])
    print('Suggested exptime: {:.2f}'.format(t))
    

    return vrf, t  , cts
    

def find_mean_and_set_vcmp(detector, fit_result):
    """
    Find the mean value of the inflection point per chip and set the **Vcmp**
    of that chip to the right value
    
    .. todo::
        Merge to one code for all geomtetries
    
    Supported geometries
     
    * 250k
    * 500k
    * 2M
    * 9M
    
    Parameters
    ---------
    detector: SlsDetector
        The detector that should be used. Can be None to skip the set dac 
        part and only return values
    fit_result: numpy_array
        An array with the result of the per pixel scurve fit
        
    Returns
    -------
    vcmp: list
        vcmp for each chip
    vcp: list
        vcp for each half module
    lines: list
        list of lines that can be used with the sls_detector_put
    

    
    """
    #Mean value for each chip to be used as threshold during scan    
    mean = np.zeros( cfg.nmod*4, dtype = np.int )
    
    #Find the mean values for both module and half module
    if cfg.geometry == 'quad':
        #Half module
        for i in range( 4, 8, 1):
            m = fit_result['mu'][mask.chip[i]]
            th = int( m[(m>10) & (m<1990)].mean() )
            detector.set_dac(mask.vcmp[i-4], th)
            mean[i - 4] = th
            
        vcp0 = int( mean[0:4][mean[0:4]>0].mean() )
        detector.set_dac('0:vcp', vcp0)
        
    elif cfg.geometry == '500k':
        #OK for Python API
        for i in range( cfg.nmod*4 ):
            m = fit_result['mu'][mask.chip[i]]
            try:
                th = int( m[(m>100) & (m<1900)].mean() )
            except:
                th = 0
            mean[i] = th
        
        vcp0 = int( mean[0:4][mean[0:4]>0].mean() )
        vcp1 = int( mean[4:][mean[4:]>0].mean() )
        detector.vcmp = mean
        detector.dacs.vcp = [vcp0, vcp1]

        
    elif cfg.geometry == '2M':
        #Module stuff
        vcmp = np.zeros( (len(mask.eiger2M.module), 8) )
        vcp  = np.zeros( (len(mask.eiger2M.module), 2) )
        
        for j,mod in enumerate( mask.eiger2M.module):
            for i in range( 8 ):
                m = fit_result['mu'][mod][mask.chip[i]]
                try:
                    th = int( m[(m>10) & (m<1990)].mean() )
                except:
                    th = 0
                vcmp[j,i] = th
                
                if type(detector) != type( None ):
                    detector.set_dac(mask.eiger2M.vcmp[j*8+i], th)
                    
                mean[i] = th
        
            vcp0 = int( mean[0:4][mean[0:4]>0].mean() )
            vcp1 = int( mean[4:][mean[4:]>0].mean() )
            vcp[j,0] = vcp0
            vcp[j,1] = vcp1
            
            if type( detector ) != type(None):
                detector.set_dac('{:d}:vcp'.format(j*2), vcp0)
                detector.set_dac('{:d}:vcp'.format(j*2+1), vcp1)  


    elif cfg.geometry == '9M':
        print( 'Geometry == ', cfg.geometry )
        lines = []
        
        #Module stuff
        dm = mask.detector[cfg.geometry]
        vcmp = np.zeros( (len(dm.module), 8) )
        vcp  = np.zeros( (len(dm.module), 2) )
        
        for j,mod in enumerate( dm.module ):
            for i in range( 8 ):
                m = fit_result['mu'][mod][mask.chip[i]]
                try:
                    th = int( m[(m>10) & (m<1990)].mean() )
                except:
                    th = 0

                vcmp[j,i] = th
                
#                if type(detector) != type( None ):
##                    detector.set_dac(mask.eiger9M.vcmp[j*8+i], th)
  
                #Integer division!
                lines.append('./sls_detector_put {:s} {:d}'.format( dm.vcmp[j*8+i], th) )
                    
                mean[i] = th
        
            vcp0 = int( mean[0:4][mean[0:4]>0].mean() )
            vcp1 = int( mean[4:][mean[4:]>0].mean() )
            vcp[j,0] = vcp0
            vcp[j,1] = vcp1
            
#            if type( detector ) != type(None):
                
#                detector.set_dac('{:d}:vcp'.format(j*2), vcp0)
#                detector.set_dac('{:d}:vcp'.format(j*2+1), vcp1) 
            
            lines.append('./sls_detector_put {:d}:vcp {:d}'.format(j*2, vcp0))
            lines.append('./sls_detector_put {:d}:vcp {:d}'.format(j*2+1, vcp1))
        
        if detector is not None:
            print('Setting vcmp')
            for i, v in enumerate(vcmp.flat):
                detector.vcmp[i] = int(v)
            detector.dacs.vcp = vcp.astype(np.int).flat[:]
        
        return vcmp, vcp, lines


def find_initial_parameters(x,y, thrange = (0,2200)):
    """
    Tries to find the best initial parameters for an per pixel scurve fit
    by using the global average of a chip or module. 
    Parameters
    ----------
    x: numpy_array
        x values
    y: numpy_array 
        y values
    thrange: (low, high)
        low and high edge for the fitting
        
    Returns
    --------
    par: numpy_array
        Array with the initial parameters of the fit
        
    .. todo::
        Remove hardcoded values where possible and be more flexible 
        with the range
    """
    
    func = TF1('func', function.root.scurve, thrange[0], thrange[1])
    if cfg.calibration.type == 'XRF':
        
        func.SetParLimits(0, 0,y[0]*5)
        func.SetParLimits(2,thrange[0],thrange[1])
        func.SetParLimits(3,0,500)
        func.SetParLimits(4, 0, y[-1])
        func.SetParameter(5,1)
        ipar = np.array([y[0],0,1200,200,1500,1], dtype = np.double)
   
    elif cfg.calibration.type == 'TP':
        #Assumes 1000 pulses
        func.FixParameter(0,0)
        func.FixParameter(1,0)
        func.FixParameter(4,1000)
        func.FixParameter(5,0)
        ipar = np.array([0,0,200,200,1000,0], dtype = np.double)
        
    elif cfg.calibration.type == 'beam':
        raise NotImplemented("type == beam currently not implemented")
    
    #Set par and fit
    func.SetParameters(ipar)
    c,h = r.plot(x, y, draw = False)
    fit = h.Fit('func', 'NRSQ')        
    par = np.array( [fit.Get().Parameter(i) for i in range(6)] )
    return par


def _plot_scurve(data, x):
    """
    The purpouse of this plot is to verify that the scurve data is ok
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,7))
    for p in u.random_pixel(n_pixels = 50, rows = (0, data.shape[0],), cols = (0, data.shape[1])):
        ax1.plot(x, data[p[0], p[1], :])
    
    for c in mask.chip:
        ax2.plot(x, data[c].sum(axis = 0).sum(axis = 0))
        
    ax1.set_xlabel('vcmp [1]')
    ax1.set_ylabel('counts [1]')
    ax1.set_title('Sample pixels')
    ax2.set_xlabel('vcmp [1]')
    ax2.set_ylabel('counts [1]')
    ax2.set_title('Sum per chip')    
    
    
    fig.suptitle('{:s} threshold scan: {:s}'.format(cfg.det_id, cfg.calibration.target))
    fig.tight_layout()
    plt.savefig( os.path.join( cfg.path.data, get_data_fname().strip('.npz') ) )
    
def do_scurve(detector, xraybox,
              start = 0, 
              stop = 2001,
              step = 40,):
    """
    Take scurve data for calibration. When not using the Xray box pass a 
    dummy xray box to the function and make sure that shutter is open and 
    target is correct!
    
        
    Examples
    ---------
    
    ::
        
        data, x = calibration.do_scurve(d, box)
        
        
    .. image:: _static/scurve.png    
    
    
    """
        
    with xrf_shutter_open(xraybox, cfg.calibration.target):
        data, x = _threshold_scan(detector, start = start, stop = stop, step = step)
        np.savez(os.path.join(cfg.path.data, get_data_fname()), data = data, x = x)

    #plotting the result of the scurve scan
    if cfg.calibration.plot:
        _plot_scurve(data, x)
   
    #Save data
#    np.savez(os.path.join(cfg.path.data, get_data_fname()), data = data, x = x)
    
    #if data should be used interactivly
    return data, x
    
def do_scurve_fit(mask = None, fname = None, thrange = (0,2000)):
    """
    Per pixel scurve fit from saved data and save the result in an npz file

    .. todo ::
        
        Move to scaled fits?
        
    Examples
    ---------
    
    ::
        
        fit_result = calibration.do_scurve_fit()
        
        
    .. image:: _static/fit_scurve.png   

    """
    #Load the scurve data
    if type(fname) == type(None):
        fname = get_data_fname()
                

    pathname = os.path.join( cfg.path.data, fname)
    with np.load( pathname ) as f:
        data = f['data']
        x = f['x']
    
    #if a mask was supplied take out the masked pixels
    if type(mask) != type(None):
        for i in range( data.shape[2] ):
            data[:,:,i][mask] = 0
     

    
    #Normalize y then find initial prameters and fit
    #Should thir be rewritten?
    
    y = data.sum(axis = 0).sum(axis = 0)
    y = y / ( data.sum(axis = 2)>0 ).sum()
    par = find_initial_parameters(x,y)

    fit_result = mpfit.fit(data, x, cfg.calibration.nproc, par)  
    
    #If specified plot and save the result
    if cfg.calibration.plot:
        mean, std, lines = plot.chip_histograms( fit_result['mu'], xmin = thrange[0], xmax = thrange[1] ) 
        plt.xlabel('Vcmp [DAC LSB]')
        plt.ylabel('Number of Pixels')
        plt.savefig( os.path.join( cfg.path.data, get_fit_fname().strip('.npy') ) )
    
    
    #Save the fit result
    fname = get_fit_fname().strip('.npy')
    pathname = os.path.join(cfg.path.data, fname)
    np.save(pathname, fit_result)
    
    return fit_result

def do_scurve_fit_scaled(  mask = None, fname = None, thrange = (0,2000) ):
    """
    Per pixel scurve fit from saved data and save the result in an npy file
    """
    #Load the scurve data
    if type(fname) == type(None):
        fname = get_data_fname()
                

    pathname = os.path.join( cfg.path.data, fname)
    with np.load( pathname ) as f:
        data = f['data']
        x = f['x']
    
    #if a mask was supplied take out the masked pixels
    if type(mask) != type(None):
        for i in range( data.shape[2] ):
            data[:,:,i][mask] = 0
     
    data = data.astype(np.double)
    for i in range(data.shape[2]):
        data[:,:,i] /= data[:,:,-1]
    data *= 1000
    
    #Normalize y then find initial prameters and fit
#    y = data.sum(axis = 0).sum(axis = 0)
#    y /=  ( data.sum(axis = 2)>0 ).sum()
#    par = find_initial_parameters(x,y)
    par = np.array([ 0,   0,   1.11495212e+03,
         1.98609468e+02,   5.94207866e+02,   4.47860380e-01])

    fit_result = mpfit.fit(data, x, cfg.calibration.nproc, par)  
    
#    #If specified plot and save the result
#    if cfg.calibration.plot:
#        mean, std, lines = plot.chip_histograms( fit_result['mu'], xmin = thrange[0], xmax = thrange[1] ) 
#        plt.xlabel('Vcmp [DAC LSB]')
#        plt.ylabel('Number of Pixels')
#        plt.savefig( os.path.join( cfg.path.data, get_fit_fname().strip('.npy') ) )
#    
#    
#    #Save the fit result
#    fname = get_fit_fname().strip('.npy')
#    pathname = os.path.join(cfg.path.data, fname)
#    np.save(pathname, fit_result)
    
    return fit_result


def _trimbit_scan(detector, step = 2):
    """
    Internal function to scan the trimbits and save the data
    can also be used when the detector is not directly controlled
    from the machine processing the data as in the case with the 9M
    """


    detector.exposure_time = cfg.calibration.exptime

    tb_array = np.arange(0, 64, step)
    print(tb_array)
    
    _s = detector.image_size
    data = np.zeros((_s.rows, _s.cols, tb_array.size))
    
    with setup_measurement(detector) as receiver:
        for i,v in enumerate(tb_array):
            detector.trimbits = v
            print(detector.trimbits)
            detector.acq()
            data[:,:,i] = receiver.get_frame()
    

    return data, tb_array

def _plot_trimbit_scan(data,x):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,7))
    ax1.plot(x, data.sum(axis = 0).sum(axis = 0))
    for p in u.random_pixel(n_pixels = 50, rows = (0, data.shape[0],), cols = (0, data.shape[1])):
        ax2.plot(x, data[p])
        
    ax1.set_xlabel('Trimbit [1]')
    ax1.set_ylabel('Counts [1]')
    ax1.set_title('Sum')
    ax2.set_xlabel('Trimbit [1]')
    ax2.set_ylabel('Counts [1]')
    ax2.set_title('Sample pixels')
    fig.suptitle('Trimbit scan')
    fig.tight_layout()
    return fig, ax1, ax2
    
def do_trimbit_scan(detector, xraybox, step = 2, data_mask = None):
    """
    Setup the detector and then scan trough the trimbits. Normally with 
    step of 2
    performa a trimbit scan 
    
    Examples
    ---------
    
    ::
        
        fit_result = calibration.do_trimbit_scan(detector, xraybox)
        
        
    .. image:: _static/tb_scan.png  
    
    
    """
    #Load data
    data_fname = get_data_fname()
    data_pathname = os.path.join( cfg.path.data, data_fname)
    with np.load( data_pathname) as f:
        data = f['data']
        x = f['x']
        
    #Load scurve fit
    fit_fname = get_fit_fname()
    fit_pathname = os.path.join( cfg.path.data, fit_fname)
    fit_result = np.load( fit_pathname )
    
    
    #Find the mean values for both module and half module
    find_mean_and_set_vcmp(detector, fit_result)
           
#    _take_trimbit_data(detector, xraybox, step = step, data_mask = data_mask)
    with xrf_shutter_open(xraybox, cfg.calibration.target):
        data, x = _trimbit_scan(detector)
        np.savez(os.path.join(cfg.path.data, get_tbdata_fname()), 
                 data = data, x = x)
    
    
    if cfg.calibration.plot is True:
        fig, ax1, ax2 = _plot_trimbit_scan(data, x)
        fig.savefig( os.path.join( cfg.path.data, get_tbdata_fname().strip('.npz') ) )
    
    return data, x


 

def load_trimbits(detector):
    """
    Load trimbits for the current calibration settings. Defined in 
    config.py 
    
    Examples
    ----------
    
    ::
    
        calibration.load_trimbits(d)
        >> Settings file loaded: /mnt/disk1/calibration/T63/gain5/T63_CuXRF_gain5.sn058
        >> Settings file loaded: /mnt/disk1/calibration/T63/gain5/T63_CuXRF_gain5.sn059
    
    """
    fname = get_trimbit_fname()
    pathname = os.path.join(cfg.path.data, fname)
    detector.load_trimbits(pathname)

def find_and_write_trimbits_scaled(fname = None, tb_fname = None, tau = None):
    
    #Filename for scurve
    if fname is None:
        fname = get_data_fname()
        
    #filename for tb data
    if tb_fname is None:
        tb_fname = get_tbdata_fname()
        
    #Load scurve data and calculate scale
    pathname = os.path.join( cfg.path.data, fname)
    tb_pathname = os.path.join( cfg.path.data, tb_fname)
    with np.load( pathname ) as f:
        scale =f['data'][:,:,-1]
        scale = scale / 1000.
    

    
    #Load trimbit scan    
    with np.load( tb_pathname ) as f:
        data = f['data']
        x = f['x']

    #Scale data
    data = data.astype( np.double )
    for i in range(data.shape[2]):
        data[:,:,i] /= scale
    
    #Load the fit result from the vcmp scan 
    fit_result = np.load( os.path.join(cfg.path.data, get_fit_fname()) )
    
    #Find the number of counts at the inflection point
    target = function.scurve( fit_result['mu'], 
                fit_result['p0'],
                fit_result['p1'],
                fit_result['mu'],
                fit_result['sigma'], 
                fit_result['A'],
                fit_result['C'])
                

    par = np.array([9.35, 2.08, 33.5 , 14.47, 507, 4])
         
    result = mpfit.find_trimbits(data, x, target, cfg.calibration.nproc, par) 
    tb = result['trimbits']
    tb[tb>63] = 63
    tb[tb<0] = 0
    tb = tb.round()
#    c,h = r.hist(tb, xmin = -.5, xmax = 63.5, bins = 64)
    tb = tb.astype(np.int32)
    ax, im = plot.imshow(tb)
#    plt.savefig( os.path.join( cfg.path.data, get_tbdata_fname().strip('.npz') + '_image' ) )
    
    return tb, target, data,x, result
    

    
def find_and_write_trimbits(detector, tau = None):
    """
        Examples
    ---------
    
    ::
        
        fit_result = calibration.find_and_write_trimbits(decector)
        
        
    .. image:: _static/trimbit_map.png  
    
    
    """
    #Get the correct filenames depending on configuration settings
    fit_fname = get_fit_fname()
    data_fname = get_tbdata_fname()
    data_pathname = os.path.join( cfg.path.data, data_fname)
    fit_pathname = os.path.join( cfg.path.data, fit_fname)
    
    #Load trimbit scan    
    with np.load( data_pathname ) as f:
        data = f['data']
        x = f['x']
    
    #Load the fit result from the vcmp scan 
    fit_result = np.load( fit_pathname )
    
    #Find the number of counts at the inflection point
    target = function.scurve( fit_result['mu'], 
                fit_result['p0'],
                fit_result['p1'],
                fit_result['mu'],
                fit_result['sigma'], 
                fit_result['A'],
                fit_result['C'])
                
    #Magic numers for the initial parameters
    #TODO! Use a global fit for these
#    par = np.array([  8.65703068e+00,   1.46117014e+00,   20,
#         1.30881741e+01,   4.37825173e+03,   0.00000000e+00])

    par = np.array([  60,   1.46117014e+00,   20,
             1.30881741e+01,   4.37825173e+03,   0.00000000e+00])

         
    result = mpfit.find_trimbits(data, x, target, cfg.calibration.nproc, par) 
    tb = result['trimbits']
    tb[tb>63] = 63
    tb[tb<0] = 0
    tb = tb.round()
    c,h = r.hist(tb, xmin = -.5, xmax = 63.5, bins = 64)
    c.Draw()
    tb = tb.astype(np.int32)
    ax, im = plot.imshow(tb)
    plt.savefig( os.path.join( cfg.path.data, get_tbdata_fname().strip('.npz') + '_image' ) )
    
    
    #Save trimbits in np file
    
    dacs = detector.dacs.get_asarray()

    #Add tau for the new style trimbit files, to be default
    if type(tau) != type(None):
        tmp = np.zeros((1, dacs.shape[1]))
        tmp[:] = tau
        dacs = np.vstack( ( dacs, tmp) )
        
    fname = get_trimbit_fname()
    pathname = os.path.join(cfg.path.data, fname)
    

    np.savez( pathname, trimbits = tb, fit = result)
    
    hostname = detector.hostname
    halfmodule = get_halfmodule_mask()

        
    for i in range( len(halfmodule) ):
        fname = pathname + '.sn' + hostname[i][3:]
        print(fname)
        io.write_trimbit_file(fname, tb[halfmodule[i]], dacs[:, i])



def rewrite_calibration_files(detector, tau = None, hostname = None):
    """
    Rewrite the calibration files using the center of the trimmed distributions
    saved with suffix _rw, needs a connected detector
    
    
    Specifying hostnames will use offline files only
    TODO! Tau should check if a tau is already present
    """
    print('det', detector)
    #Load scurve fit
    fit_fname = get_fit_fname()
    fit_pathname = os.path.join( cfg.path.data, fit_fname)
    fit_result = np.load( fit_pathname ) 
 
    #We have an detector online and should use it
    if type(hostname) == type(None):
        find_mean_and_set_vcmp(detector, fit_result)
        hostname= detector.get_hostname()
        dacs = detector.dacs.get_asarray()
        
    elif type(hostname) == type( [] ):
        dacs = np.zeros((len( hostname ), cfg.ndacs + 2 ))
        print( dacs.shape )
        vcmp, vcp = find_mean_and_set_vcmp(None, fit_result)
#        return vcmp, vcp
        dacobj = Dacs.Dacs( None )

    else:
        raise TypeError('Type of hostname needs to be None for online or a list of hostnames for offline')
        
    #Adding tau, todo fix if tau is already there
    if type(tau) != type(None):
        tmp = np.zeros((dacs.shape[0], 1))
        tmp[:] = tau
        dacs = np.hstack( ( dacs, tmp) )

        
    #Load trimbits 
    fname = get_trimbit_fname()
    pathname = os.path.join(cfg.path.data, fname)

    
    if cfg.geometry == '2M':
        #ESRF 2M detector
        halfmodule = mask.eiger2M.halfmodule
        vcmp_name = mask.eiger2M.vcmp
        
    elif cfg.geometry == '500k':
        halfmodule = mask.halfmodule
    else:
        raise NotImplementedError("Add support for other modules")
        
    for i in range( len( halfmodule ) ):
        fname = pathname + '.sn' + hostname[i][3:]
        print(fname)
        tb, tmpdacs = io.io.read_trimbit_file( fname )
        if type(detector) == type(None):
            dacs[i] = tmpdacs
            
            for j in range( cfg.nchips_per_halfmodule ):
                idx =i*cfg.nchips_per_halfmodule + j
                vcmp_idx = dacobj.get_index( vcmp_name[idx][2:] )
                dacs[i, vcmp_idx] = vcmp.flat[ i*cfg.nchips_per_halfmodule + j ]
                
            vcp_idx = dacobj.get_index('vcp')
            dacs[i, vcp_idx] = vcp.flat[i]
            print( tmpdacs[vcp_idx], dacs[i, vcp_idx], vcp.flat[i] )
                
        
        #Write with correct trimbits
        fname = pathname + '_rw.sn' + hostname[i][3:]
        print(fname)
#        io.write_trimbit_file(fname, tb, dacs[i])




def find_mean_and_std( fit_result ):
    """
    Find mean and standard deviation for each chip in the detector
    """
    mean = []
    std = []
    for m in mask.detector[ cfg.geometry ].module:
        for c in mask.chip:
            mean.append( fit_result['mu'][m][c].mean() )
            std.append( fit_result['mu'][m][c].std() )
            
    return mean, std

def generate_mask():
    """
    Generate mask of bad pixels using calibration data
    TODO! verify with multi module systems
    """
    
    #Load data
    fit_result = np.load( os.path.join( cfg.path.data, get_fit_fname()) )
    mean, std = find_mean_and_std( fit_result )
    
    #Mask
    bad_pixels = np.zeros( (mask.detector[ cfg.geometry ].nrow,
                            mask.detector[ cfg.geometry ].ncol),
                            dtype = np.bool)
    
    i = 0
    for m in mask.detector[ cfg.geometry ].module:
        for c in mask.chip:
            tmp = fit_result['mu'][m][c]
            tmp= (tmp < mean[i]-cfg.calibration.std*std[i]) | (tmp > mean[i]+cfg.calibration.std*std[i]) 
            bad_pixels[m][c][tmp] = 1
            i+= 1
            
    print( 'Masking {:d} pixels using {:d} std'.format(bad_pixels.sum(), cfg.calibration.std) )
    return bad_pixels

    
def take_global_calibration_data(detector, xraybox, start = 0, stop = 2001, step = 40):
    
    
    vcmp_array = np.arange(start, stop, step)
    print(vcmp_array)
    
    _s = detector.image_size
    data = np.zeros((_s.rows, _s.cols, vcmp_array.size))
    detector.exposure_time = cfg.calibration.vrf_scan_exptime

    for t in cfg.calibration.global_targets[ cfg.calibration.target ]:
        print( t )
        with xrf_shutter_open(xraybox, t):
            
            with setup_measurement(detector) as receiver:
                for i,v in enumerate(vcmp_array):
                    detector.vthreshold = v
                    detector.acq()
                    data[:,:,i] = receiver.get_frame()

        fname = '{:s}_vcmp_{:s}XRF_{:d}'.format( cfg.det_id, t, cfg.calibration.run_id)
        pathname = os.path.join(cfg.path.data, fname)
        np.savez(pathname, data = data, x = vcmp_array)
        
    #Normally only one point    
    return data, vcmp_array    

        
def per_chip_global_calibration( swap_axes = False ):
    #Find which calibration points we have

    plotfit = True
    target = []
    energy = []
    for f in os.listdir( cfg.path.data ):
        if f.startswith(cfg.det_id + '_vcmp_') and f.endswith(str( cfg.calibration.run_id )+'.npz'):
            print( f )
            t = f.split('_')[2][0:2]
            target.append( t )
            energy.append( xrf[t] )
    
    xx = np.linspace(0,2000)
    
    mu = np.zeros((8,len(target)))

    
    p0 = np.zeros(8)
    p1 = np.zeros(8)

    """
    scruve function used for energy calibration
    scurve(x, p0,p1, mu, sigma, A, C)
    
    [0] - p0
    [1] - p1
    [2] - mu
    [3] - sigma
    [4] - A
    [5] - C
    """
    print( target )

    
    for i,t in enumerate(target):
        
        #Custom limits
        if t == 'Ag':
            lim_high_fit = 1800
        else:
            lim_high_fit = 2400
            
        #Load data
        fname = '{:s}_vcmp_{:s}XRF_{:d}.npz'.format(cfg.det_id, t, cfg.calibration.run_id)
        pathname = os.path.join( cfg.path.data, fname )
        with np.load( pathname ) as f:
            data = f['data']
            x = f['x']
            
        if swap_axes:
            data = np.swapaxes(data, 1, 2 )
            data = np.swapaxes(data, 0, 1 )
            print( data.shape )
            
        #Clean up data
        for k in range( data.shape[2] ):
            m = data[:,:,k].mean()
            if m < cfg.calibration.clean_threshold:
                m  = cfg.calibration.clean_threshold
                
            data[:,:,k][data[:,:,k]> 3* m] = 0            
            
            
        #Fit scurve for each chip and save mu parameter
        plt.figure( figsize = (14,7))
        colors = sns.color_palette(n_colors = 8)
        
        print( 'target:', t)
        for j in range(8):
            y = np.zeros( x.size )
            for k in range(x.size):
                y[k] = data[:,:,k][mask.chip[j]].sum()
            
            print( "[dbg] 1" )
            #Linear backgroud
            c,g = r.plot(x[x<100], y[x<100], draw=False) 
            fit = g.Fit('pol1', 'NQS')
            p0_estimate = fit.Get().Parameter(0)
            p1_estimate = fit.Get().Parameter(1)
            
            #Graph and fit function
            c,h = r.plot(x,y)
            func = TF1('scurve', function.root.scurve, 0, lim_high_fit)


            print( "[dbg] 2" )
            
            #Background from a linear fit in the 0-100 region
            func.SetParameter(0, p0_estimate) 
            func.SetParLimits(0, p0_estimate*.5, p0_estimate*2)
            func.SetParameter(1, p1_estimate)
            func.SetParLimits(1, 0, p1_estimate*2)
    
            #mu must be in the range that we are using
            func.SetParLimits(2,0,2500)
            func.SetParameter(2, 1000 ) 
    
            #Give this a reasonable value       
            func.SetParameter(3, 100)
            func.SetParLimits(3, 0, 300)         
    
            #TODO! Find better estimates
            func.SetParLimits(4, 0, 1e10)
            func.SetParLimits(5, 0, 1e7)
    
          
            fit = h.Fit('scurve', 'SQR')        
            fit = h.Fit('scurve', 'SQR')     

            par = [fit.Get().Parameter(k) for k in range(6)]
            y_fit = function.scurve(x, *par)
            plt.subplot(1,2,1)
            plt.plot(x,y, 'o', color = colors[j])
            plt.plot(x, y_fit, color = colors[j])
            
            plt.subplot(1,2,2)
            plt.plot(x, np.gradient(y),'o', color = colors[j])
            plt.plot(x, np.gradient(y_fit), color = colors[j])
            
            
            mu[j,i] = fit.Get().Parameter(2)



    #Fit linear calibration for each chip
    colors =sns.diverging_palette(145, 280, s=85, l=25, n=9)
    colors.pop(4) #Remove central gray
    for j in range(8):
        c,h = r.plot(mu[j], energy, options = 'AP')
        fit = h.Fit('pol1', 'SQ')
        p0[j] = fit.Get().Parameter(0)
        p1[j] = fit.Get().Parameter(1)
    
        plt.figure('calibration')
        plt.plot(mu[j], energy, 'o', color = colors[j], label = 'Chip: {:d}'.format(j))
        plt.plot(xx, function.pol1(xx, p0[j],p1[j]), color = colors[j])
 
    plt.legend()  
    e_low =  p0+p1*1900 
    e_high =  p0+p1*100 
        
    for i in range(8):
        print(e_low[i], e_high[i])
    print(e_low.max(), e_high.min())
    print(e_low.std(), e_high.std())
    calibration_fname = os.path.join( cfg.path.data, 'calibration' )
    np.savez(calibration_fname, p0 = p0, p1 = p1, mu = mu, energy = energy)
    
    return p0, p1
    
def generate_calibration_report( thrange = (100,1900) ):
    """
    Generate a calibration report for the current gain. 
    
    .. warning::
        Follows the old style of calibration and should be updated. Still
        gives a valuble insigt of the calibration for this gain though.
    
    """
    
    os.chdir( cfg.path.data)

    #Load trimbit data
    fname = get_trimbit_fname()
    tb_top = io.read_trimbit_file( fname + '.sn' +cfg.top[3:] )    
    tb_bottom = io.read_trimbit_file( fname + '.sn' + cfg.bottom[3:] )  
    vrf = [tb_top[1][2], tb_bottom[1][2]]
    
    #Create output file
    fname = '{:s}_{:s}_calibration_report.pdf'.format(cfg.det_id, cfg.calibration.gain)    
    pdf = PdfPages( fname )
    
    #Plotting
    plt.ion()
    sns.set_context('talk', font_scale = 1)
    colors = sns.color_palette(n_colors = 8)
    
    plt.figure(num=None, figsize=(11.69, 8.27), dpi=100)
    gs = mpl.gridspec.GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1,1]
                       )   

    
    target = []
    energy = []
    for f in os.listdir( cfg.path.data ):
        if f.startswith(cfg.det_id + '_vcmp_') and f.endswith(str( cfg.calibration.run_id_trimmed )+'.npz'):
            print( f )
            t = f.split('_')[2][0:2]
            target.append( t )
            energy.append( xrf[t] )
    
    #Load global calibration and plot  
    with np.load('calibration.npz') as f:
        mu = f['mu']
        energy = f['energy']
        p0 = f['p0']
        p1 = f['p1']
        
        xx = np.linspace(mu.min()*0.8, mu.max()*1.2, 100)
        
        ax = plt.subplot(gs[0,0])
        for i in range(8):
            plt.plot(mu[i], energy, 'o', color = colors[i], label = 'Chip '+str(i))
            plt.plot(xx, function.pol1(xx, p0[i],p1[i]), color = colors[i])
        plt.legend( fontsize = 8 )
        plt.xlabel('Vcmp')
        plt.ylabel('Energy [keV]')

    hpos = 1.20
    pos = 1
    step = -0.09
    plt.text(hpos, pos,'Eiger Calibration - '+ cfg.det_id + ' \"' +cfg.calibration.gain + '\"',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'bold')
                 
                 
    e_low = np.max( p0+p1*thrange[1] )
    e_high = np.min( p0+p1*thrange[0] )
    pos -= 0.1
    plt.text(hpos, pos,'Min th: '+ str( np.round( e_low, 2) ) + ' keV',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'normal')
    pos -= 0.09
    plt.text(hpos, pos,'Max th: '+ str( np.round( e_high, 2) ) + ' keV',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'normal')
    pos -= 0.09
    plt.text(hpos, pos,'Trim E: '+ str( xrf[ cfg.calibration.target ] ) + ' keV',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'normal')
                 
    for i in range(8):
        text  = 'Chip ' + str(i) + ': '  + \
                'p0: %6.2f ' % (p0[i]) + \
                'p1: %05.2e ' % (p1[i])  
        print( text )
    
    pos -= 0.09
    for i in range(8):
        pos -= 0.05
        text  = 'Chip ' + str(i) + ': '  + \
                'p0: %6.2f ' % (p0[i]) + \
                'p1: %05.2e ' % (p1[i])  
        plt.text(hpos, pos,text,
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'normal',
                 fontsize = 8,
                 family = 'monospace')
    pos -= 0.10

    text = 'vrf = [{:d}, {:d}]'.format(vrf[0], vrf[1])
    plt.text(hpos, pos,text,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax.transAxes,
         weight = 'normal',
         fontsize = 8,
         family = 'monospace')
    #Plot threshold dispersion before and after calibration
    bins = 300
    scale = 3.0
    bin_size = np.round(scale/bins*1000, 1)
    l = ['Untrimmed', 'Trimmed']
    xlim = (xrf[ cfg.calibration.target ]-scale/2, xrf[ cfg.calibration.target ]+scale/2)
    
    for k,j in enumerate( [cfg.calibration.run_id_untrimmed, cfg.calibration.run_id_trimmed] ):
        ax = plt.subplot(gs[1,0])
        fname = get_fit_fname( run_id = j )
        data = np.load( fname )
    
        f = open('vcmp', 'w')
        for i in range(8):
            #Trimmed distributions in DAC
            if j == cfg.calibration.run_id_trimmed:
                c,h = r.hist(data['mu'][mask.chip[i]], xmin = 0, xmax = 2000, bins = bins)
                x,y = r.getHist(h, edge = 'high')                      
                label = '%d: $\mu$:%d $\sigma$:%.2f '%(i, h.GetMean(), h.GetStdDev())
                plt.plot(x,y, ls = 'steps', label = label, color = colors[i])
                f.write('{:.0f}\n'.format( h.GetMean()) )

        f.close()
                
          
        #Vcmp plot notation
        plt.legend( fontsize = 8, loc = 'upper left')
        plt.xlabel('Vcmp')
        
        
        #Empty histogram
        c,h = r.hist([-1], xmin = xlim[0], xmax = xlim[1], bins = bins)
    
        #Fill with pixel data
        for i in range(8):
            tmp = data['mu'][mask.chip[i]]*p1[i]+p0[i]
            for pixel in tmp.flat:
                h.Fill(pixel)
                
        #Plot energy data
        x,y = r.getHist(h, edge = 'high')
        mean = str( np.round( h.GetMean(), 1 ) )
        sigma = str( np.round(h.GetStdDev()*1000, 1) )
        label = l[k]+'\n$\mu$: ' + mean + 'keV\n' + r'$\sigma$: ' + sigma +' eV'
        
        ax = plt.subplot(gs[1,1])
        plt.fill_between(x,y, step = 'pre', label = label, color = colors[k])
        plt.xlim( xlim )
        plt.legend()
        plt.xlabel('Energy [keV]')

    pdf.savefig()
    
    plt.figure(num=None, figsize=(11.69, 8.27), dpi=100)
    gs = mpl.gridspec.GridSpec(2, 6,
                       width_ratios=[1,1,1,1,1,1],
                       height_ratios=[1,1]
                       )   
    
    fname = '{:s}_vcmp_{:s}XRF_{:d}.npz'.format( cfg.det_id, cfg.calibration.target, cfg.calibration.run_id )
    with np.load( fname ) as f:
        data = f['data']
        x = f['x']      
    
    ax1 = plt.subplot(gs[0:3])
    ax2 = plt.subplot(gs[3:6])

    
    #Clean up data
    for i in range( data.shape[2] ):
        data[:,:,i][data[:,:,i]> 3* data[:,:,i].mean()] = 0
    

    
    for i in range(8):
        chip_slice = mask.chip[i] + [slice( 0, len(data), 1 )]
        y = data[chip_slice].sum(axis = 0).sum(axis = 0)
        xx = p0[i]+p1[i]*x
        ax1.plot(xx,y, label = 'Chip: {:d}'.format(i))
        
        y_gradient = np.gradient(y)
        y_gradient /= y_gradient.max()
        ax2.plot(xx, y_gradient, label = 'Chip: {:d}'.format(i))
        ax2.set_ylim(0,1.1)

    ax1.legend( loc = 'upper right' )
    ax1.set_xlabel( 'Energy [keV]' )
    ax1.set_ylabel(' Counts [1] ')
    
    ax2.legend( loc = 'upper right' )
    ax2.set_xlabel( 'Energy [keV]' )
    ax2.set_ylabel(' Counts [a.u.] ')


    fname = get_trimbit_fname() + '.npz'    
    with np.load(fname) as f:
        image = f['trimbits']

    ax = plt.subplot(gs[6:10])
    im = ax.imshow( image, cmap = 'coolwarm', origin = 'lower', interpolation = 'nearest' )
    im.set_clim(0,63)
    ax.set_xlabel('Trimbit map')
    plt.grid(False)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cb = plt.colorbar(mappable = im, cax = cax)
    
    ax =  plt.subplot(gs[10:12])
    c,h = r.hist(image, xmin = -.5, xmax = 63.5, bins = 64)
    x0,y0 = r.getHist(h)
    ax.plot(x0,y0/y0.max(), ls = 'steps')
    ax.fill_between(x0, 0, y0/y0.max(), step = 'pre')
    ax.set_xlim(0,64)
    ax.set_xlabel('Trimbit [1]')
    ax.set_ylabel('Pixels [a.u.]')  
    ax.set_ylim( 0, 1.1 )
    plt.tight_layout()
    pdf.savefig()
    pdf.close()
    
def write_calibration_files(afs = False, loc = False, afs_overwrite = False):
    """
    Write Eiger calibration files and copy to the settingsdir
    """
    source = cfg.path.data
    destination = os.path.join( cfg.path.settingsdir, cfg.calibration.gain ) 
    calibration = source + '/calibration.npz'
    

    #-----------------------------------------------------------Trimbits

    print( '*** Processing trimbit files ***')
    #find trimbit files
    top_trim = os.path.join( source, '{:s}_{:s}_{:s}.sn{:s}'.format(cfg.det_id, cfg.calibration.target, cfg.calibration.gain, cfg.top[3:]) )
    bottom_trim = os.path.join( source, '{:s}_{:s}_{:s}.sn{:s}'.format(cfg.det_id, cfg.calibration.target, cfg.calibration.gain, cfg.bottom[3:]) )
    
    if os.path.isfile( top_trim ):
        print( 'Top trimbit file found')
    else:
        raise IOError('top trimfile not found')
        
    if os.path.isfile( bottom_trim ):
        print( 'Bottom trimfile found')
    else:
        raise IOError('bottom trimfile not found')   
        
    #check output directory
    if os.path.isdir( destination ):
        print( 'settingsdir found')
    else:
        raise IOError('settingsdir not found')
        

        
    #copy trimbit files
    fname = destination + '/noise.sn{:s}'.format( cfg.top[3:] )
    print( 'Writing:', fname)
    shutil.copy( top_trim, fname)
    fname = destination + '/noise.sn{:s}'.format( cfg.bottom[3:] )
    print( 'Writing:', fname)
    shutil.copy( bottom_trim, fname)
    
    if afs:
        afs_path = os.path.join( cfg.path.afs_calibration, '{:s}-{:s}'.format(cfg.det_id, loc), cfg.calibration.gain )
        print( '\nSaving to AFS: ', afs_path)
        if os.path.isdir( afs_path ):
            pass
        else:
            os.makedirs( afs_path )
        dest_fname = afs_path + '/noise.sn{:s}'.format(  cfg.top[3:] )
        if os.path.isfile(dest_fname):
            if not afs_overwrite:
                raise IOError('File already exists')
        print( 'Writing:', dest_fname)
        shutil.copy( top_trim, dest_fname)
        
        dest_fname = afs_path + '/noise.sn{:s}'.format(  cfg.bottom[3:] )
        if os.path.isfile(dest_fname):
            if not afs_overwrite:
                raise IOError('File already exists')
        print( 'Writing:', dest_fname)
        shutil.copy( top_trim, dest_fname)
        
    #------------------------------------------------------------Calibration
    #find calibration files
    print( '\n*** Processing calibration ***')
    if os.path.isfile( calibration ):
        print( 'calibration file found')
    else:
        raise IOError('calibration file not found')
    
    #Write text files
    with np.load( calibration ) as f:
        p0 = f['p0']
        p1 = f['p1']
        
    k = np.abs( 1/p1 )
    m = np.abs( p0/p1 )   
    
    #Top
    fname = 'calibration.sn{:s}'.format(cfg.top[3:])
    pathname = os.path.join( destination, fname )
    print( 'Writing:', pathname)
    with open( pathname , 'w') as f:
        for i in range(4):
            line = '%.2f %.2f\n' % (m[i], k[i])
            f.write( line )
    if afs:
        afs_pathname = os.path.join( afs_path, fname )
        print( 'Writing:', afs_pathname)
        shutil.copy( pathname, afs_pathname )
    #Bottom
    fname = 'calibration.sn{:s}'.format(cfg.bottom[3:])
    pathname = os.path.join(destination, fname)
    print( 'Writing:', pathname)
    with open( pathname , 'w') as f:   
        for i in range(7,3,-1):
            line = '%.2f %.2f\n' % (m[i], k[i])
            f.write( line )
            
    if afs:
        afs_pathname = os.path.join( afs_path, fname )
        print( 'Writing:', afs_pathname)
        shutil.copy( pathname, afs_pathname )
        
    if afs:
        print( '\n*** Copying calibration report ***')
        #T38_veryhighgain_calibration_report.pdf
        fname = '{:s}_{:s}_calibration_report.pdf'.format(cfg.det_id, cfg.calibration.gain)
        pathname = os.path.join( source, fname ) 
        afs_pathname = os.path.join( os.path.split(afs_path)[0], fname)
        print( 'Writing:', afs_pathname)
        shutil.copy( pathname, afs_pathname )