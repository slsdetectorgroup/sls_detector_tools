#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:37:05 2017

@author: l_frojdh
"""
import logging
import os
import sys

verbose = True
debug = False

#Detector
det_id = 'Unconfigured'
geometry = '500k'


class path: 
    """
    All configuration regarding which path to write to or where to find
    the slsDetectorsPackage. 
    """
    
    base = '/home/l_msdetect/erik/'
    """Base directory for io"""


    
    test = '/home/l_frojdh/code/scripts/testing'
    """Output directory for module testing"""
    
#    settingsdir = '/home/l_frojdh/slsDetectorsPackage/settingsdir/eiger/'
#    """ Settings directory of the slsDetectorsPackage"""
#    
#    afs_calibration = '/afs/psi.ch/project/pilatusXFS/Erik/calibration'  
#    """ AFS path to write out trimbit files """
    
#    trim = os.path.join(base, 'trim')
    data = os.path.join(base, 'data')
    out = os.path.join(base, 'out')  
#    log = os.path.join(base, 'tmp')


class tests:
    plot = True
    """Plot the result of each module test"""
    
    rxb_interval= {'Half Speed': [(786.04, 30.32),(1356.17, 32.95)],
                   'Full Speed': [(921.58, 45.21),(1327.44, 33.13)]}
    iodelay_interval= {'Full Speed': [(635.49, 5.57),(694.48, 19.11)],
                   'Half Speed': [(635.62, 5.74),(754.32, 15.11)]}

#Configure logging
def set_log(fname, level = logging.INFO, stream = False):
    logger = logging.getLogger()
    logger.setLevel( level )
    formatter = logging.Formatter( '%(asctime)s - %(levelname)s- %(module)s:%(funcName)s() %(message)s ' )
    if fname:
        append = os.path.isfile( os.path.join(path.log, fname) )
        if not os.path.isdir(path.log):
            os.makedirs(path.log)
        handler = logging.FileHandler( os.path.join(path.log, fname) )
        formatter = logging.Formatter( '%(asctime)s - %(levelname)s- %(module)s:%(funcName)s() %(message)s ' )
        blank_formatter = logging.Formatter( '\n%(asctime)s - %(levelname)s- %(module)s:%(funcName)s() %(message)s ' )
        handler.setFormatter( formatter )
        logger.addHandler( handler )
    else:
        append = False


    if append:
        handler.setFormatter( blank_formatter )
        logger.info('New measurement')
        handler.setFormatter( formatter )
        print('Appendling to existing log file')
    
    #In addition output log information to stream
    if stream:
        stream_handler = logging.StreamHandler( sys.stdout )
        stream_handler.setFormatter( formatter )
        logger.addHandler( stream_handler )
    
    
class calibration():
    """
    Settings concerning the calibration
    """
    fname = 'run'    
    target = False
    gain = False
    exptime = 3
    vrf_scan_exptime = 0.1 #Short time!
    run_id = 0
    period = 0
    nframes = 1
    speed = 'Quarter Speed'
    dynamic_range = 32
    nproc = 12

    tp_dynamic_range = 16
    tp_exptime = 0.01
    
    #DACs
    vtr = 2500    
    vrs = 1600
    threshold = 1200
    
    #Trimbit setting for the first scan, trimming will be centered around this
    trimval = 32
    #Switch between XRF and testpulse calibration
    """
    Possible calibration types:
    XRF = X-ray fluorescence 
    beam  = Monochromatic (synchrotron)
    TP  = Test pulses
    """
    type = 'XRF'
    energy =  None
    plot = True
    clean_threshold = 500
    
    flags = ['nonparallel', 'continous']
    
    global_targets = {  'Sn': ['Ag'],
                         'Ag': ['Sn'],
                        'Mo': ['Ag'],
                        'Zr': ['Mo'],
                        'Ge': ['Cu'],
                        'Cu': ['Ge'],
                       'Fe': ['Cu'],
                        'Cr': ['Fe'],
                        'Ti': ['Cr']}
    
    run_id_untrimmed = 0
    run_id_trimmed = 1
    
    #Mask all pixels further than set number of std from the chip mean
    std = 5
    
    #Number of parameters in the calibration function 
    npar = 6 
    
class Eiger2M():
    """Backend board numbers and hostanmes for the ESRF Eiger2M"""
    beb = [74,71,64,102,72,73,87,88]
    hostname = ['beb{:03d}'.format(b) for b in beb]

class Eiger9M():
    """Backend board numbers, hostnames and T numbers for the 9M"""
    beb =  [111,70,
         109,116,
         92,91,
         122,113,
         105,121,
         104,54,
         117,125,
         103,100,
         127,106,
         84,61,
         78,76,
         101,56,
         97,96,
         30,38,
         40,29,
         59,58,
         95,94,
         119,55]
    
    beb = [  111, #0
              70, #1
             109, #2
             116, #3
             97,  #4
             96,  #5
             92, #6
             91, #7
             105, #8
             121, #9
             104, #10
             54,  #11
             53, #12
             4, #13
             103, #14
             100, #15
             127, #16
             106, #17
             125,  #18
             61,  #19
             78,  #20
             17,  #21
             101, #22
             113,  #23
             58,  #24
             59,  #25
             30,  #26
             38,  #27
             40,  #28
             29,  #29
             112,  #30
             41,  #31
             95,  #32
             94,  #33
             119, #34
             55]  #35
    
    hostname = ['beb{:03d}'.format(b) for b in beb]

    T = [ 50,   77,      2,
           6,   58,     60,
          33,   64,     78,
          30,   52,     73,
          79,   62,     56,
          69,   42,     59,]