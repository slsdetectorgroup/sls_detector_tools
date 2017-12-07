# -*- coding: utf-8 -*-
"""
Module to control the big xray box at PSI. Communicates using the command line
to the */afs/psi.ch/project/sls_det_software/bin/xrayClient64*. Provides a 
DummyBox that only writes the commands to the log file without doing anything.

"""
from __future__ import print_function
from . import config as cfg
#from subprocess import Popen, PIPE, run
import subprocess
import logging
logger = logging.getLogger()
from functools import partial
import re

from contextlib import contextmanager

@contextmanager
def xrf_shutter_open(box, target):
    box.target = target
    box.open_shutter('XRF')
    print('open')
    yield
    box.close_shutter('XRF')
    print('close')
    
    
    
class DummyBox:
    """
    Dummy version of the XrayBox. Can be used for testing or when using 
    an X-ray source that is not control but you don't want to modify 
    you calibration scripts
    """
    def __init__(self):
        self.kV = 0
        self.mA = 0
        logger.info('Xray box initialized')
        
    
    def target(self, t):
        """
        Write target name to logfile
        """
        logger.info('Switching to %s target', t)

    def shutter(self, s):
        """
        Write opening shutter to logfile
        """
        logger.info('Opening shutter')

    def unlock(self):
        """
        Write unlock to the logfile
        """
        logger.info('Unlocking the dummy Xraybox from other users')
        
    def HV(self, value):
        """
        Emulating High Voltage on and off
        """
        if value is True:
            logger.info('Switching on HV')

        else:
            logger.info('Switching off HV')

    def set_kV(self, kV):
        self.kV = kV
        logger.info('Setting HV to {:.2f} kV'.format(kV))
        
    def get_kV(self):
        logger.info('Voltage is {:.2f} kV'.format(self.kV))
        return self.kV
        
    def set_mA(self, mA):
        self.mA = mA
        logger.info('Setting HV to {:.2f} mA'.format(mA))
    def get_mA(self):
        logger.info('Tube current is {:.2f} mA'.format(mA))
        return self.mA
        
class XrayBox():
    """
    Wrapper around the xrayClient64, uses the executable from afs, make sure
    that you have access. Supports logging of commands using the python logger.
    
    
    Examples
    -----------
    
    ::
        
       from sls_detector_tools import XrayBox
       box = XrayBox()
       box.unlock()
       
       box.current = 40
       box.voltage = 30
       box.target = 'Cu'
       
       box.voltage
       >> 30.0
       box.current
       >> 40.0
       
       #For setting target open and close shutter we have a context manager
       from sls_detector_tools import xrf_shutter_open
       
       with xrf_shutter_open(box, 'Cu'):
           #do your measurement
           #shutter is closed when exiting
           #this block
    
    .. note:: 
        Requires access to afs, make sure that you klog or (kinit+aklog)
        before starting measurement
        
    """
    
    _shutter_name_to_index = {'XRF': 1,
                     'Direct beam': 3}
    _shutter_index_to_name = {1: 'XRF',
                              3: 'Direct beam'}    
    
    _xrayClient = '/afs/psi.ch/project/sls_det_software/bin/xrayClient64'
    
    def __init__(self):
        if cfg.verbose:
            print('XrayBox')
        logger.info('Class instace initialized')
        
        
    def _call(self, *args):
        """
        Default function to handle command line calls
        Waits for process to return then returns the output
        as a list using a CompletedProcess object
        """
        args = (self._xrayClient, *args)   
        logger.debug(args)
        return subprocess.run(args, stdout=subprocess.PIPE)



    @property
    def HV(self):
        """
        Check or switch on/off the high voltage of the xray tube.

        
        Examples
        ----------
        
        ::
            
            box.HV
            >> True
        
            box.HV = False
            
        """
        out = self._call('getHV')
        a = re.search('(?<=Rxd data:)\w+', out.stdout.decode())
        if a.group() == 'Yes':
            return True
        elif a.group()=='No':
            return False
        else:
            raise ValueError('Could not see if HV is on')
        return out
   
    @HV.setter
    def HV( self, value ):
        if value is True:
            logger.info('Switching on HV')
            out = self._call('HV', 'on')
        else:
            logger.info('Switching off HV')
            out = self._call('HV', 'off')

#            
    @property
    def safe(self):
        """
        Check if it is safe to open the door.
        
        Returns
        --------
        value: bool
            :py:obj:`True` if it is safe and otherwise :py:obj:`False`
        """
        out = self._call('issafe')
        a = re.search('[^:]+[!]',out.stdout.decode())
        print(a.group())
        
        if a.group() == 'Yes, You may open the door now!':
            return True
        else:
            return False



    @property
    def warmup_time(self):
        """
        Read the warmuptime left, returns 0 if warmup not in progress
        """
        out = self._call('getwtime')
        a = re.search('(?<=Rxd data:)\w+', out.stdout.decode())
        return int(a.group())

    @property 
    def voltage(self):
        """
        High voltage of the X-ray tube in kV
        
        ::
            
            xray_box.voltge = 60
            
            xray_box.voltage
            >> 60.0
            
        """
        out = self._call('getActualV')
        a = re.search('(?<=Rxd data:)\w+', out.stdout.decode())
        kV = float(a.group())/1e3
        logger.info('Voltage is %f kV', kV)
        return kV
    
    @voltage.setter
    def voltage(self, kV):
        logger.info('Setting HV to %f kV', kV)
        self._call('setv', str(kV))

    @property
    def current(self):
        """
        Tube current in mA
        
        ::
            
            xray_box.current = 40
            
            xray_box.current
            >> 40.0
        
        """
        out = self._call('getActualC')
        a = re.search('(?<=Rxd data:)\w+', out.stdout.decode())
        mA = float(a.group())/1e3
        logger.info('Current is {:.2f} mA'.format(mA))
        return mA
    
    @current.setter
    def current(self, mA):
        logger.info('Setting current to: {:.2f} mA'.format(mA))
        self._call( 'setc', str(mA))  

        

    def _set_shutter(self, sh, status = 'off'):
        """
        Internal function to open and close shutter. For scripting
        use open_shutter() and close_shutter()
        """
        if isinstance(sh, str):
            try:
                sh_index = self._shutter_name_to_index[sh]
            except KeyError:
                raise ValueError('Shutter name not recognised')
            out = self._call('shutter', str(sh_index), status)
            logger.info('Shutter {:s}: {:s}'.format(sh, status))
            
        elif isinstance(sh, int):
            try:
                sh_name = self._shutter_index_to_name[sh]
            except KeyError:
                raise ValueError('Shutter index not found')
            out = self._call('shutter', str(sh), status)
            logger.info('Shutter {:s}: {:s}'.format(sh_name, status))   

    def open_shutter(self, sh):
        """
        Open shutter
        
        Parameters
        -----------
        sh: :py:obj:`int` or :py:obj:`str` 
            Shutter to open
            
        Examples
        ---------
        
        ::
            
            box.open_shutter('XRF')
            
        """
        self._set_shutter(sh, status = 'on')        
        
    def close_shutter(self, sh):
        """
        Close shutter
        
        Parameters
        -----------
        sh: :py:obj:`int` or :py:obj:`str` 
            Shutter to close
            
        Examples
        ---------
        
        ::
            
            box.close_shutter('XRF')
            
        """
        self._set_shutter(sh, status = 'off')

    @property
    def shutter_status(self):
        """
        Check status of shutters and return dictionary
        
        Examples
        ---------
        
        ::
            
            box.shutter_status
            >> {'Direct beam': 'OFF', 'XRF': 'OFF'}
            
        """
        status = {}
        for i in [1,3]:
            out = self._call('getshutter{:d}'.format(i)) 
            a = re.search('(?<=Shutter )\d:\S+', out.stdout.decode())
            j, s = a.group().split(':')
            sh = self._shutter_index_to_name[int(j)]
            status[sh] = s
      
        return status

    @property
    def target(self):
        """
        Fluorescence target
        
        Examples
        ---------
        
        ::
            
            box.target = 'Fe'
            
            box.target
            >> 'Fe'
            
        """
        out = self._call('getfl')
        return re.search('(?<=Fl is )\w\S+', out.stdout.decode()).group()

               
    @target.setter
    def target( self, target_name ):
        """Set the target of the xray box

        Parameters
        ----------
        value: target_name, str
            Two letter name of the target: Ti, Sc, Fe etc. 

        Returns
        ----------
        t_set:  str   
            The name of the set target
        
        """
        if target_name == 'Zr':
            target_name = 'Empty'
        logger.info('Switching to %s target', target_name)
        out = self._call('movefl', target_name)
        if cfg.debug:
            print(out)
        return out
        t_set = re.search('(?<=to )\w+', out.stdout.decode()).group()
        if t_set != target_name:
            print(out)
            logger.error('Target not found')
            raise ValueError('Target not found!') 
        logger.debug('Target set to: %s', t_set)
#        return t_set
              
    def unlock(self):
        """
        Unlock the control from another user
        """
        logger.info('Unlocking Xray box from other users')
        out = self._call( 'unlock')
        print(out.stdout.decode())

        
