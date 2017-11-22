# -*- coding: utf-8 -*-
"""
Module to control the big xray box at PSI. Communicates using the command line
to the */afs/psi.ch/project/sls_det_software/bin/xrayClient64*. Provides a 
DummyBox that only writes the commands to the log file without doing anything.

"""
from __future__ import print_function
from . import config as cfg
from subprocess import Popen, PIPE
import logging
logger = logging.getLogger()

from contextlib import contextmanager
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
        
    @contextmanager
    def shutter_open(self):
        print('OPEN')
        yield
        print('CLOSE')

 
    
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
    
    TODO! Inspect return values to look for errors
    
    .. note:: 
        Requires access to afs 
    """
    def __init__(self):
        self.box = '/afs/psi.ch/project/sls_det_software/bin/xrayClient64'
        if cfg.verbose:
            print('XrayBox')
        logger.info('Class instace initialized')
        

            
    def _call(self, arg):
        """
        Default function to handle command line calls
        Waits for process to return then returns the output
        as a list
        """
        logger.debug(arg)
        p = Popen(arg, stdout=PIPE)
        p.wait()
        out = p.stdout.readlines()            
        return out

   
    def HV( self, value ):
        """
        Switch on the high voltage is True otherwise switch off
        
        Parameters
        ----------
        value: bool
            True for on, False for off
        """
        if value is True:
            logger.info('Switching on HV')
            out = self._call([self.box, 'HV', 'on'])
        else:
            logger.info('Switching off HV')
            out = self._call([self.box, 'HV', 'off'])

            
    def set_kV( self, kV ):
        """
        Set the kV of the tube
 
        Parameters
        ----------
        kV: int
            Voltage of the tube in kV       
        """
        logger.info('Setting HV to %f kV', kV)
        out = self._call([self.box, 'setv', str( kV )])
        return out
        
    def get_kV( self ):
        """
        Read the tube voltage
        
        Returns
        ----------
        kV: int
            Voltage of the tube in kV
            
        """
        kV = 0

        out = self._call([self.box, 'getv'])
        try:
            kV = float(out[2].split()[-1].split(':')[-1])/1e3
        except IndexError:
            print(out)
            
        if cfg.verbose:
            print('XrayBox: Voltage '+str(kV) + ' kV')
        logger.info('Voltage is %f kV', kV)
        return kV

        
    def set_mA( self, mA ):
        """
        Set the tube current in mA
        """
        logger.info('Setting current to: %f mA', mA)
        out = self._call([self.box, 'setc', str( mA )])

        
    def get_mA( self ):

        out = self._call([self.box, 'getc'])
        try:
            mA = float(out[2].split()[-1].split(':')[-1])/1e3
        except IndexError:
            print(out)
            mA = False
            
        if cfg.verbose:
            print('XrayBox: Current '+str(mA) + ' mA')
        logger.info('Current is %f mA', mA)
        return mA
        
    def shutter(self, value, sh = 1):
        """
        Open the shutter th if value is True otherwise close shutter sh
        
        Parameters
        ----------
        value: bool
            True for on, False for off
        sh: int, optional
            1 for XRF, 3 for direct beam
    
        """
        if sh == 1:
            shutter_type = 'XRF'
        elif sh==3:
            shutter_type = 'Direct beam'
        else:
            raise ValueError('Invalid shutter index, needs to be 1 for XRF or 3 for direct beam')
            
        if value is True:
            out = self._call([self.box, 'shutter', str(sh), 'on'])
            logger.info('Opening shutter for %s', shutter_type)
        else:
            out = self._call([self.box, 'shutter', str(sh), 'off'])
            logger.info('Closing shutter for %s', shutter_type)

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
        out = self._call([self.box, 'movefl', target_name])
        if cfg.debug:
            print(out)
        t_set = out[3].split()[4].strip(':')
        if t_set != target_name:
            print(out)
            logger.error('Target not found')
            raise ValueError('Target not found!') 
        logger.debug('Target set to: %s', t_set)
        return t_set
              
    def unlock(self):
        """
        Unlock the control from another user
        """
        logger.info('Unlocking Xray box from other users')
        out = self._call([self.box, 'unlock'])

        
