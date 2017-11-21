# -*- coding: utf-8 -*-
"""
Author: Erik Frojdh
"""

import serial
import time
import logging
logger = logging.getLogger()

import sls_detector.config as cfg
class SourceMeter():
    """
    Class to control a Keithley 2410 SourceMeeter over serial
    interface
    """
    def __init__(self, verbose = False):
#        print "Keithley 2400"
        self.verbose = verbose
    
    def open_port(self, 
                  port = '/dev/ttyUSB0',
                  baudrate=9600,
                  parity=serial.PARITY_NONE,
                  stopbits=serial.STOPBITS_ONE,
                  bytesize=serial.EIGHTBITS,
                  xonxoff=0,
                  timeout=1
                  ):
        """
        Open serial port to communicate with the Keithley 
        make sure that it's set up in a matching way and that
        port number is correct
        """
        if self.verbose:
            print('Keithley: Opening serial port')      
        
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=parity,
            stopbits=stopbits,
            bytesize=bytesize,
            xonxoff=xonxoff,
            timeout=timeout
            )
                   
    def close_port(self):
        if self.verbose:
            print("Keithley: Closing serial port")
        self.serial.close()
        
    def data_elements(self):
        """
        Set output data elements from the Keithley
        now set to Voltage Current and Time
        """
        self.serial.write(':FORM:ELEM VOLT,CURR,TIME\n')
       
    def read(self):
        self.serial.write(':READ?\n')
        m=self.serial.readlines()
        return m
        
    def get_digits(self):
        self.serial.write(':SENSE:CURR:DIG?\n')
        m=self.serial.readlines()
        return m
    def set_digits(self, n):
        self.serial.write(':SENSE:CURR:DIG '+str(n)+'\n')
     
    def on(self):
        self.serial.write('output on\n')
    def off(self):
        self.serial.write('output off\n')
        
    def set_voltage(self,V):
        s='SOUR:VOLT:LEV '+str(V)+'\n'
        self.serial.write(s)

    def remote(self, flag):
        """
        Turn on and off remote operation of the 
        keithley , remote(True) locks the physical
        panel
        """
        if flag:
            self.serial.write( 'syst:rem\n')
        else:
            self.serial.write( 'syst:loc\n')


