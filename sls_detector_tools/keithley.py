# -*- coding: utf-8 -*-
"""
Author: Erik Frojdh
"""

import serial
import time
import logging
logger = logging.getLogger()


class SerialInstrument:
    """
    Base class for Keithley serial instruments like SourceMeter and PicoamMeter
    """

    def __init__(self, verbose=False):
        self.verbose = verbose


    def open_port(self,
                  port='/dev/ttyUSB0',
                  baudrate=9600,
                  parity=serial.PARITY_NONE,
                  stopbits=serial.STOPBITS_ONE,
                  bytesize=serial.EIGHTBITS,
                  xonxoff=0,
                  timeout=3,
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

    def read_until(self, char):
        """
        Read from the serial port one character at the time until
        char or timeout
        """
        out = ''
        while True:
            r = self.serial.read(1).decode()
            out += r
            if r == char or len(r) == 0:
                break
        return out

    def write_and_read(self, cmd):
        """
        Write a command to the serial port and read the answer.
        Returns after reading size or waiting for timeout
        """
        self.serial.write(cmd)
        return self.read_until('\r')



    @property
    def instrument_id(self):
        return self.write_and_read(b'*IDN?\n')

class PicoamMeter(SerialInstrument):
    
    def configure(self):
        """Standard configuration for reading current"""
        self.clear()
        self.serial.write(b':FORM:ELEM READ,UNIT\n')
        self.serial.write(b':CONF:CURR\n')
        print(self.write_and_read(b'FORM:ELEM?\n'))
        print(self.write_and_read(b'CONF:CURR?\n'))
    
    @property
    def current(self):
        s = self.write_and_read(b':MEAS:CURR?\n')
        return float(s.strip('A\r'))

    def clear(self):
        self.serial.write(b'CLS\n')

    @property
    def voltage(self):
        s = self.write_and_read(b':SOUR:VOLT:LEV?\n')
        return float(s)
    
    @voltage.setter
    def voltage(self, value):
        tmp = ':SOUR:VOLT:LEV {}\n'.format(value)
        tmp = bytes(tmp, 'utf')
        self.serial.write(tmp)
        
    @property
    def output(self):
        s = self.write_and_read(b':SOUR:VOLT:STAT?\n')[0]
        print(s)
        if s == '0':
            return False
        elif s == '1':
            return True
        else:
            raise ValueError('Unknown return: {}'.format(s))

    @output.setter
    def output(self, value):
        if value == True:
            self.serial.write(b':SOUR:VOLT:STAT 1\n')
        else:
            self.serial.write(b':SOUR:VOLT:STAT 0\n')

class SourceMeter:
    """
    Class to control a Keithley 2410 SourceMeeter over serial
    interface
    """
    def __init__(self, verbose = False):
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
        self.serial.write(b':READ?\n')
        m=self.serial.readlines()
        return m
        
    def get_digits(self):
        self.serial.write(b':SENSE:CURR:DIG?\n')
        m=self.serial.readlines()
        return m
    def set_digits(self, n):
        self.serial.write(b':SENSE:CURR:DIG '+str(n)+'\n')
     
    def on(self):
        self.serial.write(b'output on\n')
    def off(self):
        self.serial.write(b'output off\n')
        
    def set_voltage(self,V):
        s='SOUR:VOLT:LEV {:d}\n'.format(V)
        self.serial.write(bytes(s, 'utf-8'))

    def remote(self, flag):
        """
        Turn on and off remote operation of the 
        keithley , remote(True) locks the physical
        panel
        """
        if flag:
            self.serial.write( b'syst:rem\n')
        else:
            self.serial.write( b'syst:loc\n')


