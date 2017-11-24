#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:08:09 2017

@author: l_frojdh
"""
import telnetlib
import time

class AgilentMultiMeter:

    def __init__(self, host, port):
        """AgilentMultiMeter(host, port)
        Open the communication with the instrument and resets the  outVals.
        host: hostname of the instrument
        port: communication port to open"""
        self.tn = telnetlib.Telnet(host, port)
        time.sleep(1)
        print(self.tn.read_eager())
        time.sleep(1)
        print(self.tn.read_eager())
        self.outVals=[]
        
    def measure(self):
        """AgilentMultiMeter.measure()
        Does one measurement with the open instrument, adds the measured value to the outVals and returns it"""
        self.tn.write( b'Measure?\n' )
        time.sleep(0.1)
        out = self.tn.read_eager()
        v=float(out.split(b'\r')[0])
        self.writeVal(v)

        return v

    def measureNtimes(self,N):
        """AgilentMultiMeter.measureNtimes(N)
        Does N measurements with the open instrument and calculates the average, adds the averaged value to the outVals and returns it"""
        v0=0
        time.sleep(0.05)
        for i in range(0,N):
            self.tn.write( b'Measure?\n' )
            time.sleep(0.1)
            out = self.tn.read_eager()
            v=float(out.split(b'\r')[0])
            v=v0+v
        newVal=v/N
        self.writeVal(newVal)
        return v

    def clearVals(self):
        """AgilentMultiMeter.clearVals()
        Clears the output values outVals"""
        self.outVals=[]
        print("***Output values buffer cleared****")

    def writeVal(self,v):
        """AgilentMultiMeter.writeVals(v)
        Writes v into the output values outVals"""
        self.outVals.append(v)



#if __name__ == '__main__':
#        host = 'A-34410A-13330.psi.ch'
#        port = 5024
#        inst = Agilent(host, port)
#     
#class AgilentMultiMeterOld:
#
#    def __init__(self, host, port):
#        self.tn = telnetlib.Telnet(host, port)
#        time.sleep(1)
#        print(self.tn.read_eager())
#        time.sleep(1)
#        print(self.tn.read_eager())
#
#    def measure(self):
#        self.tn.write( b'Measure?\n' )
#        time.sleep(0.1)
#        out = self.tn.read_eager()
#        return float(out.split(b'\r')[0])
