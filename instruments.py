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
        self.tn = telnetlib.Telnet(host, port)
        time.sleep(1)
        print(self.tn.read_eager())
        time.sleep(1)
        print(self.tn.read_eager())

    def measure(self):
        self.tn.write( b'Measure?\n' )
        time.sleep(0.1)
        out = self.tn.read_eager()
        return float(out.split(b'\r')[0])

def do_something():
        print("hej")



if __name__ == '__main__':
        host = 'A-34410A-13330.psi.ch'
        port = 5024
        inst = Agilent(host, port)
