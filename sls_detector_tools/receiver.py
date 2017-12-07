#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:52:12 2017

@author: l_frojdh
"""

import zmq
import json
import numpy as np
import warnings

#import sys
#sys.path.append('/home/l_frojdh/slsdetectorgrup/sls_detector_tools')
from .mask import eiger500k
from .utils import get_dtype



class ZmqReceiver:
    """
    Simple receiver that reads data from zmq streams and put this together
    to an image. 
    
    .. warning ::
        
        Currently only Eiger500k
    
    expects:
    json - header
    data - as specified in header
    json - end of acq
    """
    def __init__(self, ip, ports):
        warnings.warn('ZmqReceiver currently only supports Eiger500k')
        
        self.ports = ports
        self.ip = ip
        self.context = zmq.Context()
        self.sockets = [ self.context.socket(zmq.SUB) for p in self.ports ]
        self.mask = eiger500k()
        #connect sockets
        for p,s in zip(self.ports, self.sockets):
            print('Initializing: {:d}'.format(p))
            s.connect('tcp://{:s}:{:d}'.format(self.ip, p))
            s.setsockopt(zmq.SUBSCRIBE, b'')

    def get_frame(self):
        """
        Read one frame from the streams
        
        .. todo::
            
            4 bit mode
            
            
        """
        image = np.zeros((512,1024))
        for p,s in zip(self.mask.port, self.sockets):
            header = json.loads( s.recv() )
            data = s.recv()
            end = json.loads( s.recv() )
            if header['bitmode'] == 4:
                print('4bit')
                tmp = np.frombuffer(data, dtype=np.uint8)
                tmp2 = np.zeros(tmp.size*2, dtype = tmp.dtype)
                tmp2[0::2] = np.bitwise_and(tmp, 0x0f)
                tmp2[1::2] = np.bitwise_and(tmp >> 4, 0x0f)
                image[p] = tmp2.reshape(256,512)
#                image[p][1::2] = np.bitwise_and(tmp >> 4, 0x0f)
            else:
                image[p] = np.frombuffer(data, dtype = get_dtype(header['bitmode'])).reshape(256,512)
        
        #flip bottom
        image[0:256,:] = image[255::-1,:]
        return image
#        return data
            
