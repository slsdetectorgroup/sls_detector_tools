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
from . import mask as mask
from .utils import get_dtype
from . import config as cfg


class ZmqReceiver:
    """
    Simple receiver that reads data from zmq streams and put this together
    to an image. 
    
    .. warning ::
        
        Current support: 250k, 500k and 9M. Only single frame acq.
    
    expects:
    json - header
    data - as specified in header
    json - end of acq
    """
    def __init__(self, detector):
        warnings.warn('ZmqReceiver currently only supports single frames')

        ip = detector.rx_zmqip #Workaround until we get zmqip
        ports = detector.rx_zmqport

        #ip and ports
        self.image_size = detector.image_size
        self.ports = ports
        self.ip = ip
        self.context = zmq.Context()
        self.sockets = [ self.context.socket(zmq.SUB) for p in self.ports ]
        
        self.mask = mask.detector[cfg.geometry]
        #connect sockets
        for p,s in zip(self.ports, self.sockets):
            print('Initializing: {:d}'.format(p))
            s.connect('tcp://{:s}:{:d}'.format(self.ip, p))
            s.setsockopt(zmq.SUBSCRIBE, b'')

    def get_frame(self):
        """
        Read one frame from the streams

            
        """
        image = np.zeros(self.image_size.y, self.image_size.x)
            
        for p,s in zip(self.mask.port, self.sockets):
#            header =  s.recv()
#            return header
            header = json.loads( s.recv()[0:-1] ) #Temporary fix for 4.0.0
            data = s.recv()
            end = json.loads( s.recv()[0:-1] )
            if header['bitmode'] == 4:
                print('4bit')
                tmp = np.frombuffer(data, dtype=np.uint8)
                tmp2 = np.zeros(tmp.size*2, dtype = tmp.dtype)
                tmp2[0::2] = np.bitwise_and(tmp, 0x0f)
                tmp2[1::2] = np.bitwise_and(tmp >> 4, 0x0f)
                image[p] = tmp2.reshape(256,512)
            else:
                image[p] = np.frombuffer(data, dtype = get_dtype(header['bitmode'])).reshape(256,512)
        
        #flip bottom
        for hm in self.mask.halfmodule[1::2]:
            image[hm] = image[hm][::-1,:]

        return image

