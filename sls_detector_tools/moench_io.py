#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:06:00 2018

@author: l_frojdh
"""
import numpy as np

def generateDataMap(block_size):
    dataMap = np.zeros((160,160), dtype = np.int)
    adc_off = [40,0,120,80]
    adc_nr = [8,10,20,23]
    sc_width = 40
    sc_height = 160
    for iiadc in range(4):
        iadc = adc_nr[iiadc]
        for i in range(sc_width*sc_height):
            col = adc_off[iiadc] + (i % sc_width)
            row = i // sc_width
            dataMap[row, col] = (block_size * i + iadc) #*2
    return dataMap 


dataMapAnalog = generateDataMap(32)
dataMapAnalogDigital = generateDataMap(36)


def _read_frame2(fname, block_size):
    block = np.fromfile(fname, count = 6410*block_size, dtype = np.int16).reshape(6410,block_size)
    image = np.zeros((160,160))
    image[:,0:40] = block[0:6400,10].reshape(160,40)
    image[:,40:80] = block[0:6400,8].reshape(160,40)
    image[:,80:120] = block[0:6400,23].reshape(160,40)
    image[:,120:160] = block[0:6400,20].reshape(160,40)
    return image

def read_analog(fname):
    block_size = 32
    buffer = np.fromfile(fname, count = 6400*block_size, dtype = np.int16)
    image = np.zeros((160,160))
    for r in range(160):
        for c in range(160):
            image[r,c] = buffer[dataMapAnalog[r,c]]
    return image

def read_analog_digital(fname):
    block_size = 36
    buffer = np.fromfile(fname, count = 6400*block_size, dtype = np.int16)
    image = np.zeros((160,160))
    gain = np.zeros(image.shape)
    
    #Decode image
    for r in range(160):
        for c in range(160):
            image[r,c] = buffer[dataMapAnalogDigital[r,c]]
            gain[r,c] = buffer[dataMapAnalogDigital[r,c]+22]
            
    return image, gain