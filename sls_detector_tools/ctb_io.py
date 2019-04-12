#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype to read Files from the ctb
"""


import numpy as np


n_counters = 64*3
header_dt = [('frameNumber',np.uint64),
             ('expLength',np.uint32),
             ('packetNumber', np.uint32),
             ('bunchId', np.uint64),
             ('timestamp', np.uint64),
             ('modId', np.uint16),
             ('row', np.uint16),
             ('col', np.uint16),
             ('reserved', np.uint16),
             ('debug', np.uint32),
             ('roundRNumber', np.uint16),
             ('detType', np.uint8),
             ('version', np.uint8)]


def ExtractBits(data, dr=24, bit_nr0 = 17, bit_nr1 = 6):
    bit_nr0 = np.uint64(bit_nr0)
    bit_nr1 = np.uint64(bit_nr1)
    mask0 = np.uint64(1) << bit_nr0
    mask1 = np.uint64(1) << bit_nr1
    counters = np.zeros(n_counters, dtype = np.uint64)
    
    data_it = np.nditer(data, op_flags=['readwrite'])
    counter_it0 = np.nditer(counters, op_flags=['readwrite'])
    counter_it1 = np.nditer(counters[96:], op_flags=['readwrite'])
    c=0
    while not(counter_it1.finished):
        for i in np.arange(dr, dtype = np.uint64):
#            print(f'{data_it.value:064b}')
            tmp0 = np.uint64((data_it.value  & mask0) >> bit_nr0)
            tmp1 = np.uint64((data_it.value  & mask1) >> bit_nr1)
            counter_it0[0] = counter_it0[0] | (tmp0 << i)
            counter_it1[0] = counter_it1[0] | (tmp1 << i)
            data_it.iternext()
#        print('\n')
        counter_it0.iternext()
        counter_it1.iternext()


    return counters


def read_my302_file(fname, dr=24, bit_nr0 = 17, bit_nr1 = 6, offset=0):
    with open(fname, 'rb') as f:
        header = np.fromfile(f, count=1, dtype = header_dt)
        bitfield = np.fromfile(f, count=64, dtype = np.uint8)
        
        f.seek(offset,1)
        #data should start here
        data = np.fromfile(f, dtype = np.uint64)
    counters = ExtractBits(data, dr=dr, bit_nr0=bit_nr0, bit_nr1=bit_nr1)
    return header, counters





