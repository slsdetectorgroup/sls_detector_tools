#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 16:34:39 2018

@author: l_frojdh
"""

import numpy as np
import matplotlib.pyplot as plt


# Constants
p =  2.328e3       #kg/m3
eh = 3.62          #eV/eh-pair
q = 1.6022e-19     #C

#Geometry
l = 0.01           #m
t = 10e-6          #m (10um)
volume = l*l*t     #m3
mass = volume * p  #kg

#Measured
#I = 2.37e-5        #A
I = 1.15e-5        #A
e_abs = I * eh / q
dose_rate = e_abs*q/mass #Gray/s

print(f'Dose rate: {dose_rate:.3} Gy/s')


I0 = 1

    
beam = np.array([0.99**i for i in range(11)])
absorption = np.abs(np.diff(beam))
plt.figure()
plt.plot(absorption/absorption[0])