#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:16:27 2018

@author: l_frojdh
"""
import numpy as np
from ROOT import TF1, TGraph
import matplotlib.pyplot as plt

from . import root_helper as r
from . function import pol1, pol2, gaus_edge, ideal_mtf


def _fit_rows(image):
    col = np.arange(image.shape[1])
    edge_pos = np.zeros(image.shape[0])
    for i,row in enumerate(image):
        #Normalize the row
        row -= row[-7::].mean()
        row /= row[0:10].mean()
    
        #Find inital parameters for the fit
        C0 = row.max()
        mu0 = col[row<0.5][0]
        sigma0 = .15 
        
        func = TF1('func', '[0]/2 * (1-TMath::Erf( (x-[1])/([2]*sqrt(2)) ))')  
        func.SetParameter(0, C0)
        func.SetParameter(1, mu0)
        func.SetParameter(2, sigma0)
        g = TGraph(row.size, col.astype(np.double), row.astype(np.double))
        for j in range(3):
            fit = g.Fit('func', 'SQ')
        edge_pos[i] = fit.Get().Parameter(1)   
        
    return edge_pos

def _fit_edge_angle(edge_pos, image):
    x = np.arange(image.shape[0], dtype = np.double)
    c,h = r.plot(x, edge_pos)
    fit = h.Fit('pol2', 'SQ')
    par = [fit.Get().Parameter(i) for i in range(3)]
    angle = np.arctan(par[1])
    print(f'Edge angle is: {np.rad2deg(angle):.2f} degrees')
    return par, angle

def _find_presample_edge(image, par, angle):
    xx = np.zeros(image.size)
    yy = np.zeros(xx.size)
    xpixel = np.arange(image.shape[1])
    
    for i in range(image.shape[0]):
        x = xpixel-pol2(i, *par)
        x *= np.cos(angle) #Projection perpendicular to the edge
        y = image[i,:]
        xx[i*image.shape[1]:(i+1)*image.shape[1]] = x
        yy[i*image.shape[1]:(i+1)*image.shape[1]] = y
    
    idx = np.argsort(xx)
    xx = xx[idx]
    yy = yy[idx]
    return xx, yy

def find_edge(image):
    edge_pos = _fit_rows(image)
    par, a = _fit_edge_angle(edge_pos, image)
    x,y = _find_presample_edge(image, par, a)
    return x, y

def calculate_mtf(x,y):
    pixel_range = (-3,3)
    c,h = r.plot(x[(x>pixel_range[0])&(x<pixel_range[1])],y[(x>pixel_range[0])&(x<pixel_range[1])])
    func = TF1('func', '[0]/2 * (1-TMath::Erf( (x-[1])/([2]*sqrt(2)) ))')  
    func.SetParameter(0, 1)
    func.SetParameter(1, 0)
    func.SetParameter(2, .15)
    for i in range(3):
        fit = h.Fit('func', 'SQ')
    par = [fit.Get().Parameter(i) for i in range(3)]
    c.Draw()

    #Plot the presample edge, fit and residuals
    fig, ax = plt.subplots(2,1, figsize = (14,14))
    ax[0].plot(x,y, '.', label= 'Presample edge')
    ax[0].plot(x, gaus_edge(x,*par), label = 'Fit')
    ax[0].set_xlim(-2,2)
    ax[0].grid(True)
    ax[0].set_xlabel('Distance [pixels]')
    ax[0].set_ylabel('Normalized edge')
    ax[0].legend()
    ax[1].plot(x, y-gaus_edge(x,*par))
    ax[1].set_xlim(-2,2)
    ax[1].grid(True)
    ax[1].set_xlabel('Distance [pixels]')
    ax[1].set_ylabel('Resiudals')
    
    #Calculate MTF from fit
    x_fit = np.linspace(-200,200, 20000)
    y_fit = gaus_edge(x_fit,*par)
    psf = -np.gradient(y_fit, x_fit)
    f = np.abs( np.fft.fft(psf))
    f = f/f[0]
    d =x_fit[1]-x_fit[0]
    n = x_fit.size
    u = np.fft.fftfreq(n,d)
    f = f[0:u.size//2]
    u = u[0:u.size//2]
    
    fig, ax = plt.subplots(2,1, figsize = (14,14))
    ax[0].plot(x_fit,psf, label = 'Fitted LSF')
    ax[0].set_xlim(-2,2)
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(u, f, '-', label = 'MTF from fitted edge')
    ax[1].set_xlim(0,0.5)
    ax[1].plot(u, ideal_mtf(u), label = 'Ideal MTF')
    ax[1].legend()
    ax[1].grid(True)
    
    #Calculate direct MTF
    delta = 0.1
    x_resample = np.arange(-30,30,delta)
    y_resample = np.zeros(x_resample.size)
    for i, xp in enumerate(x_resample):
        y_resample[i] = y[(x>(xp-delta/2)) & (x<(xp+delta/2))].mean()
    

    lsf_resample = -np.gradient(y_resample, x_resample)
    f = np.abs( np.fft.fft(lsf_resample))
    f = f/f[0]
    d =x_resample[1]-x_resample[0]
    n = x_resample.size
    u_resample = np.fft.fftfreq(n,d)
    f = f[0:u_resample.size//2]
    u_resample = u_resample[0:u_resample.size//2]
    

    ax[0].plot(x_resample,lsf_resample, 'o', label = 'Direct LSF')
    ax[0].set_xlim(-2,2)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_xlabel('Distance [pixels]')
    ax[0].set_ylabel('Differential of normalized edge')
    ax[1].plot(u_resample, f, 'o', label = 'Direct MTF')
    ax[1].legend()
#    ax[1].set_xlabel[]
    ax[1].grid(True)