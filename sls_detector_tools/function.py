# -*- coding: utf-8 -*-
"""
Commonly used functions
"""

import numpy as np
from scipy.special import erf



#------------------------------------------------------------General functions

def gaus(x, A, mu, sigma):
    """
    Gaussian function

    .. math ::

        f(x) = A e^{-0.5*(x-\mu) / {\sigma^2}}


    Parameters
    -----------
    x:
        x values to evaluate the function at
    A: double
        scaling
    mu: double
        center
    sigma: double
        width

    Returns
    --------
    y: value or array
        evaluated vales at each x
    """
    return A*np.exp(-0.5*((x-mu)/sigma)**2)


def expo(x, p0, p1):
    """
    Exponential


    :math:`e^{p0+p1*x}`


    """
    return np.exp(p0+p1*x)


def pol1(x, p0, p1):
    """
    Linear function. Parameters in the same order as in ROOT

    .. math ::

        f(x) = p_0 + p_1x

    """
    return p0+p1*x

def pol2(x, p0, p1, p2):
    """
    Second degree polynomial

    .. math ::

        f(x) = p_0 + p_1x + p_2x^2
    """
    return p2*x**2 + p1*x + p0

def pol3(x, p0, p1, p2, p3):
    """
    Third degree polynomial

    .. math ::

        f(x) = p_0 + p_1x + p_2x^2 + p_3x^3
    """
    return p3*x**3 + p2*x**2 + p1*x + p0


#---------------------------------------------------------Special functions

def paralyzable(x, tau):
    """
    Paralyzable detector model, used for rate measurements

    .. math ::

        f(x) = xe^{- \\tau x}

    """
    return x*np.exp(-tau*x)

#--------------------------------------------------------- Edge functions


#[0]/2 * (1-TMath::Erf( (x-[1])/([2]*sqrt(2)) ))
def gaus_edge(x, A, mu, sigma):
    return A/2 * (1-erf((x-mu)/(sigma*np.sqrt(2))))

def double_gaus_edge(x, A, mu, sigma1, sigma2):
    """
    double gaussian
    """
    return A/2 * ((1-erf((x-mu)/(sigma1 * np.sqrt(2)))) +
                  (1-erf((x-mu)/(sigma2 * np.sqrt(2)))))

def double_gaus_edge_new(x, p0, A, mu, sigma1, sigma2):
    """
    Variant of the double gaussian.
    """
    return A/4 * ((1-erf((x-mu)/(sigma1 * np.sqrt(2)))) +
                  (1-erf((x-mu)/(sigma2 * np.sqrt(2)))))

#---------------------------------------------------------- scurves

def scurve(x, p0, p1, mu, sigma, A, C):
    """
    scruve function used for energy calibration
    scurve(x, p0,p1, mu, sigma, A, C)

    [0] - p0
    [1] - p1
    [2] - mu
    [3] - sigma
    [4] - A
    [5] - C
    """
    y = (p0+p1*x) + 0.5*(1+erf((x-mu)/(np.sqrt(2)*sigma))) * (A + C*(x-mu))
    return y


def scurve2(x, p0, p1, mu, sigma, A, C):
    """
    scruve function used for energy calibration
    scurve(x, p0,p1, mu, sigma, A, C)

    [0] - p0
    [1] - p1
    [2] - mu
    [3] - sigma
    [4] - A
    [5] - C
    """
    y = (p0+p1*x) + 0.5*(1-erf((x-mu)/(np.sqrt(2)*sigma))) * (A + C*(x-mu))
    return y

def scurve4(x, p0, p1, mu, sigma, A, C):
    """
    scruve function used for energy calibration
    scurve(x, p0,p1, mu, sigma, A, C)

    [0] - p0
    [1] - p1
    [2] - mu
    [3] - sigma
    [4] - A
    [5] - C
    """
    return (p0+p1*x) + 0.5*(1-erf((x-mu)/(np.sqrt(2)*sigma))) * (A + A/C*(x-mu))


def fermi(x, a,b,c):
    return a/(1+np.exp(-(x-b)/c))


def ideal_mtf(omega):
    """
    mtf for an ideal pixel detector
    """
    return np.sin((np.pi*omega)) / (np.pi*omega)

def ideal_dqe(omega):
    """
    Expression for the ideal DQE given an ideal MTF
    """
    return ideal_mtf(omega)**2


#------------------------------------------ROOT style functions for fitting


def eval_func_obj(f, x, par):
    """Evaluate a ROOT style function object for all x"""
    assert len(par) == f.npar
    y = np.zeros(len(x))
    for i,val in enumerate(x):
        y[i] = f([val], par)
    return y


class GausFunc:
    npar = 3
    par_names = ['A', 'mu', 'sigma']
    def __init__(self, par = None):
        if par is not None:
            if len(par) != self.npar:
                raise ValueError("pars")
        self.ipar = par
    def eval(self, x, par):
        return eval_func_obj(self, x, par)

    def __call__(self, x, par):
        return par[0]*np.exp(-0.5*((x[0]-par[1])/par[2])**2)

class GausEdgeFunc:
    npar = 3
    par_names = ['A', 'mu', 'sigma']
    def __init__(self, par = None):
        if par is not None:
            if len(par) != self.npar:
                raise ValueError("pars")
        self.ipar = par
    def eval(self, x, par):
        return eval_func_obj(self, x, par)

    def __call__(self, x, par):
        return par[0]/2 * (1-erf((x[0]-par[1])/(par[2]*np.sqrt(2))))


class TrapFunc:
    npar = 5
    par_names = ['Offset', 'Center', 'Width', 'Amplitude', 'Charge Sharing']
    def __init__(self, par = None):
        if par is not None:
            if len(par) != self.npar:
                raise ValueError("pars")
        self.ipar = par

    def eval(self, x, par):
        return eval_func_obj(self, x, par)

    def __call__(self, x, par):
    #     par0 is a global offset
    #     par1 Is the center (inflection point?)
    #     par2 is the width of the tilted part of the trapezium
    #     par3 is the amplitude
    #     par4 is the region without charge sharing
        z = (x[0]-par[1])
        
        if z > 0:
            y =- 1.*z
        else:
            y = z

        if par[4] <= 0: 
            if z >= 0:
                return par[3]+par[0]
            else:
                return 0
        
        if par[2] <= 0:
            if z<-0.5*par[4]:
                return 0
            elif z>0.5*par[4]:
                return par[3]+par[0]
            else:
                return 0.5*par[3]+par[3]*z/par[4]
        
        if par[2]>par[4]:
            par[2]=par[4]

        f=0.
        if y<=(-0.5*(par[4]+par[2])):
            f=par[0]
        elif y<=(-0.5*(par[4]-par[2])):
            f=par[0]+0.5*par[3]*(y+(0.5*(par[4]+par[2])))*(y+(0.5*(par[4]+par[2])))/((par[4]*par[2]))
        elif y<=0.5*(par[4]-par[2]):
            f=par[0]+par[3]*0.5+par[3]*y/par[4]
        
        if z>0:
            f=par[3]+par[0]-f
        
        f=par[3]+par[0]-f
        return f





#-- ROOT strings to create TF1 functions
class root_fstring:
    """
    Strings to build ROOT functions from
    """
    scurve = ' ([0]+[1]*x) + 0.5 * (1+TMath::Erf( (x-[2])/(sqrt(2)*[3]) ) )'\
             '* ( [4] + [5]*(x-[2])) '
    scurve2 = ' ([0]+[1]*x) + 0.5 * (1-TMath::Erf( (x-[2])/(sqrt(2)*[3]) ) )'\
              '* ( [4] + [5]*(x-[2])) '

    #normalize [5]
    scurve4 = ' ([0]+[1]*x) + 0.5 * (1-TMath::Erf( (x-[2])/(sqrt(2)*[3]) ) )'\
              '* ( [4] + [4]/[5]*(x-[2])) '

    gaus_edge = '[0]/2 * (1-TMath::Erf( (x-[1])/([2]*sqrt(2)) ))'

    # Doulble edge
    double_gaus_edge = '[0]+[1]/4 * ((1-TMath::Erf( (x-[2])/(sqrt(2)*[3])))'\
                       '+ (1-TMath::Erf( (x-[2])/(sqrt(2)*[4]) ) ) ) '

    fermi = '[0]/(1+TMath::Exp(-(x-[1])/[2]))'
