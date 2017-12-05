# -*- coding: utf-8 -*-
"""
Various functions that might come in handy but doesn't really
fit well into a specific block
"""
#Python

import numpy as np
from scipy.interpolate import interp1d

#sls detector
from sls_detector_tools import function


def get_dtype(dr):
    """
    Returns the correct numpy dtype from a number or string
    """
    if isinstance(dr, str):
        dr = int(dr)
        
    if dr == 32:
        return np.int32
    elif dr == 16:
        return np.int16
    elif dr == 8:
        return np.uint8
    elif dr == 4:
        return np.uint8
    else:
        raise TypeError('dtype: {:d} not supported'.format(dr))




def normalize_flatfield(image):
    """
    Return a normalized flatfield image based on the current image
    """
    #Remove 1% lowest and 1% highest pixels and take mean value
    a_sorted = np.argsort(image.flat)
    index = a_sorted.size//100
    low = image.flat[a_sorted[index]]
    high = image.flat[a_sorted[-index]]
    mean = image[(image > low) & (image < high)].mean()

    #Normalize and remove zeros to avoid NaN
    flatfield = image/mean
    flatfield[flatfield == 0] = 1

    return flatfield

def random_pixel(n_pixels=1, rows=(0, 512), cols=(0, 1024)):
    """
    Generate a list of random pixels with the default beeing one
    pixel in a single module

    Parameters
    -----------
    n_pixels: int, optional
        Number of  pixels to return
    rows: (int, int), optional
        Lower and upper bounds for the rows
    cols: (int, int), optional
        Lower and upper bounds for the cols

    Returns
    --------
    pixels: list of tuples
        List of the pixels [(row, col), (...)]

    Examples:
    ---------

    ::

        random_pixel()
        >> [(151, 30)]

        random_pixel(n_pixels = 3)
        >> [(21, 33), (65, 300), (800,231)]


    """
    return [(np.random.randint(*rows), np.random.randint(*cols))
            for i in range(n_pixels)]


def generate_scurve(x, n_photons):
    """
    Return an scurve with some typical parameters
    """
    #Scale C propotional to A s for a real measurement
    C = n_photons/1000.*0.4
    y = function.scurve(x, 0, 0, 1000, 170, n_photons, C)
    y = np.random.poisson(y)
    return y


def R(x):
    """
    Quality measurement for one refelction spot. To be used with
    a numpy array having the number of counts for a simulated
    or measured spot
    """
    return np.sum(np.abs(x-x.mean())) / x.sum()


def ratecorr(values, tau, exptime):
    """
    perform rate correction on a numpy array given tau and expsure time
    values above the maximum possible will be replaced with the maximum

    data = data to correct
    tau = dead time ns
    exptime = exposuretime in s
    """
    n = 1./exptime
    tau *= 1e-9
    data = values.copy()
    data *= n

    #Generate data for function
    x = np.linspace(0, 10e6, 1000)
    y = x*np.exp(-tau*x)

    #Position of maximum counts
    j = np.argmin(np.abs(np.gradient(y))) - 1

    #Shorten arrays to not include values above max
    x = x[0:j]
    y = y[0:j]
    f = interp1d(y, x)

    #Find max and correct values below the maximum
    ratemask = data < y.max()
    corrected_data = data.copy()
    corrected_data[ratemask] = f(data[ratemask])
    corrected_data[~ratemask] = f(y.max())
    corrected_data /= n

    return corrected_data

def sum_array(data, sum_size):
    """
    Sum evry sum_size element of a numpy array and return the
    summed array.

    data = 1d numpy array
    sum_size = number of consecutive elements to sum
    """
    #Check lengt
    remove = data.size % sum_size
    return data[0:-remove].reshape(-1, sum_size).sum(axis=1)
