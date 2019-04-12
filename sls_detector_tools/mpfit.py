# -*- coding: utf-8 -*-
"""
Wrapper around the sls_cmodule to provide easy fitting of scurve and
trimbit data. Uses the **multprocessing** module for parallelization. 
"""
#Python
#import sys
import time
import numpy as np
import multiprocessing as mp
import errno

#sls_detector
#from . import config as cfg
import _sls_cmodule



#Workaround for crash in some Python2.7 dists, fixed in Python3
def my_queue_get(q, block = True, timeout = None):
    """
    Get items from queue. This was implemented to avoid a crash in some
    Python2.7 distributions and should not be needed now in Python3
    """
    while True:
        try:
            return q.get(block, timeout)
        except( IOError, e):
            if e.errno != errno.EINTR:
                raise

def mp_wrapper(cols, args, output):
    """
    Wrapper around the sls_cmodule.fit function to fit scurves. Calls the
    function and puts the return value in a queue
    
    Parameters
    ---------
    cols: (int, int)
        Column indicies to put the result in the right place
    args: [data, x, par]
        data to fit, xaxis and initial parameters
    output: Queue
        Queue for the result of the fit
        
    """
    res = (_sls_cmodule.fit( *args ))
    output.put( (cols, res) )
    return

def mp_wrapper_float(cols, args, output):
    """
    Wrapper around the sls_cmodule.fit function to fit scurves. Calls the
    function and puts the return value in a queue
    
    Parameters
    ---------
    cols: (int, int)
        Column indicies to put the result in the right place
    args: [data, x, par]
        data to fit, xaxis and initial parameters
    output: Queue
        Queue for the result of the fit
        
    """
    res = (_sls_cmodule.fit_float( *args ))
    output.put( (cols, res) )
    return

def mp_wrapper2(cols, args, output):
    """
    Wrapper around the sls_cmodule.find_trimbits function to fit the trimbit
    scan data and extract trimbits
    
    Parameters
    ---------
    cols: (int, int)
        Column indicies to put the result in the right place
    args: [data, x, target, par]
        data to fit, xaxis and initial parameters
    output: Queue
        Queue for the result of the fit
        
    """
    res = (_sls_cmodule.find_trimbits( *args ))
    output.put( (cols, res) )
    return
    
def fit(data, x, n_proc, par): 
    """
    Fit scurve per pixel in the 3d numpy_array data[row, col, N] where N in 
    the number of measurements
    
    Parameters
    --------
    data: numpy_array[row, col, N]
        array with the data from the vcmp scan
    x: numpy_array[N]
        xaxis for the fit. Usually vcmp values
    n_proc: int
        Number of processes to run the fit
    par: numpy_array[npar]
        Initial parameters for the fit
        
    Returns
    -------
    fit_result: numpy_array
        Array containing the result of the fit [row, col]. Using named fields
        for the fit parameters
    """
    #Output data type for the fit result
    dt = [('p0', np.double),
          ('p1', np.double),
          ('mu', np.double),
          ('sigma', np.double),
          ('A', np.double),
          ('C', np.double)]
          
    #Numpy array to store the fit result
    shape = np.array( data.shape ) 
    result = np.zeros( (shape[0], shape[1]), dtype = dt )
    
    t0 = time.time()
    output = mp.Queue()
    processes = []
    
    #Launch n_proc processes
    for i in range(n_proc):
        low =  i*shape[1]//n_proc
        high = (i+1)*shape[1]//n_proc
#        print(low, high)
        p = mp.Process(target = mp_wrapper, args = ((low,high), [data[:,low:high,:], x, par], output))
        processes.append(p)
    
    for p in processes:
        p.start()
        
    #Combine the result form the different processes
    for p in processes:
        tmp = my_queue_get(output)
        for i in range(par.size):
            result[dt[i][0]][:, tmp[0][0]:tmp[0][1]] = tmp[1][:,:,i]

    #Wait for finish
    for p in processes:
        p.join()
    
    #Report time
    print( f'Fitting done in: {time.time()-t0:.3}s')
    return result





def find_trimbits(data, x, target, n_proc, par): 
    """
    Fit the trimbit scan to extract the trimbit per pixel
    
    Parameters
    --------
    data: numpy_array[row, col, N]
        array with the data from the trimbit scan
    x: numpy_array[N]
        xaxis for the fit. Usually trimbit values
    target: numpy_array[row, col]
        target value for finding the trimbits 
    n_proc: int
        Number of processes to run the fit
    par: numpy_array[npar]
        Initial parameters for the fit
        
    Returns
    -------
    fit_result: numpy_array
        Array containing the result of the fit [row, col]. Using named fields
        for the fit parameters with trimbits last. 
    """
    #Data type for result, now including trimibts
    dt = [('p0', np.double),
          ('p1', np.double),
          ('mu', np.double),
          ('sigma', np.double),
          ('A', np.double),
          ('C', np.double),
          ('trimbits', np.double)]
          
    #Output array
    shape = np.array( data.shape ) 
    result = np.zeros( (shape[0], shape[1]), dtype = dt )
    
    t0 = time.time()
    output = mp.Queue()
    processes = []
    
    #Launch n_proc processes
    for i in range(n_proc):
        low =  i*shape[1]//n_proc
        high = (i+1)*shape[1]//n_proc
        p = mp.Process(target = mp_wrapper2, args = ((low,high), [data[:,low:high,:], x, target[:,low:high], par], output))
        processes.append(p)
    
    for p in processes:
        p.start()
    
    #Combine the output
    for p in processes:
#        tmp = output.get()
        tmp = my_queue_get(output)
        for i in range(par.size+1):
            result[dt[i][0]][:, tmp[0][0]:tmp[0][1]] = tmp[1][:,:,i]

 
    #Wait for finish    
    for p in processes:
        p.join()
    
    print( "Fitting done in: ", time.time()-t0, "s")
    return result