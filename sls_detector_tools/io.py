# -*- coding: utf-8 -*-
"""
File io for the EIGER detector. General io that have not been changed is 
kept in this file. While relese specific versions can be found in the v18 etc. 
files. This numbering corresponds loosely to the firmware versions.

Current io version by default imported but can be changed in the config file

 * v18 - old style with ascii header
 * v19 - master file, no header
 * v20 - modified master file?

"""
#Python imports
from __future__ import print_function
import numpy as np
import logging
from . import config as cfg
from collections import OrderedDict

#Logger to record progress of data processing
logger = logging.getLogger()

    
def write_trimbit_file( fname, data, dacs, ndacs = 18 ):
    """
    Write trimbits and dacs to a file. The function checkes that you supply
    18 dacs and 256*1024 (halfmodule) trimbits. But you can override the 
    number of dacs if you are writing files for an older software.
    
    Parameters
    ----------
    fname: str
        Filename
    data: numpy_array
        Array wihth the trimbits
    dacs: numpy_array
        Array with the dacs
    ndacs: int
        The number of dacs to write to the trimbit file. Defaults to 18 which 
        is the current value. But was 16
    
    Raises
    -------
    ValueError
        If the size of the dacs is not the specified one
    ValueError
        If the number of trimbits are not 256*1024 since everything is handled
        as a half module in the slsDetectorSoftware
    
    """
    
    if dacs.size != ndacs:
        raise ValueError('The array of dacs.size: {:d} is different from'\
                         ' the specified ndacs: {:d}'.format(dacs.size, ndacs))
    if data.size != 256*1024:
        raise ValueError('data contains: {:d} elements instead of the'\
                         ' expected: {:d}'.format(data.size, 256*1024))
    
    tb = data.astype( np.int32 )
    dacs = dacs.astype( np.int32 )
    with open(fname, 'wb') as f:
        f.write( dacs.tobytes() )
        f.write( tb.tobytes() )


def read_trimbit_file( fname, ndacs = 18):
    """
    Read a trimbit file and return the trimbits and dacs
    
    Parameters
    ---------
    fname: str
        Filename for the trimbit file
    ndacs: int, optional
        Number of dacs to expect in the trimbit file . Defaults to 18.
        *Note that for old files it used to be 16 dacs8
    
    Returns
    -------
    tb: numpy_array[row, col]
        2d array with trimbits
    dacs: numpy_array[ndacs]
        dacs stored in the trimbit file
        
    Raises
    -------
    ValueError
        If the file size is wrong
   
     
    **Fileformat** ::
        
        dacs     : (int32)*ndacs
        trimbits : (int32)*256*1024
        
    """
    data = np.fromfile( fname, dtype = np.int32 )   
    
    #Check that the file is ok 
    if data.size-(256*1024+ndacs) != 0:
        raise ValueError('Wrong file size! Expected '+str((256*1024+ndacs)) +' ints and got '+ str(data.size))
    tb = np.reshape(data[ndacs:], (256,1024))
    dacs = data[0:ndacs]
    return tb, dacs
    
    
    
def save_txt(fname, header, data, delimiter = ','):
    """ 
    Save variables in an ascii file function mainly used
    for chip testing
    """
    f = open(fname, 'w')
    for h,v in zip(header, data):
        f.write(h+delimiter)
        for i in v:
            f.write(str(i)+delimiter)
        f.write('\n')
    f.close()

def load_txt(fname, delimiter = ',', data_type = float):
    """ 
    Load variables from an ascii file function mainly used
    for chip testing
    """
    f = open( fname )
    data = []
    for line in f:
        tmp = line.split( delimiter )[1:-1]
        x = [data_type(x) for x in tmp]
        x = np.asarray(x)
        data.append(x)
    return data

class geant4:
    """
    Class that holds the data type for reading x,y,c files from the 
    geant4medipix framework.
    """
    sparse_dt = [('event', np.uint32),
      ('col', np.uint32),
      ('row', np.uint32),
      ('energy', np.double),
      ('tot', np.double),
      ('toa', np.double)]

def read_header( fname ):
    """
    Read the header from the master file
    
    Parameters
    ----------
    fname: str
        Filename of the master file
    
    Returns
    --------
    header: dict
        Dictionary with the fields in the header file
    """
    #Read all lines from the master file 
    with open(fname) as f:
        tmp = f.readlines()

    #Print the file if we are in debug mode
    if cfg.debug is True:
        for l in tmp:
            print(l)

    #Put information in the dict
    header = OrderedDict()
    for i in range(11):
        field = tmp[i][0:19].strip(' ')
        value = tmp[i][22:].strip('\n').strip(' ')
        if field == 'Dynamic Range':
            value = int(value)
        header[ field ] = value


    return header


def read_frame_header( fname ):
    """
    Read and decode the frame header of a raw file
    """
    with open( fname ) as f:
        tmp = f.read( 48 )
    fh = struct.unpack('Q2I2Q4HIH2B',tmp)
    return fh
    

def read_frame(f, dr):
    """
    Reads a single frame from an open file 
    """
    
    #Skip the frame header
    f.seek(48, 1)

    
    if dr in [8,16,32]:
        dt = np.dtype( 'uint{:d}'.format(dr) )
        return np.fromfile(f, dtype = dt, count = 256*512).reshape((256,512))
        
    elif dr == 4:
        dt = np.dtype('uint8')
        tmp = np.fromfile(f, dtype = dt, count = 256*256)
        data = np.zeros( tmp.size * 2, dtype = tmp.dtype )
        data[0::2] = np.bitwise_and(tmp, 0x0f)
        data[1::2] = np.bitwise_and(tmp >> 4, 0x0f)
        print('shape', data.shape)
        return data.reshape((256,512))

def load_file(fname, header, N = 1):
    """
    Load EIGER raw file and return image data as numpy 2D array
    To be used as base for loading frames for single and multi module systems 
    """
    
    #Assign correct dynamic range
    dr = int( header['Dynamic Range'] )
    if dr in [8,16,32]:
        dt = np.dtype( 'uint{:d}'.format(dr) )
    elif dr == 4:
        dt = np.dtype( 'uint8' )
    else:
        raise TypeError('Unknown dynamic range')

    if cfg.verbose:
        print('sls_detecot/io/load_file: Loading ', fname)
        
   #Open file and read 
    with open( fname, 'rb' ) as f:
        if N == 1:    
            image = read_frame(f, dr)
        else:
            #Reading more than one frame from the file
            image = np.zeros((256,512, N), dtype = dt)
            for i in xrange(N):        
                image[:,:,i] = read_frame(f, dr)
    return image


def load_frame(bname, run_id, 
               frameindex = -1,
               N = 1, 
               shift = 0, 
               geometry = '500k',
               default = 0):
    """
    Read one or several Eiger frames from a file. Default function to 
    read data form disk.
    
    .. note::
        The function does always return the right size array and only prints
        a warning if the file is missing. This is useful for example for the 
        9M if one or several files are missing. 
    
    Parameters
    -----------
    bname: str
        base name of the file used for consructing master and port files
    frameindex: int
        index of the starting file for multi image files
        None if not used. Changes the pattern of the filename
    N: int 
        Number of frames to read
    shift: int
        Determines which d number to start reading at. Can be used to read
        parts of a larger detector
    geometry: str
        the geometry of the detector. Name identifies both number of pixels
        and layout.
    default: number
        default value if the pixels are not filled. Can be used to highligt 
        missing data
        
    Returns
    --------
    data: numpu_array
        data[row, col, N], of the same type as in the file or some cases 
        double.
    """
    #Construct ending of filename depending on index or not
    if frameindex == -1:
        fname_end =  '_%d.raw' % run_id
    else:
        fname_end =  '_f{:012d}_{:d}.raw'.format( frameindex, run_id )

    header = read_header('{:s}_master_{:d}.raw'.format(bname, run_id))

    if geometry == '500k':
        if N == 1:
            # Should we control the data type more strictly?
            image = np.full((512, 1024), default)

            try:
                fname = bname + '_d'+ str(shift) + fname_end
                image[256:512,0:512] = load_file( fname, header )
            except IOError:
                print('Missing file: ', fname)
                
            try:
                fname = bname + '_d'+ str(shift + 1) + fname_end
                image[256:512,512:1024] = load_file( fname, header )    
            except IOError:
                print('Missing file: ', fname)
            try:
                fname = bname + '_d'+ str(shift + 2 ) + fname_end
                image[0:256,0:512] = np.flipud( load_file( fname, header ) )
            except IOError:
                print('Missing file: ', fname)
            try:
                fname = bname + '_d'+ str(shift + 3) + fname_end
                image[0:256,512:1024] = np.flipud( load_file( fname, header ) )     
            except IOError:
                print('Missing file: ', fname)
            
        else:
            #0 constructing from type of returned data, use header instead
            fname = bname + '_d'+str(shift) + fname_end
            tmp = load_file(fname, header, N = N )        
            image = np.full((512, 1024, N), default, dtype = tmp.dtype)
            image[256:512,0:512,:] = tmp
            
            #1
            fname = bname + '_d'+str(shift +1 ) + fname_end
            image[256:512,512:1024, :] = load_file( fname, header, N = N )

            #2
            fname = bname + '_d'+str(shift +2 ) + fname_end
            image[0:256,0:512, :] = np.flipud( load_file( fname, header, N = N ) )
            
            #3
            fname = bname + '_d'+str(shift +3 ) + fname_end
            image[0:256,512:1024, :] = np.flipud( load_file( fname,header, N = N ) )
            
        return image

    elif geometry == '250k':
        if N == 1:
            #TODO! Fix dt
            image = np.full((256, 1024), default)
            fname = bname + '_d'+ str(shift ) + fname_end
            image[0:256,0:512] = np.flipud( load_file( fname ) )
            fname = bname + '_d'+ str(shift + 1) + fname_end
            image[0:256,512:1024] = np.flipud( load_file( fname ) )              
            
        else:
            raise NotImplementedError('Programmer was lazy')
            
        return image

        
    elif geometry == '9M':
        #Recursive calling for multi module systems
        #TODO verify with latest version of the software
        modules = range(0,72,4)
        if N == 1:
            image = np.full((3072,3072), default)
            image[:] = -1 
            for i, s in enumerate( modules ):
                try:
                    image[ mask.detector[geometry].module[i] ] = load_frame(
                                                      bname, run_id, shift = s)
                except (IOError, ValueError) as err:
                    print( err )
                    print('LOST FRAME IN: {:s}, {:d}, {:d}'.format(bname, 
                                                                   run_id, s))
            return image
        else:
            raise NotImplementedError('Multi frame files for 9M is not yet implemented')
