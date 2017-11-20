# -*- coding: utf-8 -*-
"""
Limited support for loading TIFF files from Python
Designed to work with the uXAS Eiger TIFF files
Currently only supports single image files...


.. warning:: 
    This module is not complete!!!
"""
#Python imports
import os
import numpy as np
import struct

debug = True

tag = {256: 'ImageWidth',
       257: 'ImageHeight',
       258: 'BitsPerSample',
       259: 'Compression',
       262: 'PhotometricInterpretation',
       271: 'Make',
       272: 'Model',
       273: 'StripOffsets',
       277: 'SamplesPerPixel',
       278: 'RowsPerStrip',
       279: 'StripByteCounts',
       284: 'PlanarConfiguration',
       305: 'Software',
       339: 'SampleFormat',
       65000: 'Unknown',
       65001: 'Unknown',
       65002: 'Unknown',
       65003: 'Unknown',
       65010: 'Unknown',}


data_type = {1: 'B',    #uint8
             2: 's',    #string
             3: 'H',    #uint16
             4: 'I',     #uint32
             5: 'II',      #rational = 2 x uint32    
             6: 'b',     #int8
             7: 'B',     #undefined uint8
             8: 'h',     #int16
             9: 'i',     #int32
             10: 'ii',    #srational 2 x int32
             11: 'f',    #float
             12: 'd',    #double
             }

data_size = {1: 1,
             2: 1,
             3: 2,
             4: 4,
             5: 8,
             6: 1,
             7: 1,
             8: 2,
             9: 4,
             10: 8,
             11: 4,
             12: 8}

h_size = 8 #TIFF file header size


def load_tiff( fname ):
    
    #Open file
    f = open( fname,  'rb')

    #Read Image File Header
    byte_order, version, offset = struct.unpack('HHI', f.read( h_size ))
    if debug:
        print( '\n--- TIFF Image File Header ---')
        print( 'byte_order: {:d}, version: {:d}, offset: {:d}\n'.format(
                byte_order, version, offset))

    #Go to and read the first Image File Directory
    f.seek( offset )
    n_tags = struct.unpack('H', f.read(2))[0]

    if debug:
        print( 'n_tags: {:d} \n'.format( n_tags ))
        
    #Loop over all tags
    for i in range( n_tags ):
    
        #Read tag
        tag_id, dtype, dcount, doffset = struct.unpack( 'HHII', f.read(12) )  
        
        try:
            if debug:
                print( 'tag_id: {:>5d} type: {:>20s} dtype: {:d} dcount: {:d}'.format( 
                        tag_id, 
                        tag[tag_id],
                        dtype,
                        dcount))

                print( 'doffset/data:', doffset )
    
            #read tag data if we have more than four bits of data
            if data_size[dtype] * dcount > 4:
                p = f.tell()
                f.seek( doffset )
                s = f.read( data_size[dtype] * dcount )  
                tag_data = struct.unpack( str(dcount)+data_type[dtype], s )
    #            print s.strip('\x00')
                print( tag_data )
                f.seek( p )
    
    
    
        except:
            print( 'tag: {:d} not found'.format(tag_id) )
            pass



#f.seek(8)
#data = np.fromfile(f, dtype = np.float64, count = 512*1024).reshape( (512,1024) )
#ax, im = imshow(data)


#Find the data 
#os.chdir('/mnt/disk1/elettra_testbeam/elettra_testbeam/T38/Cr/data/Energy12.8keV/high_wide')
#os.chdir('/home/l_frojdh/python/sls_detector_tools/datasets')


#image = load_frame('data', 0,  bitdepth = 32)

#a = file_info('data_d0_0.raw')

#f = open('data_d0_0.raw')
#data = f.read(500)


##Load and plot
#for T in ['T60']:
#    for v in [150]:
#        image = load_frame('{:s}_MoXRF_{:d}V'.format(T,v), 10)
#        image = fix_large_pixels( image )
#        ax, im = imshow( image )
#        v = 20
#        im.set_clim(0,1500)
#        ax.set_title('{:s} CuXRF {:d}V'.format(T,v))
#        
#        fname = '{:s}_CuXRF_{:d}V'.format(T,v)
#        path = '/afs/psi.ch/project/pilatusXFS/Erik/9M/retest2'
#        pathname = os.path.join(path, fname)
#        plt.savefig( pathname )
##        plt.close()