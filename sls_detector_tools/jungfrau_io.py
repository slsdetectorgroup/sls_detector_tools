#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:42:50 2018

@author: l_frojdh
"""
import os
import numpy as np
import time
import bottleneck as bn
import multiprocessing as mp

nrows = 512
ncols = 1024
ngain = 3
nrow_himac = 256 #half module broken
bitmask = np.array([0x3fff], dtype = np.uint16)
header_size = 112           #bytes
frame_size = nrows*ncols*2 #bytes

def load_file(fname, 
              n_frames, 
              skipframes = 0,
              roi = (slice(0,512,1), slice(0,1024,1))):
    nr = (roi[0].stop-roi[0].start)//roi[0].step
    nc = (roi[1].stop-roi[1].start)//roi[1].step
    t0 = time.time()
    print(f'Loading: {fname}')
    image = np.zeros((n_frames,nr, nc), dtype = np.uint16)
    with open(fname, 'rb') as f:
        f.seek( skipframes * (header_size+frame_size) )
        for i in range(n_frames):
            if i%2000==0:
                print(f'Current frame: {i}')
            f.read(header_size) #frame header TODO! inspect
            try:
                image[i] = np.fromfile(f, dtype = np.uint16, count = nrows*ncols).reshape((nrows,ncols))[roi]
                image[i][image[i]==0xFFFF] == np.nan
            except ValueError:
                print(f'File ended after {i} frames')
                break
    print(f'Loaded {n_frames} frames in {time.time()-t0:.2f}s')
    return image

def load_file_with_calibration(fname, 
                               n_frames, 
                               pedestal, 
                               calibration,
                               skipframes=0,
                               roi = (slice(0,512,1), slice(0,1024,1)),
                               dt = np.double):
    t0 = time.time()
    print(f'Processing: {fname}')
#    image = np.zeros((nrows, ncols), dtype = np.float32)
    
    nr = (roi[0].stop-roi[0].start)//roi[0].step
    nc = (roi[1].stop-roi[1].start)//roi[1].step
    
    calibrated_images = np.zeros((n_frames,nr, nc), dtype = dt)
    
    with open(fname, 'rb') as f:
        f.seek( skipframes * (header_size+frame_size) )
        for i in range(n_frames):
            if i%2000==0:
                print(f'Current frame: {i}')
            f.read(header_size) #frame header TODO! inspect
            try:
                raw_image = np.fromfile(f, dtype = np.uint16, count = ncols*nrows).reshape((nrows,ncols))
            except ValueError:
                print(f'File ended after {i} frames')
                break
            mask = raw_image==0xFFFF
            gain = np.right_shift(raw_image, 14)
            gain[gain==3]=2
            image = np.bitwise_and(raw_image, bitmask).astype(dt)
            for j in range(3):
                image[gain==j]  -= pedestal[:,:,j][gain==j]
                image[gain==j] /= calibration[:,:,j][gain==j]
#                image[236,166]=0;
            image[mask] = 0
            calibrated_images[i]=image[roi]


    print(f'Processed {n_frames} frames in {time.time()-t0:.2f}s') 
    return calibrated_images


def load_calibration(filepath):
    calibration = np.zeros((nrows, ncols, 3))
    with open(filepath) as f:
        for i in range(3):
            calibration[:,:,i] = np.fromfile(f, count = nrows*ncols, dtype = np.double).reshape((nrows, ncols))
    return calibration

def find_frames_with_events(fname, n_frames, pedestal, calibration,
                            threshold=100, skipframes=100):
    t0 = time.time()
    print(f'Processing: {fname}')
    image = np.zeros((nrows, ncols))
    
    #lists for return values
    event_images = []
    raw_event_images= []
    gain_images= []
    
    with open(fname, 'rb') as f:
        f.seek( skipframes * (header_size+frame_size) )
        for i in range(n_frames):
            if i%2000==0:
                print(f'Current frame: {i}')
            f.read(48) #frame header TODO! inspect
            try:
                raw_image = np.fromfile(f, dtype = np.uint16, count = 1024*512).reshape((512,1024))[0:256]
            except ValueError:
                print(f'File ended after {i} frames')
                break
            gain = np.right_shift(raw_image, 14)
            gain[gain==3]=2
            image = np.bitwise_and(raw_image, bitmask).astype(np.double)
            for j in range(3):
                image[gain==j]  -= pedestal[:,:,j][gain==j]
                image[gain==j] /= calibration[:,:,j][gain==j]
                image[236,166]=0;
            if image[200:, 10:].max()>threshold:
#                 print(i)
                event_images.append(image)
                raw_event_images.append(raw_image)
                gain_images.append(gain)
    image = np.asarray(event_images)
    gain = np.asarray(gain_images)
    raw_image  = np.asarray(raw_event_images, dtype = np.uint16)

    print(f'Processed {n_frames} frames in {time.time()-t0:.2f}s') 
    print(f'Found {image.shape[0]} frames with events')
    return image, gain, raw_image

def find_frames_with_events_fixgain(fname, n_frames, 
                                    pedestal, calibration,
                                    threshold = 100,
                                    skipframes = 100, 
                                    gain_idx = 0):
    t0 = time.time()
    print(f'Processing: {fname}')
    image = np.zeros((nrows, ncols))
    
    #lists for return values
    event_images = []
    raw_event_images= []
    gain_images= []
    
    with open(fname, 'rb') as f:
        f.seek( skipframes * (header_size+frame_size) )
        for i in range(n_frames):
            if i%2000==0:
                print(f'Current frame: {i}')
            f.read(48) #frame header TODO! inspect
            try:
                raw_image = np.fromfile(f, dtype = np.uint16, count = nrows*ncols).reshape((nrows,ncols))[0:256]
            except ValueError:
                print(f'File ended after {i} frames')
                break
            gain = np.right_shift(raw_image, 14)
            gain[gain==3]=2
            image = np.bitwise_and(raw_image, bitmask).astype(np.double)

            image  -= pedestal[:,:,gain_idx]
            image /= -calibration[:,:,gain_idx]
            image[236,166]=0;
            image[:,640:704]=0
            if image[200:, 10:].max()>threshold:
#                 print(i)
                event_images.append(image)
                raw_event_images.append(raw_image)
                gain_images.append(gain)
    image = np.asarray(event_images)
    gain = np.asarray(gain_images)
    raw_image  = np.asarray(raw_event_images, dtype = np.uint16)

    print(f'Processed {n_frames} frames in {time.time()-t0:.2f}s') 
    print(f'Found {image.shape[0]} frames with events')
    return image, gain, raw_image


def sum_gain_bits(fname, n_frames, skipframes = 100):
    t0 = time.time()
    print(f'Processing: {fname}')
    image = np.zeros((256, ncols))
    
    with open(fname, 'rb') as f:
        f.seek( skipframes * (header_size+frame_size) )
        for i in range(n_frames):
            if i%2000==0:
                print(f'Current frame: {i}')
            f.read(header_size) #frame header TODO! inspect
            try:
                raw_image = np.fromfile(f, dtype = np.uint16, count = nrows*ncols).reshape((nrows,ncols))[0:256]
            except ValueError:
                print(f'File ended after {i} frames')
                break
            gain = np.right_shift(raw_image, 14)
            gain[gain==3]=2
            image += gain 


    print(f'Processed {n_frames} frames in {time.time()-t0:.2f}s') 
    print(f'Found {image.shape[0]} frames with events')
    return image

def correct_data(data, threshold = 2):
    for frame in data:
        corr_image = np.copy(frame)
        corr_image[corr_image>threshold] = np.nan
        row_correction = bn.nanmean(corr_image, axis = 1).reshape((frame.shape[0],1))
        col_correction_bottom = bn.nanmean(corr_image[0:256,:]-row_correction[0:256], axis = 0).reshape((1,frame.shape[1]))
        col_correction_top = bn.nanmean(corr_image[256:,:]-row_correction[256:], axis = 0).reshape((1,frame.shape[1]))
        frame -= row_correction
        frame[0:256,:] -= col_correction_bottom
        frame[256:,:] -= col_correction_top
    return data


def mp_calculate_pedestal(output, gain_id):
    n_frames_pedestal = 1000
    data = load_file(f'raw/pedestal_d0_f000000000000_{gain_id}.raw', n_frames_pedestal, skipframes = 100)
    gain = np.right_shift(data[0], 14)
    gain[gain==3]=2
    np.bitwise_and(data, bitmask, out=data)
#    imshow(gain)
    pd = data.mean(axis = 2)
    output.put((pd, gain_id))
    
def calculate_pedestal():
    t0 = time.time()
    pedestal = np.zeros((nrows, ncols, 3))
    output = mp.Queue()
    processes = []
    for i in range(3):
        p = mp.Process(target = mp_calculate_pedestal, args = (output, i))
        processes.append(p)
        
    for p in processes:
        p.start()
        
    for p in processes:
        tmp, gain_id = output.get()
        pedestal[:,:,gain_id] = tmp    
        
    for p in processes:
        p.join()
    print(f'Pedestal and noise calculated in {time.time()-t0:.2f}s')
    return pedestal


class Rollover_aware_file:
    """
    Class to read jungfrau calibration files as one file
    """
    header_dt = np.dtype([
        ('frame_number', np.uint64),
        ('bunch_id', np.uint64)
        ])

    
    def __init__(self, base_name, file_index, 
                 order = 'C',
                 nframes_pedestal = 640):
        self._nframes_pedestal = nframes_pedestal
        self._frame_size = self.header_dt.itemsize + 2 * nrows * ncols
        self._base_name = base_name
        self._file_index = file_index
        self._order = order
        self._open()
        self._slice = (slice(0,256,1),(slice(300,640,1)))
        
    def _open(self):
        self._file_handle = open(f'{self._base_name}_{self._file_index:06d}.dat', 'rb')
        self._file_size = os.path.getsize(self.current_file)
        
    def open_next(self):
        self._file_handle.close()
        self._file_index += 1
        self._open()
        
    def close(self):
        self._file_handle.close()
        
    @property
    def current_file(self):
        return self._file_handle.name
    
    @property
    def current_index(self):
        return self._file_index

    def read_frames(self, n_frames):
#        t0 = time.time()
        data = np.zeros((n_frames,
                         self._slice[0].stop-self._slice[0].start, 
                         self._slice[1].stop-self._slice[1].start, 
                         ), dtype = np.uint16, order = self._order)
        header = np.zeros(n_frames, dtype = self.header_dt)
        for i in range(n_frames):
            if self._file_handle.tell()==self._file_size:
                print('Opening next file!!!')
                self.open_next()
            header[i] = np.fromfile(self._file_handle, dtype = self.header_dt, count = 1)
            data[i, :,:] = np.fromfile(self._file_handle, dtype = np.uint16, count = nrows*ncols).reshape(nrows, ncols)[self._slice[0],self._slice[1]]  
#        print(f'load_frames: {time.time()-t0:.2f}')
        return data, header
    
    @property
    def current_frame(self):
        return self._file_handle.tell()/self._frame_size
    
    def seek(self, nframes):
        self._file_handle.seek(nframes * self._frame_size)

    def read_pedestal(self):
        print('Starting to read pedestals')
        t0 = time.time()
        pedestal = np.zeros((nrow_himac,ncols, ngain))
        noise = np.zeros((nrow_himac,ncols,ngain))
        
        for i in range(ngain):
            data, header = self.read_frames(self._nframes_pedestal)
            if not np.all(np.diff(header['frame_number'])==1):
                raise ValueError('Invalid data, lost frames')
            data = np.bitwise_and(data, bitmask).astype(np.double) #Throw away gain data
            pedestal[:,:,i] = data.mean(axis = 2)
            noise[:, :, i] = data.std(axis = 2)
        print(f'Pedestals read in: {time.time()-t0:.2f}s')
        return pedestal, noise
    
    def read_cs_point(self):
        image = np.zeros((nrow_himac, ncol))
        gain = np.zeros((nrow_himac, ncol))
        data, header = self.read_frames(64)
        raw_gain = np.right_shift(data, 14)
        data = np.bitwise_and(data, jf.bitmask).astype(np.double)
        for i in range(64):
            image[:,i::64] = data[:,i::64,i]
            gain[:,i::64] = raw_gain[:,i::64,i]
        return image, gain
    


region = {'Silicon': (slice(None,None,None), slice(256,512,1), slice(256,512,1)),
          'CdTe': (slice(None,None,None), slice(256,512,1), slice(0,256,1)),
          'GaAs': (slice(None,None,None), slice(256,512,1), slice(768,1024,1))}

polarity = {'Silicon': 1, 'CdTe': -1, 'GaAs': -1}