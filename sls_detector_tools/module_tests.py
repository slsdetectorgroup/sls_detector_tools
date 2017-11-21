import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import time 
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable



from .io import load_frame, save_txt, load_txt
from .plot import *
from . import root_helper as r
from . import config as cfg

plt.ion()
sns.set()

def rx_bias(detector, clk = 'Full Speed', npulse = 1, plot = False):
    """
    Scans rx bias and checks for each value if the data is read
    out correctly. Uses testing with enable to load counter values. 
    Default number of pulses is 1. Probe station testing uses 721 pulses
    721 (1444 counter value)
    
    """

    
    #Output config
    out = cfg.path.out
    path = os.path.join( cfg.path.test, cfg.det_id )
    tmp_fname = 'test'
    
    detector.file_name = tmp_fname
    detector.file_write = True 
    detector.dynamic_range = 16 
    detector.readout_clock = clk


    detector.exposure_time = 0.01 

    #Save dacs to recover them later    
    dacs = detector.dacs.get_asarray()  
    
    #Set dacs for testing mode
    detector.vthreshold = 4000
    detector.dacs.vtr = 4000
    print( detector.dacs )    
    

    #should use masks!
    rows = [slice(256,512,1), slice(0,256,1)]
    cols = [slice(0,256,1), slice(256,512,1), slice(512,768,1), slice(768,1024,1)]
    chips = [[r,c] for r in rows for c in cols]

    #Scan range 
#    rx_list = np.arange(500,1801,100)
    rxb_values = list(range(500,1801,100))
    N = np.zeros((8, len(rxb_values)))
    print( "rx_bias test at: ", detector.readout_clock)
    t0 = time.time()
    
    data_list = []
    for i,rx in enumerate(rxb_values):
        detector.dacs.rxb_lb= rx
        detector.dacs.rxb_rb = rx
        detector.file_index = 0
        detector.pulse_chip(npulse)
        detector.acq()
#        print(detector.dacs.rxb_rb)
#        print(detector.dacs.rxb_lb)
#        print(rx)

        data = load_frame(os.path.join(out, tmp_fname), 0, geometry = cfg.geometry) 
        data_list.append(data)
        for j,c in enumerate(chips):
            N[j, i] = (data[c] != int(npulse*2+2)).sum()
#
#    sys.stdout.write(' ' + str(time.time()-t0))
#    sys.stdout.write('\n')


    
    #Clean up last files
    os.remove( os.path.join(out, tmp_fname)+'_d0_0.raw' )
    os.remove( os.path.join(out, tmp_fname)+'_d1_0.raw' )
    os.remove( os.path.join(out, tmp_fname)+'_d2_0.raw' )
    os.remove( os.path.join(out, tmp_fname)+'_d3_0.raw' )



    #plot rx bias
    if plot:
        plt.figure()
        for i in range(8):
            plt.plot(rxb_values, N[i], 'o-')
        plt.xlabel('rx_bias [DAC code]')
        plt.ylabel('number of pixels [1]')
        plt.title('RX bias test, clk: ' + str(clk))

    #Save data for report
    header = ['rx_bias', 'chip0', 'chip1', 'chip2', 'chip3', 'chip4', 'chip5', 'chip6', 'chip7']
#    fname = os.path.join(path, cfg.det_id+'_rxbias_'+str(clk)+'.txt')
    n = detector._speed_int[ detector.readout_clock ]
    fname = os.path.join(path, '{:s}_rxbias_{:d}.txt'.format(cfg.det_id, n))
    save_txt(fname, header, [rxb_values] + [x for x in N])

    
    #Reset to previous state
    detector.pulse_chip(-1)
    detector.dacs.set_from_array( dacs )
    return rxb_values, N

def io_delay( name, d, clk = 1, npulse = -1, plot = False ):
    """
    Test to determine the settings for iodelay
    npulses = -1 makes the test using a normal readout
    npulses = 0 puts the threshold on the far side of noise
                but does not pulse 
                
    npulse > 0 pulses using enable
    """

    if cfg.nmod == 1:
        geo = '250k'
    else:
        geo = '500k'
    
    
    #Output setup
    out = cfg.path.out
    path = os.path.join( cfg.path.test,name )
    tmp_fname = 'test'
    d.set_fname( tmp_fname )
    d.set_fwrite( True )
    
    d.set_dr( 16 )
    d.set_clkdivider(clk)
    d.set_exptime( 0.01 )    
    
    #Save dacs
    dacs = d.dacs.get_asarray()
    iodelay = d.get_dac('iodelay')
    
    #If we have pulses
    if npulse > 0:
        d.set_threshold(4000)
        d.dacs['vtr'] = 4000
    else:
        d.set_threshold(1500)
        
    print( d.dacs )
    
    #Generate slices to access chip data
    rows = [slice(256,512,1), slice(0,256,1)]
    cols = [slice(0,256,1), slice(256,512,1), slice(512,768,1), slice(768,1024,1)]
    chips = [[r,c] for r in rows for c in cols]

    #Scan range 
#    io_list = np.arange(550,851,5)
    io_list = np.arange(550,851,5)
    N = np.zeros((8, io_list.size))
    t0 = time.time()
    print( "iodelay test at clkdivider", clk )   
    
    #Iodelay scan
    for i,io in enumerate(io_list):
        if not i % (len(io_list)/10):
            sys.stdout.write('=')
            
        d.set_dac('iodelay', io)
        d.set_index(0)
        
        if npulse > 0:
            print( npulse )
            d.pulsechip( npulse )
        d.acq()
        data = load_frame(os.path.join(out, tmp_fname), 0, geometry = geo) #Single frame

        for j,c in enumerate(chips):
            if npulse >= 0:
                N[j, i] = (data[c] != int(npulse*2+2)).sum()
                print( data.mean() )
            else:
                N[j, i] = (data[c] != 0).sum()
    
    sys.stdout.write(' ' + str(time.time()-t0))
    sys.stdout.write('\n')
    
    #Plotting results 
    if plot:
        plt.figure()
        for i in range(8):
            plt.plot(io_list, N[i], 'o-')
        plt.xlabel('io delay [DAC code]')
        plt.ylabel('nr of pixels [1]')
        plt.title('io delay scan, clk: ' + str(clk))

    #Saving result
    header = ['iodelay', 'chip0', 'chip1', 'chip2', 'chip3', 'chip4', 'chip5', 'chip6', 'chip7']
    fname = os.path.join(path, name+'_iodelay_'+str(clk)+'.txt')
    save_txt(fname, header, [io_list] + [x for x in N])

    #Clean up last files
    os.remove( os.path.join(out, tmp_fname)+'_d0_0.raw' )
    os.remove( os.path.join(out, tmp_fname)+'_d1_0.raw' )
    
    #Reset    
    d.pulsechip(-1)
    d.set_dac('iodelay', iodelay)
    d.dacs.set_asarray( dacs )

    return io_list, N

def analog_pulses(name, d, clk = 1, N = 1000, plot = False):
    """
    Test the analog side of the pixel using test pulses
    Normally this test is run only using clock divider 1
    It is normal to see cosmics in the final image
    """
    print( 'analog_pulses()' )
    print( "Analog pulsing with clkdivider", clk )
    
    #Output setup
    out = cfg.path.out
    path = os.path.join( cfg.path.test,name )
    tmp_fname = 'test'
    d.set_fname( tmp_fname )
    d.set_fwrite( True )
    d.set_clkdivider(clk)
    d.set_dr( 16 )
    d.set_exptime( 0.01 )        
    
    
    #Get dacs
    dacs = d.dacs.get_asarray()
    d.set_all_trimbits(63)
    d.dacs['vtr'] = 2600
    d.dacs['vrf'] = 2900
    d.dacs['vcall'] = 3600
    d.set_threshold(1500)    

    print( d.dacs )
    t0 = time.time()
    d.set_index(0)
    d.pulse_all(N = N)
    d.acq()
    data = load_frame(os.path.join(out, tmp_fname), 0, geometry = cfg.geometry)
    print( time.time()-t0 )
    np.savetxt(os.path.join(path, name+'_pulse.txt'), data)

    if plot:
        ax, im = imshow( data )
        im.set_clim(N-1, N+1)
        plt.draw()
    
    #Reset dacs
    d.dacs.set_asarray( dacs )
    return data

def counter(name, d, clk = 1):
    """
    Test the digital counter logic. 
    Note this does not test the overflow
    
    Test is done by toggeling enable:
    1364 --> 2730 --> 101010101010
    682 --> 1366 --> 010101010110
    
    Using enable only increments of 2 in the 
    counter value is possible
    """
    print( "Counter tests with clkdivider", clk )
    
    #Keep bad pixels form the tests
    bad_pixels = np.zeros((512,1024), dtype = np.bool)    
    
    #Output setup
    out = cfg.path.out
    path = os.path.join( cfg.path.test,name )
    tmp_fname = 'test'
    d.set_fname( tmp_fname )
    d.set_fwrite( True )
    d.set_clkdivider(clk)
    d.set_dr( 16 )
    d.set_exptime( 0.01 )  
    
    dacs = d.dacs.get_asarray()
    
    #To enter test mode
    d.set_threshold(4000)
    d.dacs['vtr'] = 4000
    d.set_index(0)
    
    #Test 1
    d.pulsechip(1364)
    d.acq()
    data = load_frame(os.path.join(out, tmp_fname), 0, geometry = cfg.geometry)
    print( 'Bad pixels:', (data != 2730).sum(), '\n' )
    bad_pixels[data != 2730] = True
    
    #Test 2
    d.set_index(0)
    d.pulsechip(682)
    d.acq()
    data = load_frame(os.path.join(out, tmp_fname), 0, geometry = cfg.geometry)
    print( 'Bad pixels:', (data != 1366).sum(), '\n')  
    bad_pixels[data != 1366] = True
    
    
    #Output
    tmp = np.where(bad_pixels == True)
    fname = os.path.join(path, name+'_counter_'+str(clk)+'.txt')
    save_txt(fname, ['row', 'col'], tmp)

    #Reset
    d.dacs.set_asarray( dacs )
    d.pulsechip( -1 )
    
    return bad_pixels

    

def overflow(name, d, clk = 1):
    """
    Overflow test using analog pulses
    
    Unfortunatley the enable toggling cannot be used to check this because
    of problems when going to overflow
    """
    bad_pixels = np.zeros((512,1024), dtype = np.bool)       
    
    #Output setup
    out = cfg.path.out
    path = os.path.join( cfg.path.test,name )
    tmp_fname = 'test'
    d.set_fname( tmp_fname )
    d.set_fwrite( True )
    d.set_clkdivider(clk)
    d.set_dr( 16 )
    d.set_exptime( 0.01 )       
    
    
    
    #Get dacs
    dacs = d.dacs.get_asarray()
    d.set_all_trimbits(63)
    d.dacs['vtr'] = 2600
    d.dacs['vrf'] = 2900
    d.dacs['vcall'] = 3600
    d.set_threshold(1500)    

    print( d.dacs )
    
    # 4 bit mode 
    t0 = time.time()  
    d.set_index(0)
    d.set_dr(4)
    d.pulse_all(N = 20)
    d.acq()
    data = load_frame(os.path.join(out, tmp_fname), 0)
    print( '4 bit test found', (data != 15).sum() , 'bad pixels in', time.time()-t0, 's' )
    bad_pixels[data != 15] = True
    
    # 8 bit mode
    t0 = time.time()   
    d.set_index(0)
    d.set_dr(8)
    d.pulse_all(N = 260)
    d.acq()
    data = load_frame(os.path.join(out, tmp_fname), 0)
    print( '8 bit test found', (data != 255).sum() , 'bad pixels in', time.time()-t0, 's')
    bad_pixels[data != 255] = True
    
    # 12 bit mode
    t0 = time.time()   
    d.set_index(0)
    d.set_dr(16)
    d.pulse_all(N = 4100)
    d.acq()
    data = load_frame(os.path.join(out, tmp_fname), 0)
    print( '16 bit test found', (data != 4095).sum() , 'bad pixels in', time.time()-t0, 's')
    bad_pixels[data != 4095] = True
    

    #Output
    tmp = np.where(bad_pixels == True)
    fname = os.path.join(path, name+'_overflow_'+str(clk)+'.txt')
    save_txt(fname, ['row', 'col'], tmp)
    
    return bad_pixels
    

def tp_scurve( name, d, clk = 1 ):


    #Output setup
    out = d.config.get('Python', 'sls_data_path')
    path = os.path.join( d.config.get('Python', 'module_test_path'),name )
    tmp_fname = 'test'
    d.set_fname( tmp_fname )
    d.set_fwrite( True )
    d.set_clkdivider(clk)
    d.set_dr( 16 )
    d.set_exptime( 0.01 )        
    d.set_index(0)
    
    #Get dacs
    dacs = d.dacs.get_asarray()
    d.set_all_trimbits(63)
    d.dacs['vtr'] = 2600
    d.dacs['vrf'] = 2900
    d.dacs['vcall'] = 3700
    d.set_threshold(1500)  
    
    th_list = range(0,2000,50)
    data = np.zeros((len(th_list), 512,1024))
    for i, th in enumerate(th_list):    
        print( th )
        d.set_threshold( th )
        d.dacs['vcp'] = th
        d.pulse(N = 1000)
        d.acq()
        data[i] = load_frame(os.path.join(out, tmp_fname), i, tengiga = False, bitdepth = 16)
        
    plt.figure()
    plt.plot(th_list, data.sum(axis = 1).sum(axis = 1))
    return data
        

def generate_report(path):
    """
    Generate a report based on the contents in a folder
    outputs a multipage pdf file
    """
    os.chdir( path )
    name = os.path.split( path )[1]
    pdf = PdfPages( name + '_test.pdf' )
    
    #--------------------------------------------------------------------page 1
    #rx bias 
    plt.figure(num=None, figsize=(11.69, 8.27), dpi=100)
    plt.suptitle('Eiger Module Test '+name, size = 32)

    rxb = []
    gs = gridspec.GridSpec(3, 3,
                       width_ratios=[2,2,1],
                       height_ratios=[1,5,5]
                       )
    for clk in range(2):
        try:
            ax = plt.subplot(gs[clk+3])
            data = load_txt( os.path.join(path, name+'_rxbias_'+str(clk)+'.txt'))
            rxbias = data[0]
            
            for i in range(1,9,1):
                    plt.plot(rxbias, data[i], 'o-', label = 'chip '+str(i-1))
            for i in range(1,9,2):
                    N = data[i] + data[i+1]
                    rx = rxbias[N<(N.min()+5) ].mean()
                    print( int(rx) )
                    rxb.append(int(rx))

            plt.legend(loc = 'lower right', fontsize = 10)
            plt.xlabel('rx bias')
            plt.ylabel('Bad pixels')
            plt.yscale('log')
            if clk == 0:
                plt.title('rx_bias test full speed')
            elif clk==1:
                plt.title('rx_bias test half speed')


        except IOError:
            print( 'Error in RX bias test file not found' )
            pass

    #Formatting of rxbias labels
    hpos = 1.15
    pos = 0.95
    step = -0.09
    labels = ['0:rx_lb', '0:rx_rb','1:rx_lb', '1:rx_rb']*2
    for i,rx,l in zip(range(len(rxb)),rxb, labels):
        if i == 0:
            plt.text(hpos-.05, pos,'full speed',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'bold')
            pos += step

        elif i == 4:
            pos += step
            plt.text(hpos-.05, pos,'half speed',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'bold')
            pos += step
        plt.text(hpos, pos,l+': '+str(rx),
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes)
        pos += step


    #Counter tests
    sns.set_style('white')
    bad = [0,0]
    labels = ['counter test full speed', 'counter test half speed']
    for i in range(2):
        try:
            ax = plt.subplot(gs[6+i])  
            image = np.zeros((512,1024))
            tmp = load_txt( os.path.join(path, name+'_counter_'+str(i)+'.txt'), data_type = int)
            pixels = list( zip(tmp[0], tmp[1]))
            for p in pixels:
                print(p)
                image[p] = 1
            ax.imshow(image, interpolation = 'nearest', origin = 'lower')
            plt.title(labels[i])
            bad[i] = len(pixels)
        except IOError:
            print( "hej" )
            pass
    
    hpos = 1.15
    pos = 0.7
    step = -0.09
    plt.text(hpos-.05, pos,'full speed',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 weight = 'bold')
    pos += step
    if bad:
        plt.text(hpos, pos,'bad pixels: '+str(bad[0]),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform = ax.transAxes,
                     weight = 'normal')
        pos += 2*step
        plt.text(hpos-.05, pos,'half speed',
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform = ax.transAxes,
                     weight = 'bold')
        pos += step
        plt.text(hpos, pos,'bad pixels: '+str(bad[1]),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform = ax.transAxes,
                     weight = 'normal')

    plt.tight_layout()
    pdf.savefig()
    
    #-------------------------------------------------------------- Page 2
    sns.set_style('darkgrid')
    plt.figure(num=None, figsize=(11.69, 8.27), dpi=100)
    gs = gridspec.GridSpec(2, 4,
                       width_ratios=[1,1,1,1],
                       height_ratios=[1,1.3]
                       )   

    #IO delay tests
    iodelay = []
    for clk in range(2):
        try:
            ax = plt.subplot(gs[clk*2:clk*2+2])
            data = load_txt( os.path.join(path, name+'_iodelay_'+str(clk)+'.txt'))
            delay = data[0]
            
            for i in range(1,9,1):
                    plt.plot(delay, data[i], 'o-', label = 'chip '+str(i-1))
            for i in range(1,9,2):
                    N = data[i] + data[i+1]
                    io = delay[N<(N.min()+5) ].mean()
                    print( int(io) )
                    iodelay.append(int(io))

            plt.legend(loc = 'lower right', fontsize = 10)
            plt.xlabel('iodelay')
            plt.ylabel('Bad pixels')
            plt.yscale('log')
            if clk == 0:
                plt.title('iodelay test full speed')
            elif clk==1:
                plt.title('iodelay test half speed')


        except IOError:
            print( 'Error in iodelay test file not found' )
            pass

    sns.set_style('white')
    image = np.loadtxt(os.path.join(path, name+'_pulse.txt'))
    ax = plt.subplot(gs[4:7])
    im = ax.imshow( image, cmap = 'coolwarm', origin = 'lower', interpolation = 'nearest' )
    im.set_clim(999,1001)
    
    ax.set_title('Analog Test Pulses')    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cb = plt.colorbar(mappable = im, cax = cax)
    
    ax =  plt.subplot(gs[7])
    c,h = r.hist(image, xmin = 0, xmax = 3000, bins = 3000)
    x0,y0 = r.getHist(h)
    ax.plot(x0,y0, ls = 'steps')
    ax.set_xlim(980,1021)
    ax.set_xlabel('Counts [1]')
    ax.set_ylabel('Pixels [1]')    
    
    plt.tight_layout()
    pdf.savefig()
     #-------------------------------------------------------------- Page 3
    sns.set_style('white')
    plt.figure(num=None, figsize=(11.69, 8.27), dpi=100)
    gs = gridspec.GridSpec(2, 2,
                       width_ratios=[3,1],
                       height_ratios=[1,1]
                       )   
                       
    image = np.zeros((512,1024))
    tmp = load_txt( os.path.join(path, name+'_overflow_1.txt'))
    pixels = zip(tmp[0], tmp[1])
    for p in pixels:
        image[p] = 1
        
    ax = plt.subplot(gs[0])
    im = ax.imshow( image, cmap = 'Greys', origin = 'lower', interpolation = 'nearest' )
    im.set_clim(0,1)
    ax.set_title('Overflow tests, Bad pixels: ' + str( len(tmp) ))  
    plt.grid( False )
    pdf.close()
    return ax
