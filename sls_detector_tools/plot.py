# -*- coding: utf-8 -*-
"""
Plotting routines for displaying image sensor data using matplotlib and
seaborn. Some routines like the chip_histograms rely on PyROOT but the import
should work even without ROOT.
"""
#Print function to be ready for Python3
from __future__ import print_function

#Python imports
from itertools import permutations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


#sls_detector imports
from . import mask
from . import config as cfg
from . import utils
#from sls_detector import function


#Try to import r (root plotting stuff) otherwise fallback on python version
#try:
#    from sls_detector_tools import root_helper as r
#except ImportError:
#    pass
#    print('sls_detector/plot: ROOT version of r not imported! Using python version')
#    from sls_detector import py_r as r


def imshow(data, cmap='coolwarm',
           log=False,
           draw_asics=False,
           asic_color='white',
           asic_linewidth=2,
           figsize=(16, 10)):
    """Plot an image with colorbar

    Parameters
    ----------
    data: 2d numpy_array
        Image data
    cmap: std
        Name of the colormap that should be used for the plot. Default is
        coolwarm
    log: bool
        If True apply a logaritmical colorscale
    draw_asics: bool
        Draw the edges of the asic in the module Warning this currently
        only works for single modules




    Returns
    ----------
    ax: mpl.axes
        Matplotlib axes object
    im: mppl.image
        Matplotlib image

    Raises
    ------
    ValueError
        If the size of x and y is not the same

    """

    #Check for the dimensions of the data
    #We expect [row, col, frame] or [row, col]
    if len(data.shape) == 3:
        print('Warning data contains more than one frame, plotting frame 0')
        data = data[:, :, 0]
    elif len(data.shape) == 2:
        pass
    else:
        raise ValueError('Unknown image dimesion. Expecting [row, col] or',
                         '[row, col, frame] but got shape: ', data.shape)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    if log is True:
        im = ax.imshow(data, interpolation='nearest',
                       origin='lower', cmap=cmap,
                       norm=mpl.colors.LogNorm(vmin=0.1))
    else:
        im = ax.imshow(data, interpolation='nearest',
                       origin='lower', cmap=cmap)


    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    ax.grid(False)

    #Outline ASICs
    if draw_asics is True:
        for x_pos in np.arange(255.5, 1023.5, 256):
            ax.plot([x_pos, x_pos], [0.0, 512.0], '--', color=asic_color,
                    linewidth=asic_linewidth)
        ax.plot([0, 1024], [255.5, 255.5], '--', color=asic_color,
                linewidth=asic_linewidth)
        ax.set_xlim(0, data.shape[1])
        ax.set_ylim(0, data.shape[0])
    plt.colorbar(im, cax=cax)

    return ax, im



def setup_plot():
    """
    Setup seaborn and matplotlib for nice plots. This function uses sns.set()
    then sets style to: *talk* and font scale 1.2 for better readability.
    """
    sns.set()
    sns.set_context('talk', font_scale=1.2)
    sns.set_style('white')
    sns.set_style({'legend.frameon': True})
    plt.ion()

def draw_module_borders(ax, vertical=False, half_module=False):
    """
    Draw lines outlining the individual modules in the Eiger9M. Options as
    well for half modules

    Parameters
    ----------
    ax: mpl.axes
        axes of the image that should be drawn on
    vertical: bool, optional
        Defaults to False, set to True if the image is rotated to vertical
        layout
    half_module: bool
        Defaults to False, set to True to also draw a line at the border
        between half modules

    """

    if cfg.geometry == '9M':
        row = [512*i for i in range(5, 0, -1)]
        col = [1024, 2048]

        #x extent of the line
        lx = [0, 3072]

        #lines for half modules
        hm_row = [-256+512*i for i in range(6, 0, -1)]

        #Flip things if we draw vertical
        if vertical is True:
            row, col = col, row

        #Draw halfmodule lines
        if half_module is True:
            for tr in hm_row:
                ly = [tr, tr]
                if vertical is True:
                    ax.plot(ly, lx, '--', color='white', lw=1)
                else:
                    ax.plot(lx, ly, '--', color='white', lw=1)

        #Draw module lines
        for tr in row:
            ax.plot(lx, [tr, tr], color='white')

        for tc in col:
            ax.plot([tc, tc], lx, color='white')

        #Reset x and y limits, this can otherwise be changes by mpl
        ax.set_xlim(0, 3072)
        ax.set_ylim(0, 3072)
    else:
        raise NotImplementedError('Only for 9M so far')

def draw_module_names(ax, vertical=False, color='white'):
    """
    Write out the names of the modules in the 9M image.
    Names are fetched from sls_detector.config.Eiger9M

    Parameters
    ----------
    ax: mpl.axes
        axes to draw the names on
    vertical: bool, optional
        Defaults to False, set to True if the image is rotated
    color: str
        Textcolor

    """
    if cfg.geometry == '9M':
        if vertical is True:
            y = 952
            for j in range(3):
                x = 256
                for i in range(6):
                    t_str = 'T#{:02d}'.format(cfg.Eiger9M.T[i+j*6])
                    ax.text(x, y, t_str,
                            color=color,
                            horizontalalignment='center',
                            weight='normal')
                    x += 512
                y += 1024
        else:
            x = 512
            for j in range(3):
                y = 3000
                for i in range(6):
                    t_str = 'T#{:02d}'.format(cfg.Eiger9M.T[i+j*6])
                    ax.text(x, y, t_str,
                            color=color,
                            horizontalalignment='center',
                            weight='normal')
                    y -= 512
                x += 1024
    else:
        raise NotImplementedError('Only for 9M currently')

def fix_large_pixels(image, interpolation=True):
    """
    Expand and interpolate the values in large pixels at borders and in corners
    Works on sigle module data with one or several frames
    """
    #Check that rows and cols matches
    if image.shape[0] != 512 and image.shape[1] != 1024:
        raise ValueError('Unknown module size', image.shape)

    if len(image.shape) == 2:
        new_image = _fix_large_pixels(image, interpolation=interpolation)
    elif len(image.shape) == 3:
        #Multi frame
        new_image = np.zeros((514, 1030, image.shape[2]), dtype=image.dtype)
        for i in range(image.shape[2]):
            new_image[:, :, i] = _fix_large_pixels(image[:, :, i],
                                                   interpolation=interpolation)

    return new_image

def _fix_large_pixels(image, interpolation=True):
    """
    Internal use to expand one frame for one module
    Give the option to interplate pixels or just copy values
    """
    new_image = np.zeros((514, 1030))

    new_image[0:256,    0:256] = image[  0:256,   0: 256]
    new_image[0:256,  258:514] = image[  0:256, 256: 512]
    new_image[0:256,  516:772] = image[  0:256, 512: 768]
    new_image[0:256, 774:1030] = image[  0:256, 768:1024]

    new_image[258:514,    0:256] = image[  256:512,   0: 256]
    new_image[258:514,  258:514] = image[  256:512, 256: 512]
    new_image[258:514,  516:772] = image[  256:512, 512: 768]
    new_image[258:514, 774:1030] = image[  256:512, 768:1024]


    #Interpolation
    if interpolation:
        new_image[255, :] /= 2
        new_image[258, :] /= 2
        d = (new_image[258, :]-new_image[255, :]) / 4
        new_image[256, :] = new_image[255, :] + d
        new_image[257, :] = new_image[258, :] - d


        new_image[:, 255] /= 2
        new_image[:, 258] /= 2
        d = (new_image[:, 258]-new_image[:, 255]) / 4
        new_image[:, 256] = new_image[:, 255] + d
        new_image[:, 257] = new_image[:, 258] - d

        new_image[:, 513] /= 2
        new_image[:, 516] /= 2
        d = (new_image[:, 516]-new_image[:, 513]) / 4
        new_image[:, 514] = new_image[:, 513] + d
        new_image[:, 515] = new_image[:, 516] - d

        new_image[:, 771] /= 2
        new_image[:, 774] /= 2
        d = (new_image[:, 774]-new_image[:, 771]) / 4
        new_image[:, 772] = new_image[:, 771] + d
        new_image[:, 773] = new_image[:, 774] - d


    return new_image

def add_module_gaps(image):
    """
    Add the gaps in a multi module system

    Parameters
    ----------
    image: numpy_array
        Image used for expanding

    Returns
    --------
    new_image: numpy_array
        Image with the gaps inserted
    """
    if cfg.geometry == '9M':
        new_image = np.zeros((3264, 3106), dtype=image.dtype)
        for i in range(18):
            tmp = fix_large_pixels(image[mask.detector['9M'].module[i]])
            new_image[mask.detector['9M'].module_with_space[i]] = tmp
        return new_image
    else:
        raise NotImplementedError('Add module gaps only available for the 9M')



def _fmt(x, pos):
    """
    colorbar notation formatter for imshow
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)





def interpolate_pixel(pixel, data, pixelmask):
    """
    Return the interpolated value in pixel.

    Parameters
    -----------
    pixel: int, int
        Index of the pixel to interpolate
    data: numpy_array
        The image to use for interpolation
    pixelmask: numpy_array(bool)
        True for pixels that should be skipped.

    Returns
    value: double
        The value of the interpolated pixel

    """
    total = 0
    n_pixels = 0
    for step in set(permutations([1, 1, -1, -1, 0], 2)):
        try:
            _v = data[pixel[0]+step[0], pixel[1]+step[1]]
            m = pixelmask[pixel[0]+step[0], pixel[1]+step[1]]
            if _v and not np.isinf(_v) and not np.isnan(_v) and not m:
                total += _v
                n_pixels += 1
        except IndexError:
            print(pixel)

    return total/n_pixels


def global_scurve(data, x, chip_divided=False):
    """
    Plot a global scurve and differential scurve for the data provided.

    Parameters
    -----------
    data: numpy_array[row, col, N]
        Data to plot
    x: numpy_array[N]
        Data for the x axis
    chip_divided: bool
        Plot each chip seperate

    Returns:
        x: numpy_array
            xaxis data, usually threshold
        y: numpy_array
            summed counts per step


    .. warning::

        Chip divided plot is only implemented for a single module


    """
    plt.figure(figsize=(16, 9))
    if not chip_divided:
        y = data.sum(axis=0).sum(axis=0)

        plt.subplot(1, 2, 1)
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.plot(x, np.gradient(y))
    else:
        rows = [slice(256, 512, 1), slice(0, 256, 1)]
        cols = [slice(0, 256, 1), slice(256, 512, 1),
                slice(512, 768, 1), slice(768, 1024, 1)]
        chips = [[slice(0, data.shape[0], 1), ro, co] for
                 ro in rows for co in cols]

        for chip in chips:
            y = data[chip].sum(axis=1).sum(axis=1)
            print(y.shape, x.shape)
            plt.subplot(1, 2, 1)
            plt.plot(x, y)

            plt.subplot(1, 2, 2)
            plt.plot(x, np.gradient(y))

    return x, y

def random_pixels(data, x, n_pixels=5, rows=(0, 512), cols=(0, 1024)):
    """
    Plot data from random pixels

    Parameters
    ----------
    data: numpy_array[row, col, n]
        Detector data
    n_pixels: int, optional
        Number of pixels to plot. Defaults to 5
    rows: tuple
        Sets max and min row
    cols: tuple
        Sets max and min row

    """
    pixels = utils.random_pixel(N=n_pixels, rows=rows, cols=cols)
    plt.figure()
    for pixel in pixels:
        plt.plot(x, data[pixel])

def chip_histograms(data, xmin=0, xmax=2000, bins=400):
    """
    Plot a histogram per chip of the inflection point (fit_result['mu'])

    Parameters
    ----------
    data: numpy_array
        numpy_array in type of fit_results with named fields
    xmin: int, optional
        Lover edge of the histogram
    xmax: int, optional
        Higher edge of the histogram
    bins: int, optional
        Number of bins in the histogram
    """
    from sls_detector_tools import root_helper as r
    
    if cfg.nmod == 1:
        chips = mask.chip[4:]
    else:
        chips = mask.chip

    mean = []
    std = []
    lines = []

    if cfg.calibration.plot:
        plt.figure(figsize=(16, 9))


    max_value = 0

    for m in mask.detector[cfg.geometry].module:
        for i, c in enumerate(chips):
            canvas, histogram = r.hist(data[m][c], xmin=xmin,
                                       xmax=xmax, bins=bins)
            mean.append(histogram.GetMean())
            std.append(histogram.GetStdDev())

            label = r'{:d}: $\mu$: {:.1f} $\sigma$: {:.1f}'.format(i,
                                                                   mean[-1], std[-1])

            x0, y0 = r.getHist(histogram)
            if cfg.calibration.plot:
                plt.plot(x0, y0, ls='steps', label=label)

            if y0[1:].max() > max_value:
                max_value = y0[1:].max()


            lines.append((x0, y0))
    if cfg.calibration.plot:
        plt.legend(loc='best')
        plt.ylim(0, max_value*1.1)
        plt.xlim(xmin, xmax)


    return mean, std, lines

def plot_pixel_fit(x, y, par, px, figure=None):
    """
    Plot the fit of a single pixel, given parameters and values
    """
    colors = sns.color_palette()
    xx = np.linspace(x.min(), x.max(), 200)
    if figure is None:
        fig = plt.figure()
    else:
        fig = figure
    ax = plt.gca()
    ax.plot(x, y, 'o', label=str(px))
    label = r'$\mu$: {:.2f}\n $\sigma: {:.2f}$'.format(par[2], par[3])
    ax.plot(xx, function.scurve(xx, *par), label=label, color=colors[2])

    ax.legend(loc='upper left')
    ax.grid(True)
    sns.despine()
    ax.set_xlabel('Threshold [code]')
    ax.set_ylabel('Counts [1]')
    plt.tight_layout()

    return fig, ax


def plot_signals(data):
    """
    Plot a set of digital signals in one plot. Expects a named numpy
    array as input.
    """
    n_signals = len(data.dtype.names)
    colors = sns.color_palette(n_colors=n_signals)
    plt.figure(figsize=(10, 20))
    for i, name in enumerate(data.dtype.names):
        plt.subplot(n_signals, 1, i+1)
        plt.plot(data[name], label=name, ls='steps', color=colors[i])
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid()

    plt.tight_layout()
