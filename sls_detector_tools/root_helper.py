# -*- coding: utf-8 -*-
"""
Convenient wrapper for plotting Python things in ROOT.
Mostly uses numpy data and aims to have a Pyton like feeling.
No external dependencies except numpy and PyROOT
"""

import numpy as np
from ROOT import (
    TCanvas, TGraph, TH1D, TH2D,
    TGraphErrors,
    kGreen, kBlack, kWhite, kAzure,
    gPad, gStyle
    )


def style_histogram(h):
    h.SetFillColor(style.hist_color)
    h.SetLineColor(style.hist_line_color)
    h.SetLineWidth(style.hist_line_width)
    h.SetFillStyle(style.fill_style)

class style:
    """
    Class used to configure the apparence of PyROOT draw objects
    """
    line_color = kAzure+2
    line_width = 2
    marker_color = kAzure+2
    marker_style = 21 #Square
    hist_color = kAzure+4
    hist_line_width = 1
    hist_line_color = kBlack
    fill_style = 1001


def plot(x, y, options='ALP', title="A TGraph",
         wx=900, wy=600, draw=True, error_graph=False,
         ex=None, ey=None):
    """Plot a TGraph

    Parameters
    ----------
    x: array_like
        Values for the x axis
    y: array_like
        Values for the y axis
    options: str, optional
        Draw options for the TGraph default is ALP -> (Axis, Line, Points)
    title: str, optional
        Title of the TGraph, defaults to A TGraph
    wx: int, optional
        Width of the plot in pixels
    wy: int, optional
        Height of the plot in pixels
    draw: bool, optional
        Defaults to True, draws the plot
    error_graph: bool, optional
        Use a TGraphErrors instead of a normal TGraph
    ex: array_like, optional
        Errors in x values
    ey: array_like, optional
        Errors in y values, if not spefified it is assumed to be np.sqrt(y)

    Returns
    ----------
    c: TCanvas
        Canvas on which the TGraph is drawn if draw is set to false
        c=None
    g: TGraph
        The graph.

    Raises
    ------
    ValueError
        If the size of x and y is not the same

    """
    #To have someting to return if draw is False
    c = None

    #Show stats box if the graph is fitted
    gStyle.SetOptFit(1)

    #type conversions (Note! as array does not copy unless needed)
    xdata = np.asarray(x, dtype=np.double)
    ydata = np.asarray(y, dtype=np.double)

    #check array size
    if xdata.size != ydata.size:
        raise ValueError("Size of x and y must match. (x.size: {:d}"\
                         ", y.size: {:d})".format(x.size, y.size))

    if draw:
        #Workaround assigning random name to the histogram to avoid
        #deleting it, TODO! do we have a smarter solution
        canvas_name = 'r.plotcanvas'+str(np.random.rand(1))
        canvas_title = 'plot'
        c = TCanvas(canvas_name, canvas_title, wx, wy)
        c.SetFillColor(0)
        c.SetGrid()

    if error_graph is False:
        g = TGraph(x.size, xdata, ydata)
    else:
        #Use a TErrorGraph to display error bars
        if ex is None:
            ex = np.zeros(x.size)
        if ey is None:
            ey = np.sqrt(y).astype(np.double)

        g = TGraphErrors(x.size, xdata, ydata, ex, ey)

    g.SetTitle(title)


    #Styling of the graph
    g.SetLineColor(style.line_color)
    g.SetLineWidth(style.line_width)
    g.SetMarkerColor(style.marker_color)
    g.SetMarkerStyle(style.marker_style)


    if draw:
        g.Draw(options)
        c.Update()

    return c, g


def hist(data=None, xmin=None, xmax=None, bins=100, title='Histogram',
         add=False, c=None, h0=None, color=None,
         line_color=None, fill_style=1001):
    """Plot a histogram using TH1D also can add a new histogram as an overlay
    on the same canvas. In that case the canvas and one histogram on it
    needs to be specified

    Parameters
    ----------
    data: array_like
        An array or list with the hits that should be added to the histogram
    xmin: double, optional
        Lower edge of the lowest bin
    xmax: double, optional
        High edge of the highest bin
    bins: int, optional
        Defaults to 100
    title: std, optional
        Title of the histogram
    add: bool, optional
        If True add the new histogram to an old canvas
    c: TCanvas, optional
        Canvas to add the histogram to
    h0: TH1D, optional
        Base histogram plotted on the old canvas
    color: int, optional
        ROOT color for the histogram, if not specified the value is taken from
        style.hist_color
    line_color: int optional
        ROOT color for the line of the histogram is not specified taken from
        style.hist_line_color
    fill_style: int optional
        Style to fill the histogram, default 1001



    Returns
    ----------
    c: TCanvas
        Canvas on which the Histogram is drawn
        c=None
    h: TH1D
        The histogram that was just drawn.

    """

    #Convert other data, (Note! does not copy is its already a numpy array)
    data = np.asarray(data)

    #Apply custom plot settings
    if color is None:
        color = style.hist_color

    if line_color is None:
        line_color = style.hist_line_color


    #Check if we specified limits otherwise use min and max of the data
    if xmin is None:
        xmin = float(data.min())
    if xmax is None:
        xmax = float(data.max())



    #If we are adding to an existing histogram we should not create a new
    # TCanvas for the plotting
    if add is False:
        canvas_name = 'r.plotcanvas'+str(np.random.rand(1))
        canvas_title = 'Histogram'
        c = TCanvas(canvas_name, canvas_title, 200, 10, 900, 600)
        c.SetGrid()


    histogram_name = 'r.plothist'+str(np.random.rand(1))
    h = TH1D(histogram_name, title, int(bins), xmin, xmax)

    h.SetFillColor(color)
    h.SetLineColor(line_color)
    h.SetLineWidth(style.hist_line_width)
    h.SetFillStyle(fill_style)

#    #If we have data fill the histogram otherwise leave it
    if data is not None:
        hfill = h.Fill #slight speed up
        for x in np.nditer(data):
            hfill(x)
        del hfill

    if add is False:
        h.Draw()
    else:
        #Check axis and if the second histogram has a higher peak update
        gPad.Update()
        n = gPad.GetFrame().GetY2()
        m = h.GetMaximum()*1.05
        a = h0.GetYaxis()
        a.SetRangeUser(0, max(m, n))
        c.Update()
        h.Draw('SAME')

    c.Update()

    return c, h

def th2(x, y, xmin=0, xmax=0, ymin=0, ymax=0, bins=False):
    """Plot a TH2

    Parameters
    ----------
    x: array_like
        input
    y: array_like
        input

    Returns
    ----------
    c: TCanvas
        Canvas on which the Histogram is drawn
    h: TH2D
        The histogram that was just drawn.

    """
    c = TCanvas('r.plotcanvas'+str(np.random.rand(1)),
                'Plot'+str(np.random.rand(1)), 200, 10, 900, 600)
    c.SetFillColor(kWhite)
    c.GetFrame().SetFillColor(kWhite)

    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(y, list):
        y = np.asarray(y)
    if not ymin:
        xmin = 0
        xmax = x.max()

        ymin = 0
        ymax= y.max()


    if bins:
        if len(bins) == 1:
            nbinsx = bins[0]
            nbinsy = bins[0]
        if len(bins) == 2:
            nbinsx = bins[0]
            nbinsy = bins[1]
    else:
        nbinsx = 500
        nbinsy = 500


    h = TH2D('r.hist2'+str(np.random.rand(1)), 'Title', nbinsx,
             xmin, xmax, nbinsy, ymin, ymax)

    hfill = h.Fill
    for x1, y1 in np.nditer([x, y]):
        hfill(x1, y1)

    h.Draw('colz')
    c.Update()

    return c, h



def getHist(h, edge='center'):
    """Returns the bin edges and values for a ROOT histogram.

    Parameters
    ----------
    h: TH1
        Histogram that we need the values from
    edge: str, optional
        Which edge that should be used. Defaults to center.
        Possible values low, center high and both



    Returns
    ----------
    x: np.array
        Bin edges as spefied by edge
    y: np.array
        Values of the bins

    Raises
    ------
    ValueError
        If edge is something different from low, center, high or both

    """
    bins = h.GetNbinsX()

    if edge == 'low' or edge == 'high' or edge == 'center':
        x = np.zeros(bins)
        y = np.zeros(bins)

        if edge == 'low':
            for i in range(bins):
                x[i] = h.GetBinLowEdge(i)
                y[i] = h.GetBinContent(i)
        elif edge == 'high':
            for i in range(bins):
                x[i] = h.GetBinLowEdge(i) + h.GetBinWidth(i)
                y[i] = h.GetBinContent(i)
        elif edge == 'center':
            for i in range(bins):
                x[i] = h.GetBinCenter(i)
                y[i] = h.GetBinContent(i)

    elif edge == 'both':
        print(edge)
        print('Bins', bins)
        x = np.zeros(bins*2)
        y = np.zeros(bins*2)
        print(x.size, y.size)

        for i in range(bins):
            x[i*2] = h.GetBinLowEdge(i)
            x[i*2+1] = h.GetBinLowEdge(i) + h.GetBinWidth(i)

            y[i*2] = h.GetBinContent(i)
            y[i*2+1] = h.GetBinContent(i)

    else:
        raise ValueError('Unknown edge specification use: high, low, center or both')


    return x, y
