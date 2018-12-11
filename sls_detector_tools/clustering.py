# -*- coding: utf-8 -*-
"""
Connected componets based clustering for pixel detector images. 

"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import skimage.morphology as morph
from skimage.measure import regionprops, label


#PyTables
from tables import open_file, IsDescription, UInt32Col, Float32Col, UInt16Col, UInt32Col, Int32Atom, StringCol, Float32Atom

class open_cluster_file:
    def __init__(self, fname):
        self.fname = fname
    def __enter__(self):
        self.file = open_file(self.fname)
        self.table = self.file.root.clusters
        return self.file, self.table
    def __exit__(self, type, value, traceback):
        self.file.close()


class Cluster(IsDescription):
    """
    Cluster information class to be stored in the hdf5 file.
    
    """
    id = UInt32Col()
    frameNr = UInt32Col()
    area = UInt16Col()
    volume = Float32Col()
    center = Float32Col(2)
    row = UInt16Col()
    col = UInt16Col()
    non_weighted_center = Float32Col(2)
    region = UInt16Col(4)
    orientation = Float32Col()
    solidity = Float32Col()
    major_axis_length = Float32Col()
    minor_axis_length = Float32Col()
    circularity = Float32Col()
    max_intensity = Float32Col()
    mean_intensity = Float32Col()
    perimeter = Float32Col()
    dose = Float32Col()
    type = StringCol(50)



class Cluster_finder():
    def __init__(self, filename = 'temp.h5f',
                 basic = False):

        self._basic = basic #flag to only do basic clustering
        self.data=None
        self.n_frames = 0
        self.cluster_id = 0

        
        self.clustersPerFrame=np.zeros(0)
        self.countsPerFrame=np.zeros(0)


        #Setup the hdf5 file for persistent storage
        self.file = open_file(filename, mode = 'w', title = 'Clusters')
        self.table = self.file.create_table(self.file.root, 'clusters', Cluster, 'Clusters from run')
        self.vlarray = self.file.create_vlarray(self.file.root, 'clusterArray',
                                    Float32Atom(shape=()))

        self.clusters = self.file.root.clusters
        
    def find_clusters(self, data, threshold, expand = True):
        """
        Performs clustering on all frames in data
        Result stored as clusters and cluster info

        clusterInfo, array with [frameNr size, volume, extrema, slice]
        
        input numpy array [frame, row, col]

        """
        

        self.clustersPerFrame = np.hstack((self.clustersPerFrame, 
                                           np.zeros(data.shape[0]) ))
        self.countsPerFrame = np.hstack((self.countsPerFrame, 
                                         np.zeros(data.shape[0]) ))
        
        print('Starting to process', data.shape[0], 'frames')
        t0 = time.time()

        
        cluster = self.table.row

        connectivity = [[1,1,1],[1,1,1],[1,1,1]]

        
        for frame_nr, frame in enumerate(data):
            if frame_nr%100 == 0: 
                print(f'Processing frame: {frame_nr}')
            
            th_image = frame > threshold
            labeled, nrOfFeatures=ndimage.label(th_image, connectivity)
            self.clustersPerFrame[self.n_frames] = nrOfFeatures
            self.countsPerFrame[self.n_frames] = frame.sum()

            for p in regionprops(labeled, intensity_image = frame, coordinates='rc'):
                #Write cluster properties to table
                cluster['id'] = self.cluster_id
                cluster['frameNr'] = self.n_frames
                cluster['area'] = p.area
                cluster['volume'] = p.intensity_image.sum() #includes also below th pixels?
                centroid = p.weighted_centroid
                cluster['center'] = centroid
                cluster['row'] = np.round(centroid[0]).astype(np.uint16)
                cluster['col'] = np.round(centroid[1]).astype(np.uint16)
                cluster['max_intensity'] = p.max_intensity
                if not self._basic:
#                    cluster['center'] = p.weighted_centroid
                    cluster['non_weighted_center'] = p.centroid
                    cluster['region'] = p.bbox
                    cluster['orientation'] = p.orientation
                    cluster['solidity'] = p.solidity
                    cluster['major_axis_length'] = p.major_axis_length
                    cluster['minor_axis_length'] = p.minor_axis_length
                    cluster['max_intensity'] = p.max_intensity
                    cluster['mean_intensity'] = p.mean_intensity
                    cluster['perimeter'] = p.perimeter
                    if p.major_axis_length > 0 and p.minor_axis_length > 0:
                        circ = p.minor_axis_length/p.major_axis_length
                    else:
                        circ = -1

                    cluster['circularity'] = circ
                    cluster['type'] = 'unknown'

                cluster.append()
                #Write cluster data to vlarray
                c = list(zip(p.coords[:,0], p.coords[:,1], p.intensity_image[p.intensity_image>0]))
                self.vlarray.append( np.array(c).flatten() )

                #Update cluster id counter
                self.cluster_id += 1
            self.n_frames += 1





        self.file.flush()
        print(f'Done in {time.time()-t0:.2f}s ')
        return self.file


    def clusterSizeHist(self, label='Cluster Size', maxValue=False, vol=False):
        label=label+';Size;Number of Clusters;'

        #Colors and fill
        color = kGreen
        lineColor = kBlack
        lineWidth = 0
        fillStyle = 1001

        #Create canvas
        c = TCanvas( 'r.plotcanvas'+str(np.random.rand(1)), 'Plot'+str(np.random.rand(1)), 200, 10, 900, 600 )
        c.SetFillColor( kWhite )
        c.GetFrame().SetFillColor( kWhite )

        #Find the maximum cluster volume
        if not maxValue:
            maxValue=self.file.root.clusters.cols.area[:].max()+3
        bins = int(maxValue)


        h=TH1F("FrameHist"+str(np.random.rand(1)), "Cluster Size", bins, 0, maxValue)

        #Histogram setup
        h.SetFillColor( color )
        h.SetLineColor( lineColor )
        h.SetLineWidth( lineWidth )
        h.SetFillStyle( fillStyle )

        #Fill the histogram
        hfill = h.Fill
        if not vol:
            for cl in self.file.root.clusters.cols.area[:]: hfill(cl)
        else:
            for row in self.file.root.clusters.where('(volume == vol)'):
                hfill( row['area'] )
        h.SetTitle(label)
        h.SetTitleOffset(1.3,'Y')

        h.Draw()
        c.Update()

        return c,h



    def viewCluster(self, n, add=False, delay = False):
        gStyle.SetOptStat(0) #Disable statistics
        cluster = np.zeros((self._frame._shape[0], self._frame._shape[1]))
        data = self.file.root.clusterArray[n]
        for item in zip(data[0::3], data[1::3], data[2::3]):
            cluster[item[0], item[1]] = item[2]

        if not add:
            c = TCanvas()
            c.SetFillColor( kWhite )
            c.GetFrame().SetFillColor( kWhite )

            h=TH2F(str(int(np.random.rand(1)*1000000)), "Cluster", cluster.shape[0],0,cluster.shape[0], cluster.shape[1],0,cluster.shape[1])
            h.SetFillColor( kGreen )
            h.SetLineColor( kBlack )
        else:
            c = self._frame.canvs[-1]
            h = self._frame.hists[-1]



        for i in range(cluster.shape[0]):
            for j in range(cluster.shape[1]):
                h.Fill(i, j, cluster[i,j])
#        h.SetTitle(str(n)+' '+str(self.clusterInfo[n]['Circularity'])+' '+str(self.clusterInfo[n]['size']))

        if not delay:
            h.Draw('colz')
            c.Update()



        #Read parameters
        cl = self.file.root.clusters[n]
        x1,y1 = cl['center']
        major_axis = cl['major_axis_length']
        minor_axis= cl['minor_axis_length']
        phi = cl['orientation']
        from ROOT import TEllipse
        #TEllipse(Double_t x1, Double_t y1, Double_t r1, Double_t r2 = 0, Double_t phimin = 0, Double_t phimax = 360, Double_t theta = 0)
        e = TEllipse(x1, y1, major_axis/2, minor_axis/2, 0, 360, np.rad2deg(phi)+90)
        e.SetFillStyle(0)
        self._elipses.append(e)

        if not delay:
            for e in self._elipses:
                e.Draw('L')


        tStr = '#splitline{id: ' + str(cl['id']) + ' volume: '+str(cl['volume'].round(4))+' area: '+str(cl['area'])+'}{circ: ' + str(cl['circularity'].round(2))+' solidity: ' + str(cl['solidity'].round(2)) + 'type: '+cl['type']+'}'
        text = TLatex(x1+5,y1+5,tStr)
        text.SetTextSize(0.02)
        self._text.append(text)
        if not delay:
            for t in self._text:
                t.Draw()



        self._frame.canvs.append(c)
        self._frame.hists.append(h)

    def printCluster(self, n):
        p = self.file.root.clusters[n]
        print('=== Cluster nr', p['id'], '===')
        print('Frame:', p['frameNr'])
        print('Area', p['area'])
        print('Volume', p['volume'])
        print('Center', p['center'])
        print('Orientation', p['orientation'], np.rad2deg(p['orientation']))
        print('Solidity', p['solidity'])
        print('Major Axis Length', p['major_axis_length'])
        print('Minor Axis Length', p['minor_axis_length'])
        print('Circularity', p['circularity'])
        print('Max intensity', p['max_intensity'])
        print('Perimiter', p['perimeter'])
        print('Mean intensity', p['mean_intensity'])








    def hitMap(self, cond):
        c = TCanvas(str(int(np.random.rand(1)*1000000)), 'Cluster', 960, 800)
        c.SetFillColor( kWhite )
        c.GetFrame().SetFillColor( kWhite )

        #	TH2F(const char* name, const char* title, Int_t nbinsx, Double_t xlow, Double_t xup, Int_t nbinsy, Double_t ylow, Double_t yup)
        h=TH2F(str(int(np.random.rand(1)*1000000)), "Hit", 512, 0 ,256, 512,0,256)
        hfill = h.Fill
        for row in self.file.root.clusters.where( cond ):
            x,y=row['center']
            hfill(x,y)
        h.Draw('colz')
        c.Update()
        return c,h

    def viewCluster0(self, n):
        """
        Display clusters using a list of cluster id's
        """
        gStyle.SetOptStat(0) #Disable statistics

        cluster = np.zeros((self.data.shape[1], self.data.shape[2]))

        for i in n:
            data = self.file.root.clusterArray[i]
            for item in zip(data[0::3], data[1::3], data[2::3]):
                cluster[item[0], item[1]] += item[2]


        gStyle.SetPadLeftMargin(0.1)
        gStyle.SetPadRightMargin(0.23)
        gStyle.SetPadTopMargin(0.1)
        gStyle.SetPadBottomMargin(0.1)

        #TCanvas(const char* name, const char* title, Int_t ww, Int_t wh)
        c = TCanvas(str(int(np.random.rand(1)*1000000)), 'Cluster', 960, 800)

        c.SetFillColor( kWhite )
        c.GetFrame().SetFillColor( kWhite )

        h=TH2F(str(int(np.random.rand(1)*1000000)), "Cluster", cluster.shape[0],0,cluster.shape[0], cluster.shape[1],0,cluster.shape[1])
        h.SetFillColor( kGreen )
        h.SetLineColor( kBlack )

        for i in range(cluster.shape[0]):
            for j in range(cluster.shape[1]):
                h.Fill(i, j, cluster[i,j])



        h.Draw('colz')
        c.Update()

        return c,h

def view_clusters(cluster_id, table, cluster_array, shape):
    image = np.zeros(shape)
    for i in cluster_id:
        data = cluster_array[i]
        for item in zip(data[0::3], data[1::3], data[2::3]):
            image[int(item[0]), int(item[1])] += item[2]
    return image

def hitmap(cluster_id, table, cluster_array, shape):
    image = np.zeros(shape)
    for i in cluster_id:
        data = cluster_array[i]
        for item in zip(data[0::3], data[1::3], data[2::3]):
            image[int(item[0]), int(item[1])] += 1
    return image

def hitmap_energy(cluster_id, table, cluster_array, shape):
    image = np.zeros(shape)
    for i in cluster_id:
        data = cluster_array[i]
        for item in zip(data[0::3], data[1::3], data[2::3]):
            image[int(item[0]), int(item[1])] += item[2]
    return image