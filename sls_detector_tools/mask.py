# -*- coding: utf-8 -*-
"""
General interface to select modules and chips from detectors
also holds some geometry information

Uses a dictionary to provide easy access to different geometries
trough the cfg.geometry setting.
"""


#Generate slices to access chip data
row = [slice(256, 512, 1), slice(0, 256, 1)]
col= [slice(0, 256, 1), slice(256, 512, 1), 
      slice(512, 768, 1), slice(768, 1024, 1)]
chip = [(r, c) for r in row for c in col]
halfmodule = [(r, slice(0, 1024, 1)) for r in row]

vcmp0 = [':vcmp_ll', ':vcmp_lr', ':vcmp_rl', ':vcmp_rr']
vcmp1 = [':vcmp_rr', ':vcmp_rl', ':vcmp_lr', ':vcmp_ll']

#Vcmp corresponding to the mask in chip
vcmp = ['0:vcmp_ll',
        '0:vcmp_lr',
        '0:vcmp_rl',
        '0:vcmp_rr',
        '1:vcmp_rr',
        '1:vcmp_rl',
        '1:vcmp_lr',
        '1:vcmp_ll']




class eiger250k:
    """
    Mask for Eiger500k module
    """
    def __init__(self):
        self.nrow = 256
        self.ncol = 1024

        self.module = [[slice(0, 256, 1), slice(0, 1024, 1)]]

        #Half modules
        self.halfmodule = [[slice(0, 256, 1), slice(0, 1024, 1)]]

        self.port = []
        for col in range(1):
            for row in range(1,0,-1):
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*col*2,512*(col*2+1),1)] )
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*(col*2+1),512*(col*2+2),1)] )


class eiger500k:
    """
    Mask for Eiger500k module
    """
    def __init__(self):
        self.nrow = 512
        self.ncol = 1024

        self.module = [(slice(0, 512, 1), slice(0, 1024, 1))]

        #Half modules
        _col = [slice(256*(i-1), 256*i, 1) for i in range(2, 0, -1)]
        _row = [slice(1024*i, 1024*(i+1), 1) for i in range(1)]
        self.halfmodule = [(c,r) for r in _row for c in _col]

        self.port = []
        for col in range(1):
            for row in range(2,0,-1):
                self.port.append( (slice(256*(row-1),256*row, 1), slice(512*col*2,512*(col*2+1),1)) )
                self.port.append( (slice(256*(row-1),256*row, 1), slice(512*(col*2+1),512*(col*2+2),1)) )


class eiger9M:
    """
    9M detector for cSAXS
    """
    def __init__(self):
        self.nrow = 3072
        self.ncol = 3072

        #Single ports
        self.port = []
        for col in range(3):
            for row in range(12,0,-1):
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*col*2,512*(col*2+1),1)] )
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*(col*2+1),512*(col*2+2),1)] )

        #Half modules
        col = [slice(256*(i-1), 256*i, 1) for i in range(12, 0, -1)]
        row = [slice(1024*i, 1024*(i+1), 1) for i in range(3)]
        self.halfmodule = [[c, r] for r in row for c in col]

        #Modules
        row = [slice(512*(i-1), 512*i, 1) for i in range(6,0,-1)]
        col = [slice(1024*i, 1024*(i+1),1) for i in range(3)]
        self.module = [ (r,c) for c in col for r in row]



    #    Expanded modules
        gap_row = 36
        gap_col = 8
        row = [slice(514*(i-1)+(i-1)*gap_row, 514*i+(i-1)*gap_row, 1) for i in range(6,0,-1)]
        col = [slice(1030*i+i*gap_col,1030*(i+1)+i*gap_col,1) for i in range(3)]
        self.module_with_space = [[r,c] for c in col for r in row]

        #Half modules with space
        row = [slice(3264-257*i-i/2*36-257, 3264-257*i-i/2*36, 1) for i in range(12)]
        col = [slice( 1030*i+8*i, 1030*i+8*i+1030, 1) for i in range(3)]
        self.halfmodule_with_space = [[r,c] for c in col for r in row]

        self.vcmp = []
        for i in range(len(self.module)):
            for v in vcmp0:
                self.vcmp.append(str(i*2)+v)
            for v in vcmp1:
                self.vcmp.append(str(i*2+1)+v)

class eiger2M:
    """
    2M detector at ESRF, horizontal modules vertical stacking
    """
    def __init__(self):
        self.nrow = 2048
        self.ncol = 1024

        self.port = []
        for col in range(1):
            for row in range(8,0,-1):
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*col*2,512*(col*2+1),1)] )
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*(col*2+1),512*(col*2+2),1)] )

        #Half modules
        col = [slice(256*(i-1),256*i, 1) for i in range(8,0,-1)]
        row = [slice(1024*i,1024*(i+1),1) for i in range(1)]
        self.halfmodule = [[c,r] for r in row for c in col]

        #Modules
        row = [slice(512*(i-1),512*i, 1) for i in range(4,0,-1)]
        col = [slice(1024*i,1024*(i+1),1) for i in range(1)]
        self.module = [[r,c] for c in col for r in row]

        self.vcmp = []
        for i in range(len(self.module)):
            for v in vcmp0:
                self.vcmp.append(str(i*2)+v)
            for v in vcmp1:
                self.vcmp.append(str(i*2+1)+v)

class eiger1_5MOMNY:
    """
    1.5M detector for OMNY, horizontal modules vertical stacking
    """
    def __init__(self):
        self.nrow = 1024
        self.ncol = 1536

          #Single ports
        self.port = []
        for col in range(1):
            for row in range(6,0,-1):
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*col*2,512*(col*2+1),1)] )
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*(col*2+1),512*(col*2+2),1)] )

        #Half modules
        col = [slice(256*(i-1),256*i, 1) for i in range(6,0,-1)]
        row = [slice(1024*i,1024*(i+1),1) for i in range(1)]
        self.halfmodule = [[c,r] for r in row for c in col]

        #Modules
        row = [slice(512*(i-1),512*i, 1) for i in range(3,0,-1)]
        col = [slice(1024*i,1024*(i+1),1) for i in range(1)]
        self.module = [[r,c] for c in col for r in row]

        self.vcmp = []
        for i in range(len(self.module)):
            for v in vcmp0:
                self.vcmp.append(str(i*2)+v)
            for v in vcmp1:
                self.vcmp.append(str(i*2+1)+v)
                
class eiger1_5M:
    """
    1.5M detector for cSAXs WAXS, horizontal module stacking
    """
    def __init__(self):
        self.nrow = 3072
        self.ncol = 512

          #Single ports
        self.port = []
        for col in range(6):
            for row in range(1,0,-1):
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*col*2,512*(col*2+1),1)] )
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*(col*2+1),512*(col*2+2),1)] )

        #Half modules
        col = [slice(256*(i-1),256*i, 1) for i in range(1,0,-1)]
        row = [slice(1024*i,1024*(i+1),1) for i in range(6,0, -1)]
        self.halfmodule = [[c,r] for r in row for c in col]

        #Modules
        row = [slice(512*(i-1),512*i, 1) for i in range(1,0,-1)]
        col = [slice(1024*i,1024*(i+1),1) for i in range(6,0,-1)]
        self.module = [[r,c] for c in col for r in row]

        self.vcmp = []
        for i in range(len(self.module)):
            for v in vcmp0:
                self.vcmp.append(str(i*2)+v)
            for v in vcmp1:
                self.vcmp.append(str(i*2+1)+v)

class eiger1M:
    """
    1M detector at ESRF, horizontal modules vertical stacking (in reality flipped but calibrated not flipped)
    """
    def __init__(self):
        self.nrow = 1024
        self.ncol = 1024

        self.port = []
        for col in range(1):
            for row in range(4,0,-1):
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*col*2,512*(col*2+1),1)] )
                self.port.append( [slice(256*(row-1),256*row, 1), slice(512*(col*2+1),512*(col*2+2),1)] )

        #Half modules
        col = [slice(256*(i-1),256*i, 1) for i in range(4,0,-1)]
        row = [slice(1024*i,1024*(i+1),1) for i in range(1)]
        self.halfmodule = [[c,r] for r in row for c in col]

        #Modules
        row = [slice(512*(i-1),512*i, 1) for i in range(2,0,-1)]
        col = [slice(1024*i,1024*(i+1),1) for i in range(1)]
        self.module = [[r,c] for c in col for r in row]

        self.vcmp = []
        for i in range(len(self.module)):
            for v in vcmp0:
                self.vcmp.append(str(i*2)+v)
            for v in vcmp1:
                self.vcmp.append(str(i*2+1)+v)

#Dictionary lookup for using detector geometry for lookup
detector = {'250k': eiger250k(),
            '500k': eiger500k(),
            '2M': eiger2M(),
            '9M': eiger9M(),
            '1.5M': eiger1_5M(),
            '1.5MOMNY': eiger1_5MOMNY(),
            '1M': eiger1M()
            }
