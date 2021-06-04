import os
import numpy as np
from sls_detector_tools.io import read_trimbit_file, write_trimbit_file
from sls_detector_tools.mask import chip
top = 81
bottom = 82

T = 38


targets = ['Ti', 'Fe', 'Cu', 'Ge', 'Mo']
gains = ['veryhighgain', 'highgain', 'standard', 'lowgain', 'verylowgain']
energy = {'Ti': 4500, 'Fe':6400, 'Cu':8000, 'Ge': 9900, 'Mo': 17500}
vrfs = [{'top': 3345, 'bottom': 3353},
        {'top': 3077, 'bottom': 3088},
        {'top': 2900, 'bottom': 2912},
        {'top': 2724, 'bottom': 2738},
        {'top': 2147, 'bottom': 2159}]

_dacs = [('vsvp',    0, 4000,    0),
        ('vtr',     0, 4000, 2500),
        ('vrf',     0, 4000, 3300),
        ('vrs',     0, 4000, 1400),
        ('vsvn',    0, 4000, 4000),
        ('vtgstv',  0, 4000, 2556),
        ('vcmp_ll', 0, 4000, 1500),
        ('vcmp_lr', 0, 4000, 1500),
        ('vcall',   0, 4000, 4000),
        ('vcmp_rl', 0, 4000, 1500),
        ('rxb_rb',  0, 4000, 1100),
        ('rxb_lb',  0, 4000, 1100),
        ('vcmp_rr', 0, 4000, 1500),
        ('vcp',     0, 4000,  200),
        ('vcn',     0, 4000, 2000),
        ('vis',     0, 4000, 1550),
        ('iodelay', 0, 4000,  660)]
_dacnames = [_d[0] for _d in _dacs]

idx = { _d[0]:i for i, _d in enumerate(_dacs)}

for target, gain, vrf in zip(targets, gains, vrfs):
    src_path = f'/mnt/disk1/calibration/T{T}'
    src = f'{src_path}/{gain}/T{T}_{target}_{gain}.sn{top:03}'
    trim, dacs = read_trimbit_file(src, 17)
    tb = np.load(f'{src_path}/{gain}/T{T}_{target}_{gain}.npy')
    mu = np.load(f'{src_path}/{gain}/T{T}_vcmp_{target}XRF_fit_1.npy')['mu']
    dst_path = f'{src_path}/new/standard/{energy[target]}eV'
    try:
        os.makedirs(dst_path)
    except FileExistsError:
        pass
    
    new_dacs = np.zeros(18)
    new_dacs[0:17] = dacs 

    #calculate vcmp
    vcmp = []
    for i in range(8):
        m = np.logical_and(mu[chip[i]]>100, mu[chip[i]]<1900)
        vcmp.append(np.round(mu[chip[i]][m].mean()))


    #top
    dst = f'{dst_path}/noise.sn{top:03}'
    new_dacs[idx['vrf']] = vrf['top']
    new_dacs[idx['vcmp_ll']] = vcmp[0]
    new_dacs[idx['vcmp_lr']] = vcmp[1]
    new_dacs[idx['vcmp_rl']] = vcmp[2]
    new_dacs[idx['vcmp_rr']] = vcmp[3]
    new_dacs[idx['vcall']] = 0
    write_trimbit_file(dst, tb[256:, :], new_dacs, new_dacs.size)

    #bottom
    dst = f'{dst_path}/noise.sn{bottom:03}'
    new_dacs[idx['vrf']] = vrf['bottom']
    new_dacs[idx['vcmp_rr']] = vcmp[4]
    new_dacs[idx['vcmp_lr']] = vcmp[5]
    new_dacs[idx['vcmp_rl']] = vcmp[6]
    new_dacs[idx['vcmp_ll']] = vcmp[7]
    new_dacs[idx['vcall']] = 0
    write_trimbit_file(dst, tb[0:256, :], new_dacs, new_dacs.size)

    print(src, dst)
