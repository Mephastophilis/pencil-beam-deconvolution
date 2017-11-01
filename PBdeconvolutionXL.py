#PB Deconvolution X-ray Luminescence

from __future__ import division

import sys
sys.path.insert(0,"/Users/bryanquigley/Documents/Deconvolution_Code ")

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import libtiff
from libtiff import TIFF

#Open X-ray Luminescence Images, subtract Mean Background, and Save files

folder_name='/home/faustus/Documents/Deconvolution_Code/XLslices/'
bkgdname = folder_name + 'XLmeanbkgd.tif'
meanbkgdXL = Image.open(bkgdname)
meanbkgdXL = np.array(meanbkgdXL, dtype=np.float64)

for s in range (1,6):
    for p in range (1,24):
        if p <10:
            filename = folder_name + 'slice' + str(s) + '_position0' + str(p) + '.tif'
            XLdata = filename
            X = Image.open(filename)
            X = np.array(X, dtype=np.float64)
            XLsub = np.subtract(X,meanbkgdXL)
            XLsub[XLsub < 0]=0
            XLsub=np.array(XLsub, dtype=np.uint16)
            tiff = TIFF.open(folder_name + 'XLbkgdsub_slice' + str(s) + '_position0' + str(p) + '.tif', mode='w')
            tiff.write_image(XLsub)
            tiff.close()
        else:
            filename = folder_name + 'slice' + str(s) + '_position' + str(p) + '.tif'
            XLdata = filename
            X = Image.open(filename)
            X = np.array(X, dtype=np.float64)
            XLsub = np.subtract(X,meanbkgdXL)
            XLsub[XLsub < 0]=0
            XLsub=np.array(XLsub, dtype=np.uint16)
            tiff = TIFF.open(folder_name + 'XLbkgdsub_slice' + str(s) + '_position' + str(p) + '.tif', mode='w')
            tiff.write_image(XLsub)
            tiff.close()

#Deconvolution






depth = 0.8
max_iteration = 100000
iteration_list = [25, 100, 500, 2000, 5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 100000, 125000, 150000, 175000, 200000 ]
