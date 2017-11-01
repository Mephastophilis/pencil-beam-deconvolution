#practice deconvolution

from __future__ import division

import sys
sys.path.insert(0,"/home/faustus/Documents/Deconvolution_Code ")

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import libtiff
from libtiff import TIFF
import PBXIL_Richardson_Lucy as PBRL


slice1p12 = Image.open('/home/faustus/Documents/Deconvolution_Code/XLslices/slice1_position12.tif')
slice1p12 = np.array(slice1p12, dtype=np.float64)
darkimage = Image.open('home/faustus/Documents/Deconvolution_Code/XLslices/XLmeanbkgd.tif')
darkimage = np.array(darkimage, dtype=np.float64)

background = np.std(darkimage) ** 2

decon_filename='/home/fautus/Documents/decon_filename/decon/deconpractice.tif'

PBRL.RL_decon(slice1p12, 0.80, 0.0096, 0.15, 5, background, decon_filename, 100)
