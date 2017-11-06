#practice deconvolution

from __future__ import division

import sys
sys.path.insert(0,"/home/faustus/Documents/Deconvolution_Code ")

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#import scipy
#import libtiff
#from libtiff import TIFF
import PBXIL_Richardson_Lucy as PBRL

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tabulate import tabulate

pix_size=0.0096 #cm (96 um)

slice1p12 = Image.open('/home/faustus/Documents/Deconvolution_Code/XLslices/slice1_position12.tif')
slice1p12 = np.array(slice1p12, dtype=np.float64)
darkimage = Image.open('/home/faustus/Documents/Deconvolution_Code/XLslices/XLmeanbkgd.tif')
darkimage = np.array(darkimage, dtype=np.float64)

background = np.std(darkimage) ** 2

decon_filename='/home/faustus/Documents/Deconvolution_Code/decon/deconpractice'

decon, iter_vec, diff_vec = PBRL.RL_decon(slice1p12, 0.80, pix_size, 0.15, 5, background, decon_filename, [100000], stopping_condition=10e-5, maxIter_Number=100000)

 # ----------       Plot profile across sources       --------------
majorLocator = MultipleLocator(0.5)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(0.1)

plt.figure(figsize=(10,10))
plt.plot(pix_size * np.array(range(512)), decon[311,:] , linewidth = 1)
plt.plot(pix_size * np.array(range(512)), slice1p12[311,:], linewidth = 1)
plt.axes().xaxis.set_minor_locator(minorLocator)
plt.xlabel('Distance (cm)')
plt.legend(['Deconvolved Image', 'Pre-Processed Image'])
plt.title('Intensity Profile @ Depth 0.8 cm (Iterations: ' + str(iter_vec[-1]) + ')')
plt.savefig(decon_filename + 'profile_plot_#images_' + '_SPXIL_depth_0_8cm_iter_' + str(iter_vec[-1]) + '_padded4.png')
plt.close()

# ----------       convergence plots       --------------
plt.figure(figsize=(10,10))
plt.semilogy(iter_vec, diff_vec , linewidth = 1)
plt.xlabel('Iteration Number')
plt.ylabel('Convergence Parameter')
plt.title('Convergence @ Depth 0.8 cm (Iterations: ' + str(iter_vec[-1]) + ')')
plt.savefig(decon_filename + 'convergence_#images_' + '_SPXIL_depth_0_8cm_iter_' + str(iter_vec[-1]) + '_paddded3.png')
plt.close()

np.save(decon_filename + 'convergence_#images_' + '_SPXIL_depth_0_8cm_iter_' + str(iter_vec[-1]) + '_paddded3.npy', [iter_vec, diff_vec])

conv = [iter_vec, diff_vec]
conv_list = [list(x) for x in zip(*conv)]

f = open(decon_filename + 'convergence_#images_' + '_SPXIL_depth_0_8cm_iter_' + str(iter_vec[-1]) + '_paddded3.txt', 'w')
f.write(tabulate(conv_list ,['Number of Iterations', 'Convergence Parameter'], numalign="center"))
f.close()