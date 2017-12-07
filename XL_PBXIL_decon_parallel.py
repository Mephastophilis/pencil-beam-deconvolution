#practice deconvolution

from __future__ import division

import sys
sys.path.insert(0,"/home/bquigley/Research/Deconvolution_Code")

from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import scipy
import libtiff
from libtiff import TIFF
import PBXIL_Richardson_Lucy as PBRL
import removeHotCold as rHC
from multiprocessing import Pool

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tabulate import tabulate

pix_size = 0.0096 #cm (96 um)

darkarray = np.zeros((512,512,50))
for i in xrange(1,7):
  if i < 10:
    filepath = '/home/bquigley/Research/Deconvolution_Code/XL_Camera_Images_Labeled/XLdark_0' + str(i) + '.tif'
    dark = Image.open(filepath)

    darkarray[:,:,i-1] = np.array(dark, dtype=np.float64)

  else:
    filepath = '/home/bquigley/Research/Deconvolution_Code/XL_Camera_Images_Labeled/XLdark_' + str(i) + '.tif'
    dark = Image.open(filepath)
    darkarray[:,:,i-1] = np.array(dark, dtype=np.float64)

background = rHC.remove_hot_and_dead_pixels(darkarray[:,:,0], 1700, remove_dead=True,lower_thresh=200)

darkmean = np.mean(darkarray,2)
background = np.std(background) ** 2 

decon_filename='/home/bquigley/Research/Deconvolution_Code/results/'

recon3D = np.zeros((5,512,23))

decon_inputs = []
totalslices = 1
totalpositions = 23

for s in xrange(1,totalslices+1):
    for p in xrange(10,18):
        if p < 10:
            filepath = '/home/bquigley/Research/Deconvolution_Code/XL_Camera_Images_Labeled/XLslice' + str(s) + 'pos0' + str(p) + 'img01.tif'
        else:
            filepath = '/home/bquigley/Research/Deconvolution_Code/XL_Camera_Images_Labeled/XLslice' + str(s) + 'pos' + str(p) + 'img01.tif'  
    
        detectedimage = Image.open(filepath)
        detectedimage = np.array(detectedimage, dtype=np.float64)
        detectedimage_p = detectedimage - darkmean
        detectedimage_p += background
        detectedimage_p[detectedimage_p < 0] = 0
    
        decon_inputs.append([s, p, detectedimage_p, 0.08, pix_size, 0.15, 5, background, decon_filename, [25, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 125000, 150000, 175000], 1e-8, 200000, 4])


p = Pool(processes=8)
decon_results = p.map(PBRL.RL_decon, decon_inputs[:][:])



print('done! code ran successfully')

##
for s in xrange(1,totalslices+1):
    for p in xrange(10,18):
        vector_position=(totalpositions)*(s-1)+(p-10)
        recon3D[s-1,:,p-1] = decon_results[vector_position][0][310,:]
    
        # ----------       convergence plots       --------------
        #plt.figure(figsize=(10,10))
        #plt.semilogy(decon_results[vector_position][1], decon_results[vector_position][2], linewidth = 1)
        #plt.xlabel('Iteration Number')
        #plt.ylabel('Convergence Parameter')
        #plt.title('Convergence @ Depth 0.8 cm Slice ' + str(s) + ' Position ' + str(p) + '(Iterations: ' + str(decon_results[vector_position][1][-1]) + ')')
        #plt.savefig(decon_filename + 'convergence_plot_' + '_SPXIL_depth_0.8cm_Slice_' + str(s) +'_Position_' + str(p) + '_iter_' + str(decon_results[vector_position][1][-1]) + '_paddded3.png')
        #plt.close()
    
        np.save(decon_filename + 'convergence_#images_' + '_SPXIL_depth_0.8cm_Slice_' + str(s) +'_Position_' + str(p) + '_iter_' + str(decon_results[vector_position][1][-1]) + '_paddded3.npy', [decon_results[vector_position][1], decon_results[vector_position][2]])
    
        conv = [decon_results[vector_position][1], decon_results[vector_position][2]]
        conv_list = [list(x) for x in zip(*conv)]
    
        f = open(decon_filename + 'convergence_#images_' + '_PBXIL_depth_0.8cm_Slice_' + str(s) +'_Position_' + str(p) + '_iter_'+ str(decon_results[vector_position][1][-1]) + '_paddded3.txt', 'w')
        f.write(tabulate(conv_list ,['Number of Iterations', 'Convergence Parameter'], numalign="center"))
        f.close()
   
print('chcekpoint')

downsample_recon3D = np.zeros((5,315,23))
    
for s in xrange(1,totalslices+1):
    for p in xrange(1,totalpositions+1):
        for i in xrange(1, 316):
            x = i*1.625
            a = int(np.floor(x))       
            downsample_recon3D[s-1,i-1,p-1]=(x-a)*recon3D[s-1,a-1,p-1]+(a+1-x)*recon3D[s-1,a,p-1]
 
    
np.save(decon_filename + 'recon3D_array_phantomscan.npy', [recon3D]) 
np.save(decon_filename + 'recon3D_array_downsample_phantomscan.npy', [downsample_recon3D])

for s in xrange(1, totalslices+1):
    slice_recon = recon3D[s-1, :, :]

    save_filename = (decon_filename + 'Slice_' + str(s) + '_Reconstruction.tif')
    tiff = TIFF.open(save_filename, mode='w')
    tiff.write_image(slice_recon)
    tiff.close()
    
    slice_recon_downsample = downsample_recon3D[s-1, :, :]
    
    save_filename = (decon_filename + 'Slice_' + str(s) + '_Reconstruction_downsample.tif')
    tiff = TIFF.open(save_filename, mode='w')
    tiff.write_image(slice_recon_downsample)
    tiff.close()
