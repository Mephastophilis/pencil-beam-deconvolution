"""
X-ray Luminescence Pencil Beam Deconvolution using XF prior information reconstruction
Written by Bryan Quigley
"""

from __future__ import division

import sys
sys.path.insert(0,"/home/bquigley/Research/XFXL Simulation 1.0.0")
from PIL import Image
import numpy as np
from libtiff import TIFF
import PB_XLXF_Richardson_Lucy as PBRL
#import removeHotCold as rHC
from multiprocessing import Pool
import os

pix_size = 0.021586 #cm (215.86 um)


mask = np.load('XF_reconstruction/XF_recon.npy')
downsample_mask = np.zeros((23,512))

#tube 1 downsample [long calculation]

downsample_mask[:,255]=np.sum(mask[:,504:512], axis=1)+0.239*mask[:,503]
downsample_mask[:,254]=0.761*mask[:,503]+np.sum(mask[:,496:503], axis = 1) + 0.478*mask[:,495]
downsample_mask[:,253]=0.522*mask[:,495]+np.sum(mask[:,488:495], axis = 1) + 0.717*mask[:,487]
downsample_mask[:,252]=0.283*mask[:,487]+np.sum(mask[:,480:487], axis = 1) + 0.956*mask[:,479]
downsample_mask[:,251]=0.044*mask[:,479]+np.sum(mask[:,471:479], axis = 1) + 0.195*mask[:,470]
downsample_mask[:,250]=0.805*mask[:,470]+np.sum(mask[:,463:470], axis = 1) + 0.434*mask[:,462]
downsample_mask[:,249]=0.566*mask[:,462]+np.sum(mask[:,455:462], axis = 1) + 0.673*mask[:,454]
downsample_mask[:,248]=0.327*mask[:,454]+np.sum(mask[:,447:454], axis = 1) + 0.912*mask[:,446]
downsample_mask[:,247]=0.088*mask[:,446]+np.sum(mask[:,438:446], axis = 1) + 0.151*mask[:,437]

#tube 2 downsample
downsample_mask[:,256]=np.sum(mask[:,512:520], axis = 1)+mask[:,520]*(0.239)
for i in xrange(61):
    downsample_mask[:,257+i]=(1-np.mod((i+1)*0.239,1))*mask[:, int(np.floor(512+8.239*(i+1)))]+np.sum(mask[:, int(np.floor(512+8.239*(i+1)))+1 :int(np.floor(512+8.239*(i+1)))+8], axis=1) + np.mod((i + 2) * 8.239, 1) * mask[:, int(np.floor(512+8.239*(i+2)))]


downsample_mask[downsample_mask < 40] = 0
downsample_mask[downsample_mask >= 40] = 1
np.save('XF_reconstruction/binarymaskXF.npy', downsample_mask)
binarymaskXF = np.load('XF_reconstruction/binarymaskXF.npy')


darkarray = np.zeros((512,512,19))
for i in xrange(1,20):
  if i < 11:
    darkarray[:,:,i-1] = np.load('XL_Simulated_images/Sim_XL_position_' + str(i) + '.npy')
  else:
    darkarray[:,:,i-1] = np.load('XL_Simulated_images/Sim_XL_position_' + str(i+4) + '.npy')

#first try around I won't attempt to remove hot and dead pixels
#background = rHC.remove_hot_and_dead_pixels(darkarray[:,:,0], 1700, remove_dead=True,lower_thresh=200)

darkmean = np.mean(darkarray,2)
variance = np.std(darkarray[:,:,0]) ** 2

if not os.path.exists('Simulated_PB_Decon_Results'):
    os.makedirs('Simulated_PB_Decon_Results')
decon_filename='/home/bquigley/Research/XFXL Simulation 1.0.0/Simulated_PB_Decon_Results/159mgmL_'

recon3D = np.zeros((512,23))

decon_inputs = []
totalpositions = 23

for p in xrange(1, totalpositions+1):

    detectedimage = np.load('XL_Simulated_images/Sim_XL_position_' + str(p) + '.npy')
    detectedimage_p = detectedimage - darkmean
    detectedimage_p += variance
    detectedimage_p[detectedimage_p < 0] = 0

    decon_inputs.append([binarymaskXF, p, detectedimage_p, 0.8, pix_size, 0.15, 5, variance, decon_filename, [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 100, 150, 200], 1e-6, 50, 4])

pp = Pool()
decon_results = pp.map(PBRL.RL_decon, decon_inputs[:][:])

for x in xrange(1, len(decon_results[0][5])+1):

    a=decon_results[1][3][256,x-1]

    recon3D=np.zeros((23,512))
    recon3D = recon3D + a
    recon3D_downsampled=np.zeros((23,552))
    recon3D_downsampled = recon3D_downsampled + a


    for p in xrange(1, totalpositions+1):
        vector_position=(p-1)
        recon3D[p-1,:]=decon_results[vector_position][3][:,x-1]
        recon3D_downsampled[p-1,:]=decon_results[vector_position][4][:,x-1]
	recon3D[np.isnan(recon3D)] = 0
	recon3D_downsampled[np.isnan(recon3D_downsampled)] = 0




    save_filename = (decon_filename + 'Reconstruction_iterations' + str(decon_results[0][5][x-1]))
    tiff = TIFF.open(save_filename+ '.tif', mode='w')
    tiff.write_image(recon3D[:,:].astype(dtype=np.float32))
    tiff.close()
    np.save(save_filename, recon3D)

    save_filename = (decon_filename + 'Reconstruction_upsample_iterations' + str(decon_results[0][5][x-1]))
    tiff = TIFF.open(save_filename+ '.tif', mode='w')
    tiff.write_image(recon3D_downsampled[:,:].astype(dtype=np.float32))
    tiff.close()
    np.save(save_filename, recon3D_downsampled)





print(variance)
print('done! code ran successfully')
