#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:19:56 2018

@author: faustus
"""

import numpy as np
import matplotlib.pyplot as plt

XFXL_sim_reconstructions = []


XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations1.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations2.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations5.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations6.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations8.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations10.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations12.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations14.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations15.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations16.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations18.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations20.npy'))
XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations25.npy'))
#XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations30.npy'))
#XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations35.npy'))
#XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations40.npy'))
#XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations45.npy'))
#XFXL_sim_reconstructions.append(np.load('159mgmL_Reconstruction_iterations50.npy'))

iteration_list = [1, 2, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 25, 30]

tube1_CNRs = []
tube2_CNRs = []

for i in range(np.shape(XFXL_sim_reconstructions)[0]):
    tube1_CNRs.append( np.mean(XFXL_sim_reconstructions[i][11:13,2043:2065]) / np.std(XFXL_sim_reconstructions[i][11:13,2043:2065]) )
    tube2_CNRs.append( np.mean(XFXL_sim_reconstructions[i][11:13,2152:2174]) / np.std(XFXL_sim_reconstructions[i][11:13,2152:2174]) )
    print('CNR for tube 1 in Simulation ' + str(iteration_list[i]) + ' iterations is: ' + str(tube1_CNRs[i]))
    print('CNR for tube 2 in Simulation ' + str(iteration_list[i]) + ' iterations is: ' + str(tube2_CNRs[i]))
    
    
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar

for i in range(np.shape(XFXL_sim_reconstructions)[0]):
    fig = plt.figure()
    gs  = gridspec.GridSpec(3, 4, wspace=0.4, hspace=0.3)
    fig.add_subplot(1, 1, 1)
    plt.imshow(XFXL_sim_reconstructions[i][:,1935:2281],  aspect='auto', cmap='gist_gray')
    plt.colorbar(orientation='vertical')
    scalebar = ScaleBar(0.0262, 'mm', location='lower right', color='w', box_color='k')
    plt.gca().add_artist(scalebar)
    CNRtext = ' CNR Tube 1: ' + str(tube1_CNRs[i]) + '\n CNR Tube 2: ' + str(tube2_CNRs[i])
    plt.text(0, 3, CNRtext, color='w')
    plotname = 'Simualted_XFXL_recon_iterations_' + str(iteration_list[i]) + '.png'
    fig.savefig(plotname, transparent=False, dpi=200, bbox_inches="tight")
    

#Plotting data all in one figure
    
#Simulation XL, XF, and XLXF reconstructions.
Sim_XL_decon = np.load('Sim_XLonly_Reconstruction_iterations1000.npy')
Sim_XF_recon = np.load('XF_sim_recon_100.npy')
Sim_XFpoor_recon = np.load('XF_sim_recon_10.npy')
Sim_XLXF_recon = np.load('159mgmL_Reconstruction_iterations25.npy')
Sim_XLXFpoor_recon = np.load('Sim_XLXFpoor_Reconstruction_iterations25.npy')

fig=plt.figure()
gs  = gridspec.GridSpec(3, 4, wspace=0.4, hspace=0.3)

a = fig.add_subplot(gs[0, 1:3])
plt.imshow(Sim_XL_decon[:,235:277], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.211586, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'a', color='w')

a = fig.add_subplot(gs[1, 0:2])
plt.imshow(Sim_XF_recon[:,342:688], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.0262, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'b', color='w')

a = fig.add_subplot(gs[1, 2:4])
plt.imshow(Sim_XFpoor_recon[:,342:688], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.0262, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'c', color='w')

a = fig.add_subplot(gs[2, 0:2])
plt.imshow(Sim_XLXF_recon[:,1935:2281], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.0262, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'd', color='w')

a = fig.add_subplot(gs[2, 2:4])
plt.imshow(Sim_XLXFpoor_recon[:,235:277], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.211586, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'e', color='w')

fig.savefig('Simulation_reconstructions.png', transparent=False, dpi=200, bbox_inches="tight")
    
#TIFF images for Patrick

from libtiff import TIFF

image_name_list = ['XL_only_deconvolution', 'XF_reconstruction_100', 'XF_reconstruction_10', 'XFXL_joint_deconvolution_new', 'XFXL_joint_deconvolution_old']

Sim_XL_decon = np.load('Sim_XLonly_Reconstruction_iterations1000.npy')
Sim_XF_recon = np.load('XF_sim_recon_100.npy')
Sim_XFpoor_recon = np.load('XF_sim_recon_10.npy')
Sim_XLXF_recon = np.load('159mgmL_Reconstruction_iterations25.npy')
Sim_XLXFpoor_recon = np.load('Sim_XLXFpoor_Reconstruction_iterations25.npy')


tiff = TIFF.open(image_name_list[0] + '.tiff', mode = 'w')
tiff.write_image(Sim_XL_decon[:,235:277])
tiff.close

tiff = TIFF.open(image_name_list[1] + '.tiff', mode = 'w')
tiff.write_image(Sim_XF_recon[:,342:688])
tiff.close

tiff = TIFF.open(image_name_list[2] + '.tiff', mode = 'w')
tiff.write_image(Sim_XFpoor_recon[:,342:688])
tiff.close

tiff = TIFF.open(image_name_list[3] + '.tiff', mode = 'w')
tiff.write_image(Sim_XLXF_recon[:,1935:2281])
tiff.close

tiff = TIFF.open(image_name_list[4] + '.tiff', mode = 'w')
tiff.write_image(Sim_XLXFpoor_recon[:,235:277])
tiff.close






# Resizing XF image and calculating CNR metric
import scipy.ndimage
resize = 0.0262/0.211586
#image = Sim_XF_recon[:,342:688]
image_zoom = scipy.ndimage.zoom(Sim_XF_recon, (1, resize))


#image_2 = Sim_XF_recon[:,342:688]
image_2_zoom = scipy.ndimage.zoom(Sim_XFpoor_recon, (1, resize))


fig_2=plt.figure()
gs  = gridspec.GridSpec(3, 4, wspace=0.4, hspace=0.3)

a = fig_2.add_subplot(gs[0, 1:3])
plt.imshow(Sim_XL_decon[:,235:277], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.211586, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'a', color='w')

a = fig_2.add_subplot(gs[1, 0:2])
plt.imshow(image_zoom[:,42:84], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.211586, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'b', color='w')

a = fig_2.add_subplot(gs[1, 2:4])
plt.imshow(image_2_zoom[:,42:84], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.211586, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'c', color='w')

a = fig_2.add_subplot(gs[2, 0:2])
plt.imshow(Sim_XLXF_recon[:,1935:2281], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.0262, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'd', color='w')

a = fig_2.add_subplot(gs[2, 2:4])
plt.imshow(Sim_XLXFpoor_recon[:,235:277], aspect='auto', cmap='gist_gray')
plt.colorbar(orientation='vertical')
a.set_xticks([])
a.set_yticks([])
scalebar = ScaleBar(0.211586, 'mm', location='lower right', color='w', box_color='k')
plt.gca().add_artist(scalebar)
plt.text(0, 3, 'e', color='w')

fig_2.savefig('Simulation_reconstructions_XF_resized.png', transparent=False, dpi=200, bbox_inches="tight")









    