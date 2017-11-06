  #Richardson Lucy Pencil Beam Deconvolution function
#Written by Bryan Quigley with guidance from Corey Smith

from __future__ import division

import sys
sys.path.insert(0, "/home/faustus/Documents/Deconvolution_Code")

import numpy as np
import scipy.misc
import scipy.signal
import time
from libtiff import TIFF
import Surface_Radiance

def RL_decon(detected_image, depth, pixelsize, ua, uss, background, decon_filename,
             save_after=[0], stopping_condition=10e-5, maxIter_Number=1000,
             padsize=4):

    #starting the clock
    start_time = time.time()

    #first element is a uniform image
    num_elem = detected_image.size
    imagesize = detected_image.shape
    image_flux = np.sum(np.abs(detected_image - background))
    estimate = (image_flux/num_elem)*np.ones(imagesize)
    time_estimate = time.time()

    #Tracking the convergence
    old = np.copy(estimate)
    iteration_vector = []
    difference_vector = []

    print "background", background

    #creating the kernel
    kernel = Surface_Radiance.diffusion_kernel(depth, np.array(imagesize) * padsize, pixelsize, ua, uss)

    tiff = TIFF.open('/home/faustus/Documents/Deconvolution_Code/kernel/norm_surf_rad.tif', mode='w')
    tiff.write_image(kernel)
    tiff.close

    #kernel transpose
    flipped_kernel = np.copy(kernel[::-1,::-1])

    #Deconvolution Loop
    for iteration in range(maxIter_Number):
        start = time.time()
        actual_iteration= iteration + 1

        #Computing Af
        estimated_blur = ConvFFT(estimate, kernel, padsize)

        #computing g/(Af)
        comparing_data = (detected_image/(estimated_blur + background))

        #computing A^t * (g/(Af))
        correction = ConvFFT(comparing_data, flipped_kernel, padsize)
        estimate *= correction
        estimate *= image_flux/np.sum(estimate)

        #Convergence
        l2_diff = np.sqrt(np.sum((estimate - old) ** 2))
        l2_estimate = np.sqrt(np.sum(estimate ** 2))
        difference = l2_diff / l2_estimate
        iteration_vector.append(actual_iteration)
        difference_vector.append(difference)

        if (difference < stopping_condition):
            save_filename = (decon_filename + '_iterations_' + str(actual_iteration) + '.tif')
            tiff = TIFF.open(save_filename, mode='w')
            tiff.write_image(estimate)
            tiff.close()
            break

        old = np.copy(estimate)
        finish = time.time()

        if (any(x == actual_iteration for x in save_after)):
            save_filename = (decon_filename + '_iterations_' + str(actual_iteration) + '.tif')
            tiff = TIFF.open(save_filename, mode='w')
            tiff.write_image(estimate)
            tiff.close()

        print "Time for %i iteration was %.3f seconds" %(actual_iteration, (finish - start))

    finish_time = time.time()
    print "Total elapsed time %f for RL Deconvolution" %(finish_time - start_time)

    return estimate, iteration_vector, difference_vector

def ConvFFT(image, kernel, padsize):
    img_shape = image.shape
    kernel = Shift(kernel)
    image = addpad(image, padsize)

    tiff = TIFF.open('/home/faustus/Documents/Deconvolution_Code/kernel/padded_images.tif', mode='w')
    tiff.write_image(image)
    tiff.close()

    kern_fft = np.fft.fft2(kernel)
    image_fft = np.fft.fft2(image)
    conv_fft = kern_fft * image_fft
    conv = np.fft.ifft2(conv_fft)
    conv = cutpad(np.real(conv), img_shape)
    conv[conv <0] = 0
    return conv

def addpad(f, padsize):
    # extend the image by the padsize using constant background values
    img_size = np.array(f.shape)
    f_addpad = np.mean(f[:,0]) * np.ones(img_size * padsize)
    f_addpad[:img_size[0], :img_size[1]] = f
    return f_addpad

def Shift(kernel):
    kernel = np.array(kernel)
    kSize = np.array(kernel.shape)
    Kpad = np.zeros_like(kernel)
    Kpad = kernel
    # Move the center of the kernel to (0,0)
    if kSize[0] % 2 == 0:
        Kpad = np.roll(Kpad, -kSize[0]//2, axis=0)
    else:
        Kpad = np.roll(Kpad, -kSize[0]//2 + 1 , axis=0)

    if kSize[1] % 2 == 0:
        Kpad = np.roll(Kpad, -kSize[1]//2 , axis=1)
    else:
        Kpad = np.roll(Kpad, -kSize[1]//2 + 1 , axis=1)
    return Kpad

def cutpad(f,img_size):
    # Crop an image to its top-left quarter
    f_cut = f[:img_size[0],:img_size[1]]
    return f_cut
