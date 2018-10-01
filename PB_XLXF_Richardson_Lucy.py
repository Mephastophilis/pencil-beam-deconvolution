#Richardson Lucy Pencil Beam Deconvolution function
#Written by Bryan Quigley with guidance from Corey Smith

from __future__ import division

import sys
sys.path.insert(0, "/home/bquigley/Research/XFXL Simulation 1.0.0")

import numpy as np
import time
from libtiff import TIFF
import Surface_Radiance

def RL_decon(x):

    binarymaskXF = x[0]
    pos = x[1]
    detected_image=x[2]
    depth=x[3]
    pixelsize=x[4]
    ua=x[5]
    uss=x[6]
    background=x[7]
    decon_filename=x[8]
    save_after=x[9]
    stopping_condition=x[10]
    maxIter_Number=x[11]
    padsize=x[12]

    SA_count=0

    deconPB_SA=np.zeros((512,len(save_after)))
    downsample_deconPB_SA = np.zeros((552,len(save_after)))


    #starting the clock
    start_time = time.time()

    #first element is a uniform image
    imagesize = detected_image.shape
    image_flux = np.sum(np.abs(detected_image[256,:] - background))
    estimate = np.zeros((512,512))
    estimate[256,:] = binarymaskXF[pos-1,:]*image_flux / np.sum(binarymaskXF)

    #Tracking the convergence
    old = np.copy(estimate)
    iteration_vector = []
    difference_vector = []

    #print "background", background

    #creating the kernel
    kernel = Surface_Radiance.diffusion_kernel(depth, np.array(imagesize) * padsize, pixelsize, ua, uss)

    tiff = TIFF.open('/home/bquigley/Research/XFXL Simulation 1.0.0/kernel/norm_surf_rad.tif', mode='w')
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
        estimate *= (image_flux/np.sum(estimate))


        #Convergence
        l2_diff = np.sqrt(np.sum((estimate - old) ** 2))
        l2_estimate = np.sqrt(np.sum(estimate ** 2))
        difference = l2_diff / l2_estimate
        iteration_vector.append(actual_iteration)
        difference_vector.append(difference)

        if (difference < stopping_condition):
            save_filename = (decon_filename + '_Position_' + str(pos) + '_iterations_' + str(actual_iteration) + '.tif')
            tiff = TIFF.open(save_filename, mode='w')
            tiff.write_image(estimate.astype(dtype=np.float32))
            tiff.close()
            break

        old = np.copy(estimate)
        finish = time.time()

        if (any(x == actual_iteration for x in save_after)):


            save_filename = (decon_filename + '_Position_' + str(pos) + '_iterations_' + str(actual_iteration) + '.tif')
            tiff = TIFF.open(save_filename, mode='w')
            tiff.write_image(estimate.astype(dtype=np.float32))
            tiff.close()

            deconPB_SA[:,SA_count] = estimate[256,:]

            print('checkpoint')

            for i in xrange(1, 553):
                x = i*0.9265188068
                a = int(np.floor(x))
                downsample_deconPB_SA[i-1,SA_count]=(x-a)*deconPB_SA[a-1,SA_count]+(a+1-x)*deconPB_SA[a,SA_count]


            SA_count=SA_count+1


        finish = time.time()
        print "Time for %i iteration was %.3f seconds (Position %i)" %(actual_iteration, (finish - start), pos)

    finish_time = time.time()
    print "Total elapsed time %f for RL Deconvolution (Position %i)" %((finish_time - start_time), pos)

    save_filename = (decon_filename + '_Position_' + str(pos) + '_iterations_' + str(actual_iteration) + '.tif')
    tiff = TIFF.open(save_filename, mode='w')
    tiff.write_image(estimate.astype(dtype=np.float32))
    tiff.close()

    return [estimate, iteration_vector, difference_vector, deconPB_SA, downsample_deconPB_SA, save_after]

def ConvFFT(image, kernel, padsize):
    img_shape = image.shape
    kernel = Shift(kernel)
    image = addpad(image, padsize)
    tiff = TIFF.open('/home/bquigley/Research/XFXL Simulation 1.0.0/kernel/padded_images.tif', mode='w')
    tiff.write_image(image.astype(dtype=np.float32))
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
