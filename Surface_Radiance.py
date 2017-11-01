#Normalized Surface Radiance function
#Written by Bryan Quigley 10/24/17 with guidance from Corey Smith
#Normalized Surface Radiance function presented by Rice et al. in 'In vivo imaging of light-emitting probes', J Biomed Opt. 2001

from __future__ import division
import numpy

def diffusion_kernel(depth, imagesize, pixelsize, ua, uss, norm = True):
    imagesize = numpy.arry(image_size)
    xhalf_width = imagesize[1]/2
    yhalf_width = imagesize[0]/2
    xx = np.arange(-xhalf_width, xhalf_width)
    yy = np.arange(-yhalf_width, yhalf_width)
    X, Y = numpy.meshgrid(xx, yy)
    R = numpy.sqrt(X**2 + Y**2)
    kernel = Nsurfrad(R, depth, ua, uss)

    if norm == True:
        kernel /= kernel.sum()
    return kernel

def Nsurfrad(r, z, ua, uss):
    z = depth
    D = 1 / (3 * (ua + uss))
    ueff = sqrt(3 * ua * (uss + ua))
    Reff = 0.493
    zb = (1 + Reff) / (1 - Reff) * 2 / (3 * (ua + uss))
    r1 = sqrt(r ** 2 + z ** 2)
    r2 = sqrt(r ** 2 + (z + 2 * zb) ** 2)
    N(r, z) = 1/(4*numpy.pi*4*numpy.pi*D)*(np.exp(-ueff*r1)/r1-np.exp(-ueff*r2)/r2+3*D*(z/(r1**2)*(ueff+1/r1)*np.exp(-ueff*r1)+(z+2*zb)/(r2**2)*(ueff+1/r2)*np.exp(-ueff*r2)))
    return N
