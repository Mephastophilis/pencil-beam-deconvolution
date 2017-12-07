# -*- coding: utf-8 -*-
"""
Program for removing hot or dead pixels from a 2D image. Pixels above a 
threshold are set to nan so that they can be identified and set to the median
value of the nearest neighbors. Becasue there is a chance that not all nan 
values will be replaced on first pass, there is a built in while loop that 
runs until there are no longer any nan values in the new image.

Created on Fri Apr 8 13:44:00 2016
@author: CSmith
"""

import numpy as np

def remove_hot_and_dead_pixels(image, upper_thresh, remove_dead=False,
                               lower_thresh=0):
    
    new_image = np.array(image, dtype=np.float64)
    xSz, ySz = new_image.shape
    
    new_image[new_image > upper_thresh] = float('nan')
    
    if remove_dead == True:
        new_image[new_image == 0] = float('nan')
    if lower_thresh > 0:
        new_image[new_image < lower_thresh] = float('nan')
        
    # This while loop checks for any nan's and if there are any then it
    # replaces them with the median of its neighbors, might have to cycle
    # multiple times if there are many pixels to replace     
    while np.isnan(np.sum(new_image)):
        for i in range(xSz):
            for j in range(ySz):
                
                if np.isnan(new_image[i,j]):
                    
                    if (j == 0):
                        if i == 0:
                            new_image[i,j] = np.nanmedian([new_image[i + 1,j],\
                            new_image[i,j + 1], new_image[i + 1, j + 1]])
                        elif i == (xSz - 1):
                            new_image[i,j] = np.nanmedian([new_image[i - 1,j],\
                            new_image[i,j + 1], new_image[i - 1, j + 1]])
                        else:
                            new_image[i,j] = np.nanmedian([new_image[i - 1,j],\
                            new_image[i + 1,j], new_image[i,j + 1], \
                            new_image[i + 1, j + 1], new_image[i - 1, j + 1]])
                    
                    elif j == (ySz - 1):
                        if i == 0:
                            new_image[i,j] = np.nanmedian([new_image[i,j - 1],\
                            new_image[i + 1,j], new_image[i + 1, j - 1]])
                        elif i == (xSz - 1):
                            new_image[i,j] = np.nanmedian([new_image[i - 1,j],\
                            new_image[i,j - 1], new_image[i - 1, j - 1]])
                        else:
                            new_image[i,j] = np.nanmedian([new_image[i - 1,j],\
                            new_image[i,j - 1], new_image[i - 1, j - 1], \
                            new_image[i + 1,j], new_image[i + 1, j - 1]])
                            
                    elif (j == (ySz - 1)):
                        new_image[i,j] = np.nanmedian([new_image[i - 1,j], \
                        new_image[i,j - 1], new_image[i - 1, j - 1], \
                        new_image[i + 1,j], new_image[i + 1, j - 1]])
                        
                    elif (i == (xSz - 1)):
                        new_image[i,j] = np.nanmedian([new_image[i - 1,j], \
                        new_image[i,j - 1], new_image[i - 1, j - 1], \
                        new_image[i,j + 1], new_image[i - 1, j + 1]])
                        
                    else:
                        new_image[i,j] = np.nanmedian([new_image[i - 1,j], \
                        new_image[i,j - 1], new_image[i - 1, j - 1], \
                        new_image[i + 1,j], new_image[i,j + 1], \
                        new_image[i + 1, j + 1], new_image[i + 1, j - 1], \
                        new_image[i - 1, j + 1]])
                
    return new_image