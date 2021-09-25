import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
    #I1, I2 : Images to match
    #opts: input opts
    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

    #Convert Images to GrayScale
    togray = lambda img: 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
    I1g = togray(I1)
    I2g = togray(I2)
    
    
    #Detect Features in Both Images
    
    
    #Obtain descriptors for the computed feature locations
    

    #Match features using the descriptors
    

    return matches, locs1, locs2
