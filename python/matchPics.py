import numpy as np
import cv2
import skimage.color
from helper import briefMatch, plotMatches
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
    locs1 = corner_detection(I1g,sigma)
    locs2 = corner_detection(I2g,sigma)
    
    #Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1g,locs1)
    desc2, locs2 = computeBrief(I2g,locs2)

    #Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)
    print("# of matches: {}".format(matches.shape[0]))

    return matches, locs1, locs2
