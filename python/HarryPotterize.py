import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
import matchPics
import planarH
import matplotlib.pyplot as plt
from helper import plotMatches

#Write script for Q2.2.4

# Get options
opts = get_opts()

# Get relevant images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
hp_cover = hp_cover[:,:,[2,1,0]]

# Match cv cover and cv desk
matches, locs1, locs2 = matchPics.matchPics(cv_cover, cv_desk, opts)
# np.savez_compressed('default_match.npz',
#     matches=matches,
#     locs1=locs1,
#     locs2=locs2
# )
# default_match = np.load('default_match.npz')
# locs1=default_match['locs1']
# locs2=default_match['locs2']
# matches=default_match['matches']

# Plot matches
# plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# convert (row,col) to (x,y), scaling for new image
xy1 = np.zeros(locs1.shape)
xy1[:,0] = locs1[:,1]*hp_cover.shape[1]/cv_cover.shape[1]
xy1[:,1] = locs1[:,0]*hp_cover.shape[0]/cv_cover.shape[0]
xy2 = np.zeros(locs2.shape)
xy2[:,0] = locs2[:,1]
xy2[:,1] = locs2[:,0]

# Use matching locs1 and locs2 to compute H
H, inliers = planarH.computeH_ransac(xy1[matches[:,0]],xy2[matches[:,1]], opts)
print("H ({}):\n{}".format(inliers,H/np.max(H)))

# Use H to transform hp cover, with same output size as cv desk
composite = planarH.compositeH(H, hp_cover, cv_desk)
plt.imshow(composite)
plt.show()

