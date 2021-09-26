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
# cv_cover = cv_cover[::4,::4,:]
# cv_desk = cv_desk[::4,::4,:]
# hp_cover = hp_cover[::4,::4,:]
    #plt.imshow(cv_cover)
    #plt.show()

# Match cv cover and cv desk
# matches, locs1, locs2 = matchPics.matchPics(cv_cover, cv_desk, opts)
# np.savez_compressed('default_match.npz',
#     matches=matches,
#     locs1=locs1,
#     locs2=locs2
# )
default_match = np.load('default_match.npz')
locs1=default_match['locs1']
locs2=default_match['locs2']
matches=default_match['matches']
plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# Use matching locs1 and locs2 to compute H
H, inliers = planarH.computeH_ransac(locs2[matches[:,1]], locs1[matches[:,0]], opts)
print("H ({}):\n{}".format(inliers,H/np.max(H)))

# Use H to transform hp cover, with same output size as cv desk
    # H = np.array([[-1.85585438e+00,  1.00000000e+00, -1.70945158e+02],
    #  [ 7.77723162e-02, -5.50583782e-01, -1.35101819e+02],
    #  [ 3.51098940e-04,  1.04850911e-02, -2.75717997e+00]])
new_hp_cover = cv2.warpPerspective(cv_cover, H, cv_desk.shape[0:2])
# new_hp_cover = np.zeros(cv_desk.shape)
# for row in range(0,hp_cover.shape[0]):
#     for col in range(0,hp_cover.shape[1]):
#         new_coord = (H @ np.array([col, row, 1]))
#         new_coord = (new_coord/new_coord[-1]).astype('int')
#         if((new_coord[0] > 0) and (new_coord[0] < new_hp_cover.shape[1]) and (new_coord[1] > 0) and (new_coord[1] < new_hp_cover.shape[0])):
#             new_hp_cover[new_coord[1],new_coord[0],:] = hp_cover[row,col,:]
plt.imshow(new_hp_cover)
plt.show()

