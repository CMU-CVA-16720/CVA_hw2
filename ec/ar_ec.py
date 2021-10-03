import numpy as np
import cv2
#Import necessary functions
import matplotlib.pyplot as plt
import sys
sys.path.append('../python')
import matchPics
import planarH
from helper import plotMatches
from opts import get_opts


def compositePano(H2to1, addition, base):
    
    # Combine images together; greater intensity gets priority

    # Warp addition by appropriate homography
    addition_warp = cv2.warpPerspective(addition, np.linalg.inv(H2to1), (base.shape[1],base.shape[0]))

    # Fuse images
    togray = lambda img: 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
    base_g = togray(base)
    add_g = togray(addition_warp)
    replace_coord = add_g>base_g
    composite_base = base
    composite_base[replace_coord,:] = addition_warp[replace_coord,:]
    
    return composite_base

# Get options
opts = get_opts()

# Get relevant images
cv_cover = cv2.imread('panoR.jpg')
cv_desk = cv2.imread('panoL.jpg')
temp = np.zeros((cv_desk.shape[0]*2,cv_desk.shape[1]*2,cv_desk.shape[2])).astype('int')

temp[cv_desk.shape[0]//2:cv_desk.shape[0]+cv_desk.shape[0]//2
,cv_desk.shape[1]//2:cv_desk.shape[1]+cv_desk.shape[1]//2,:]=cv_desk
cv_desk=temp
# plt.imshow(cv_desk)
# plt.show()

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
# plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# convert (row,col) to (x,y), scaling for new image
xy1 = np.zeros(locs1.shape)
xy1[:,0] = locs1[:,1]
xy1[:,1] = locs1[:,0]
xy2 = np.zeros(locs2.shape)
xy2[:,0] = locs2[:,1]
xy2[:,1] = locs2[:,0]

# Use matching locs1 and locs2 to compute H
H, inliers = planarH.computeH_ransac(xy1[matches[:,0]],xy2[matches[:,1]], opts)
print("H ({}):\n{}".format(inliers,H/np.max(H)))

# Use H to transform hp cover, with same output size as cv desk
composite = compositePano(H, cv_cover, cv_desk)
plt.imshow(composite)
plt.show()