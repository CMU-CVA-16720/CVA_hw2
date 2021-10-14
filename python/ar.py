import numpy as np
import cv2
#Import necessary functions
import matchPics
import planarH
import matplotlib.pyplot as plt
from helper import plotMatches
from opts import get_opts
from loadVid import loadVid
from datetime import datetime
import sys


opts = get_opts()

# Get files
ar_vid = loadVid("../data/ar_source.mov")
book_vid = loadVid("../data/book.mov")
cv_cover = cv2.imread('../data/cv_cover.jpg')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# Cropping AR video to get center, and have same aspect ratio as cv cover
ar_frame = np.zeros((ar_vid.shape[1],int(ar_vid.shape[1]*cv_cover.shape[1]/cv_cover.shape[0])))
ar_offset = int((ar_vid.shape[2]-ar_frame.shape[1])/2)

# AR to AVI
print("Start time: {}".format(datetime.now()))
out = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc(*'mp4v'), 25, (book_vid[0].shape[1], book_vid[0].shape[0]))
total_frames = min(ar_vid.shape[0],book_vid.shape[0])
for f_indx in range(0,total_frames):
    print("Progress: {}/{} ({} %)".format(f_indx, total_frames, 100*(f_indx)/total_frames))
    print("Progress: {}/{} ({} %)".format(f_indx, total_frames, 100*(f_indx)/total_frames),file=sys.stderr)
    # Get frame from book vid and ar source vid
    book_frame = book_vid[f_indx]
    ar_frame = ar_vid[f_indx,:, ar_offset:ar_offset+ar_frame.shape[1],:]
    # Match cv cover to book frame
    matches, locs1, locs2 = matchPics.matchPics(cv_cover, book_frame, opts)
    # Compute H for current frame, scaling as necessary
    xy1 = np.zeros(locs1.shape)
    xy1[:,0] = locs1[:,1]*ar_frame.shape[1]/cv_cover.shape[1]
    xy1[:,1] = locs1[:,0]*ar_frame.shape[0]/cv_cover.shape[0]
    xy2 = np.zeros(locs2.shape)
    xy2[:,0] = locs2[:,1]
    xy2[:,1] = locs2[:,0]
    H, inliers = planarH.computeH_ransac(xy1[matches[:,0]],xy2[matches[:,1]], opts)
    print("H ({}):\n{}\n".format(inliers,H))
    # Apply projection
    composite = planarH.compositeH(H, ar_frame, book_frame)
    # Save result
    out.write(composite)
out.release()
print("Complete!")
print("End time: {}".format(datetime.now()))



