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

# # Get files
# ar_vid = loadVid("../data/ar_source.mov")
# book_vid = loadVid("../data/book.mov")
# cv_cover = cv2.imread('../data/cv_cover.jpg')
# hp_cover = cv2.imread('../data/hp_cover.jpg')

# # Save for speed
# np.savez_compressed('ar_files.npz',
#     ar_vid=ar_vid,
#     book_vid=book_vid,
#     cv_cover=cv_cover,
#     hp_cover=hp_cover
# )

# Load files
ar_files = np.load('ar_files.npz')
ar_vid=ar_files['ar_vid']
book_vid=ar_files['book_vid']
cv_cover=ar_files['cv_cover']
hp_cover=ar_files['hp_cover']
# print("ar_vid shape  : {}".format(ar_vid.shape))
# print("book_vid shape: {}".format(book_vid.shape))
# print("cv_cover shape: {}".format(cv_cover.shape))

# Get frames of interest for testing
frames_of_interest = [500, 225, 325] # book at left, center and right at these frames
# for frame in frames_of_interest:
#     plt.imshow(book_vid[frame])
#     plt.show()

# Cropping AR video to get center, and have same aspect ratio as cv cover
ar_frame = np.zeros((ar_vid.shape[1],int(ar_vid.shape[1]*cv_cover.shape[1]/cv_cover.shape[0])))
ar_offset = int((ar_vid.shape[2]-ar_frame.shape[1])/2)
# print("ar frame shape: {}".format(ar_frame.shape))
# print("Frame start: {}".format(ar_offset))
# for frame in frames_of_interest:
#     plt.imshow(ar_vid[frame])
#     plt.show()
#     # Get cropped frame
#     ar_frame = ar_vid[frame,:, ar_offset:ar_offset+ar_frame.shape[1],:]
#     plt.imshow(ar_frame)
#     plt.show()

# # Compute H for frames of interest
# for frame in frames_of_interest:
#     # Get frame from book vid and ar source vid
#     book_frame = book_vid[frame]
#     ar_frame = ar_vid[frame,:, ar_offset:ar_offset+ar_frame.shape[1],:]
#     print("ar frame shape: {}".format(ar_frame.shape))
#     plt.imshow(ar_frame)
#     plt.show()
#     # Match cv cover to book frame
#     matches, locs1, locs2 = matchPics.matchPics(cv_cover, book_frame, opts)
#     # Compute H for current frame, scaling as necessary
#     xy1 = np.zeros(locs1.shape)
#     xy1[:,0] = locs1[:,1]*ar_frame.shape[1]/cv_cover.shape[1]
#     xy1[:,1] = locs1[:,0]*ar_frame.shape[0]/cv_cover.shape[0]
#     xy2 = np.zeros(locs2.shape)
#     xy2[:,0] = locs2[:,1]
#     xy2[:,1] = locs2[:,0]
#     H, inliers = planarH.computeH_ransac(xy1[matches[:,0]],xy2[matches[:,1]], opts)
#     # Apply projection
#     composite = planarH.compositeH(H, ar_frame, book_frame)
#     plt.imshow(composite)
#     plt.show()

# # Save frames to AVI
# out = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc(*'mp4v'), 1, (book_vid[0].shape[1], book_vid[0].shape[0]))
# for frame in frames_of_interest:
#     book_frame = book_vid[frame]
#     out.write(book_frame)
# out.release()

# AR to AVI
print("Start time: {}".format(datetime.now()))
out = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc(*'mp4v'), 25, (book_vid[0].shape[1], book_vid[0].shape[0]))
total_frames = min(ar_vid.shape[0],book_vid.shape[0])
for f_indx in range(0,total_frames):
#for f_indx in frames_of_interest:
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



