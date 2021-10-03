import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
#from helper import plotMatches

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
original = cv2.imread('../data/cv_cover.jpg')
results = np.zeros([36])
lbls = []
for i in range(36):
    print('Progress: {}/{}'.format(i, 35))
    #Rotate Image
    rotated = rotate(original,i*10)
    #Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(original, rotated, opts)
    print('Matches: {}'.format(matches.shape[0]))
    #Update histogram
    results[i] = matches.shape[0]
    lbls.append(10*i)


#Display histogram
results = results.astype('int')
print("Results:\n{}".format(results))
plt.bar(lbls,results,width=9.8)
plt.xlabel('Rotation (deg)')
plt.ylabel('Matches')
plt.title('Matches vs Rotation')
plt.show()

