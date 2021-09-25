import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points

    # Constructing A:
    # For points (xi,yi) and (ui,vi)
    # [xi, yi, i, 0, 0, 0, -xi*ui, -yi*ui, -ui] = 0
    # [0, 0, 0, xi, yi, i, -xi*vi, -yi*vi, -vi] = 0
    # Repeat for all i
    # x2 constains (x,y) and x1 contains (u,v)
    A = np.zeros((x1.shape[0]*2,9))
    for i in range(0, x1.shape[0]):
        A[2*i,:] = [x2[i,0], x2[i,1], 1, 0, 0, 0, -x2[i,0]*x1[i,0], -x2[i,1]*x1[i,0], -x1[i,0]]
        A[2*i+1,:] = [0, 0, 0, x2[i,0], x2[i,1], 1, -x2[i,0]*x1[i,1], -x2[i,1]*x1[i,1], -x1[i,1]]
    # h is last column of v, or last row of vT
    u, s, vT = np.linalg.svd(A)
    h = vT[-1]
    H2to1 = np.reshape(h,[3,3])

    return H2to1

def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    tx1 = np.average(x1[:,0])
    ty1 = np.average(x1[:,1])
    tx2 = np.average(x2[:,0])
    ty2 = np.average(x2[:,1])

    #Shift the origin of the points to the centroid
    x1_org = np.zeros((x1.shape[0],2))
    x1_org[:,0] = x1[:,0]-tx1
    x1_org[:,1] = x1[:,1]-ty1
    x2_org = np.zeros((x2.shape[0],2))
    x2_org[:,0] = x2[:,0]-tx2
    x2_org[:,1] = x2[:,1]-ty2

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    k1 = 2**(1/2)/np.max(np.linalg.norm(x1_org,axis=1))
    k2 = 2**(1/2)/np.max(np.linalg.norm(x2_org,axis=1))

    #Similarity transform 1
    T1 = np.zeros((3,3))
    T1[0,0] = k1
    T1[1,1] = k1
    T1[2,2] = 1
    T1[0,2] = -tx1*k1
    T1[1,2] = -ty1*k1
    x1_homo = np.ones((x1.shape[0],3))
    x1_homo[:,0:2] = x1
    x1_norm = np.zeros((x1.shape[0],2))
    for i in range(0, x1.shape[0]):
        x1_norm[i,:]=(T1@x1_homo[i,:])[0:2]

    #Similarity transform 2
    T2 = np.zeros((3,3))
    T2[0,0] = k2
    T2[1,1] = k2
    T2[2,2] = 1
    T2[0,2] = -tx2*k2
    T2[1,2] = -ty2*k2
    x2_homo = np.ones((x2.shape[0],3))
    x2_homo[:,0:2] = x2
    x2_norm = np.zeros((x2.shape[0],2))
    for i in range(0, x2.shape[0]):
        x2_norm[i,:]=(T1@x2_homo[i,:])[0:2]

    #Compute homography
    H_norm = computeH(x1_norm, x2_norm)

    #Denormalization
    H2to1 = np.linalg.inv(T1) @ H_norm @ T2
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    


    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    #Create mask of same size as template

    #Warp mask by appropriate homography

    #Warp template by appropriate homography

    #Use mask to combine the warped template and the image
    
    return composite_img



v = np.zeros((4,2))
v[0,:] = [0,0]
v[1,:] = [0,1]
v[2,:] = [1,0]
v[3,:] = [1,1]
print("Hnorm:\n" + str(computeH_norm(v,v)))
print("H:\n" + str(computeH(v,v)))