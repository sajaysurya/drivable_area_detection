#!/usr/bin/env python
'''
python script to find the freespace boundary.

SYNTAX: ./freespace left.image right.image
the output opens in a separate window that can be saved later.
in the output, the free space is highlighted in green color
NOTE: please make sure that the executable flag for ease of use.

can also be used as a module and get_free_bound function can be directly called.

DISCLAIMER: has quite a few magic numbers in the code,
might not be direclty portable to vastly different videos.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import stereo
import utilities
from hmmlearn import base  # this used to be a part of sklearn
from scipy.sparse import diags

class HMM(base._BaseHMM):
    '''
    a slightly modified HMM to apply Viterbi Algorithm for this case
    '''
    def _compute_log_likelihood(self, disp):
        '''
        the u_disparity is taken as input to calculate emission probability
        '''
        # quantify the height of obstacles and threshold
        loglike = np.log((disp > 25).astype(np.float)+1e-32)  # adjust to prevent -int
        # adjust to prefer the closest obstacle
        loglike += np.tile(-np.flip(np.arange(disp.shape[1])), (disp.shape[0], 1))*0.1
        return loglike

def get_disparity(depth_map):
    '''
    uses the depth_map to find u and v disparity
    '''
    # using disparity as a measure of depth, generating one-hot world-view
    world_view = utilities.onehot_initialization(depth_map)
    # project the world views to get disparity maps
    u_disparity = np.sum(world_view, axis=0).T
    v_disparity = np.sum(world_view, axis=1)
    return v_disparity, u_disparity

def calculate_free_bound(u_disparity):
    '''
    calculates the free bound using the given u_disparity
    '''
    # defining HMM model
    num_states = u_disparity.shape[0]
    model = HMM(num_states)
    model.startprob_ = np.ones(num_states)/num_states # uniformly distributed
    # transition is a band matrix so that nearby jumps are more probable
    band = [1, 2, 3, 5, 7, 9, 11, 15, 11, 9, 7, 5, 3, 2, 1]
    posi = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    mat = diags(band, posi, shape=(num_states, num_states)).toarray()+0.5
    # normalize the matrix to get transition distribution
    mat = mat/np.sum(mat, axis=0)
    # transpose to get the right direction of normalization
    mat = mat.T
    model.transmat_ = mat
    # decode to get bound
    bound = model.decode(u_disparity.T)  # transposed to suit requirements
    return bound[1]  # [1] has the most probable states

def get_free_bound(l_image, r_image):
    '''
    uses the given input images to find bound
    l_image : left image
    r_image : right image

    outputs the following
    v_disparity : v_disparity map as a matrix
    u_disparity : u_disparity map as a matrix
    depth_map : a dense stereo map
    free_bound : a free-space bound in terms of pixel disparity values
    '''
    # get the stereo image
    depth_map = stereo.get_depth_map(l_image, r_image)
    # get the v and u_disparity
    v_disparity, u_disparity = get_disparity(depth_map)
    # threshold to remove some noise
    v_disparity_thresh = v_disparity > 100
    # get the road plane from v_disparity
    lines = np.squeeze(cv2.HoughLines(v_disparity_thresh.astype(np.uint8), 1, np.pi/180, 75))
    # extracting road planes in a certain range
    line = lines[(lines[:, 1] > 1.5) * (lines[:, 1] < 3.0), :]
    # if no line matches the criteria, just give out the first found line
    if line.size == 0:
        line = lines[0]
    else:  # else give the first line that matches the criteria
        line = line[0]
    # extract the radius and angle defining the line
    rho = line[0]
    theta = line[1]
    # create a function to map disparity values to image row indices
    project = lambda x: -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
    # get the free bound from u_disparity
    free_bound = calculate_free_bound(u_disparity)
    # return all useful stuff
    return v_disparity, u_disparity, depth_map, free_bound, project

def main(l_image_path, r_image_path):
    '''
    uses the images in the given path and outputs the
    calculated free space in a separate window
    '''
    l_image = cv2.imread(l_image_path)
    r_image = cv2.imread(r_image_path)
    v_disparity, u_disparity, depth_map, free_bound, project = get_free_bound(l_image, r_image)
    plt.imshow(cv2.cvtColor(l_image, cv2.COLOR_BGR2RGB))
    plt.fill_between(np.arange(l_image.shape[1]),
                     project(free_bound),
                     l_image.shape[0],
                     facecolor='green',
                     alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
