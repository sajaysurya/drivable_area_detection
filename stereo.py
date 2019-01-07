#!/usr/bin/env python
'''
python script to calculate dense stereo depth map.
returns a filtered dense map.

SYNTAX: ./stereo left.image right.image
the output opens in a separate window that can be saved later.
NOTE: please make sure that the executable flag for ease of use.

can also be used as a module and get_depth_map function can be directly called.

DISCLAIMER: has quite a few magic numbers in the code,
might not be direclty portable to vastly different videos.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_depth_map(l_image, r_image):
    '''
    uses the given input images to create depth map
    l_image: left image
    r_image: right image

    outputs the following:
    depth_map: a dense stereo depth map
    '''
    # create mappers
    max_disparity = 64
    window_size = 3
    l_mapper = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        uniquenessRatio=25,
    )
    r_mapper = cv2.ximgproc.createRightMatcher(l_mapper)
    # create maps (for left and right camera)
    l_map = l_mapper.compute(l_image, r_image)
    r_map = r_mapper.compute(r_image, l_image)
    # filter the image to get a nice dense map
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(l_mapper)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(2.0)
    depth_map = wls_filter.filter(l_map, l_image, None, r_map)
    # add 16 to get disparity in the range 0 to 64 pixels
    # get pixel disparity by dividing depth_map by 16
    depth_map = ((depth_map + 16)/16.0).astype(int)
    return depth_map

def main(l_image_path, r_image_path):
    '''
    uses the images in the given path and outputs the
    calculated dense stereo map in a separate window
    '''
    l_image = cv2.imread(l_image_path)
    r_image = cv2.imread(r_image_path)
    depth_map = get_depth_map(l_image, r_image)
    plt.imshow(depth_map, vmin=0, vmax=64)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
