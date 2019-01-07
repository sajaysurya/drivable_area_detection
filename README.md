# Free Space Detection
Free (drivable) space identification for self-driving cars using stereo vision.

## Aim
To find the obstacle-free space around the self-driving vehicle using images recorded by a stereo camera. The images in the Kitti Dataset are used for demonstrating the functionality of the utility.

## Result
Check out the following video demonstrating the functionality of the utility. Click the picture to play video.

[![Demo](http://img.youtube.com/vi/jA74TP_T2J0/0.jpg)](https://youtu.be/jA74TP_T2J0)

## Technique used
 - The left and right images recorded by the camera are processed by the SGBM (Semi-Global Matching) algorithm and smoothened out using a WLS (weighted least square) filter, all of which are available in OpenCV.
 - The depth map thus obtained is used to create u-disparity and v-disparity maps.
 - Free-space segmentation is done in u-disparity map and a boundary is identified. The boundary (location in each image column)is modelled as a HMM with custom transition and emission probabilities and the Viterbi algorithm is used to identify the boundary.
 - V-disparity map is used to identify the road plane, which is then used to map the boundary from the u-disparity map to the actual image.
 - The algorithm separately processes each frame, which are finally clubbed together into a video using ffmpeg.

## Limitations
 - This utility is suitable only for flat roads.
 - Currently the latency is very high (around 1s for processing a frame) with an i7-7500U.
 - Has low accuracy with quite a few false positives - failed to identify (parts of) obstacles that are very close.
 - Noisy - spatial noise can be (further) reduced by adjusting the transition probabilities of the HMM model and the temporal noise can be reduced by incorporating a Kalman filter (not-implemented) over successive frames.

## Dependencies
### Python Packages
 - numpy
 - matplotlib
 - opencv
 - scipy
 - tqdm
 - scikit-learn
 - hmmlearn
### Misc
 - unzip
 - ffmpeg

## Steps for replicating the results
 - Checkout docstrings for detailed usage instructions
 - Make sure that the listed dependencies are installed.
 - Make sure that the executable flag is set for ```setup.py```. Run it to download and extract the necessary video frames from the KITTI raw (synced+rectified) dataset.
 - Run ```./stereo.py left.png right.png``` to check the stereo utility using the supplied sample images. It can also be used as a python module and the instructions for using the corresponding function can be found in the source code.
 - Run ```./freespace.py left.png right.png``` to check the free-space mapping utility using the supplied sample images. It can also be used as a python module and the instructions for using the corresponding function can be found in the source code.
 - Run ```./render.py``` to create the output video ```output.mkv``` with all the free-space estimated in all frames of video downloaded from the Kitti Dataset repository.
 - This ```output.mkv``` demonstrates the free-space estimating utility
