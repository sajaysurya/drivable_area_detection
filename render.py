#!/usr/bin/env python
'''
python script to find the freespace boundary in all video frames
and to compile that as a video.

SYNTAX: ./render [f_rate]
where f_rate is the frame rate for ffmpeg (optional argument)
the frames are stores in the folder named output
output.mkv is the final video
in the output, the free space is highlighted in green color
NOTE: please make sure that the executable flag for ease of use.
'''
import sys
from pathlib import Path
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import freespace
from tqdm import tqdm

def main(f_rate):
    '''
    c-style main function
    '''
    # if output folder is present, remove it
    if Path('./output').exists():
        subprocess.run(['rm', '-rf', 'output'])
        subprocess.run(['rm', '-rf', 'output.mkv'])
    # create output folder
    subprocess.run(['mkdir', 'output'])
    # get left and right image lists
    l_list = sorted(list(Path('./video/2011_09_26/2011_09_26_drive_0018_sync/image_02/data').glob('*')))
    r_list = sorted(list(Path('./video/2011_09_26/2011_09_26_drive_0018_sync/image_03/data').glob('*')))
    # for each image (with status bar)
    for count in tqdm(range(len(l_list))):
        # clear matplotlib figure
        plt.clf()
        #load image
        l_image = cv2.imread(str(l_list[count]))
        r_image = cv2.imread(str(r_list[count]))
        # get free space related info
        (v_disparity,
         u_disparity,
         depth_map,
         free_bound,
         project) = freespace.get_free_bound(l_image, r_image)
        # make subplots
        gs = gridspec.GridSpec(4, 3)
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, :-1])
        ax3 = plt.subplot(gs[1, -1])
        ax4 = plt.subplot(gs[2:, :])
        # plot stereo image
        ax1.imshow(depth_map[:, 64:])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title('Depth Map')
        # plot v-disparity
        ax2.imshow(u_disparity[:, 64:], interpolation='nearest', aspect='auto')
        ax2.set_ylim([64, 0])
        ax2.set_title('U-Disparity')
        # plot u-disparity
        ax3.imshow(v_disparity, interpolation='nearest', aspect='auto')
        ax3.set_xlim([0, 64])
        ax3.set_title('V-Disparity')
        # plot freespace
        ax4.imshow(cv2.cvtColor(l_image[:, 64:], cv2.COLOR_BGR2RGB))
        ax4.fill_between(np.arange(l_image.shape[1]-64),
                         project(free_bound)[64:],
                         l_image.shape[0],
                         facecolor='green',
                         alpha=0.5)
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax4.set_title('Free Space')
        # save file in output folder
        plt.tight_layout()
        plt.savefig('output/'+'{:04d}'.format(count)+'.png')
    # use ffmpeg to combine frames into video
    print('Making the video output.mkv.. this might take a minute...')
    subprocess.run(['ffmpeg',
                    '-r',
                    str(f_rate),
                    '-start_number', '1',
                    '-i', 'output/%04d.png',
                    '-crf', '0',
                    '-vcodec', 'libx264',
                    'output.mkv'])
    print("All done, checkout the video 'output.mkv' in the pwd")


if __name__ == '__main__':
    # set default frame rate if the argument is absent
    if len(sys.argv) < 2:
        F_RATE = 8
    else:
        F_RATE = sys.argv[1]
    main(F_RATE)
