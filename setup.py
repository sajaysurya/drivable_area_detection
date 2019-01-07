#!/usr/bin/env python
'''
python script to download the the necessary dataset.
depends on wget and unzip and calls them as subprocesses.
NOTE: please make sure that the executable flag is setup for this python file.
'''
import subprocess

def main():
    '''
    main script
    '''
    name = 'video'
    # construct command for downloading
    program = ['wget']
    flags = ['-c',  # to continue interrupted downloads
             '-O', name+'.zip']  # custom name for video
    target = ['https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'+
              '2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip']
    command = program + flags + target
    # donwload video
    print("Downloading video...")
    subprocess.run(command)

    # construct command for unzipping
    subprocess.run(['mkdir', name])  # create requisite directory
    program = ['unzip']
    flags = ['-d', name]  # extract to custom directory
    target = [name+'.zip']
    command = program + flags + target
    # extract video
    print("Extrating video...")
    subprocess.run(command)
    print("Done")

if __name__ == '__main__':
    main()
