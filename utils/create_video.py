import cv2
import argparse
import glob
import os
# import numpy as np

def create_video(path, videoName):

    img_array = []
    size = 0
    for filename in sorted(glob.glob(os.path.join(path+'/*.png'))):
        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)

    if size !=0:
        out = cv2.VideoWriter(f'{path+videoName}.avi',cv2.VideoWriter_fourcc(*'DIVX'),60, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    else:
        print('No video created')


if __name__=='__main__':
    parser = argparse.ArgumentParser('Create video from images')
    parser.add_argument('--path_pictures', '-p',default='./')
    parser.add_argument('--video_name', '-n',default='video')
    args = parser.parse_args()
    create_video(args.path_pictures, args.video_name)
