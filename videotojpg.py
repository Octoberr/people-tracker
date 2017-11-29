# coding:utf-8
import os
filedir = os.path.dirname(os.path.realpath(__file__))
import argparse
import cv2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='swmdemo')
    parser.add_argument('--video', type=str, help='The video path.',
                        default=os.path.join(filedir, '1511260824video_share.mp4'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    videopath = args.video
    camera = cv2.VideoCapture(videopath)
    while True:
        (grabbed, image) = camera.read()
        # img numpy array, 裁剪后的img
        if not grabbed:
            break
