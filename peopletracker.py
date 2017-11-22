# coding:utf-8

import numpy as np
import cv2
import imutils
from sort import *
import os
filedir = os.path.dirname(os.path.realpath(__file__))

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects):
    # 左下和右上
    for x, y, a, b, z in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        cv2.rectangle(img, (int(x), int(y)), (int(a), int(b)), (0, 255, 0), thickness=1)
        cv2.putText(img, str(int(z)), (int(x), int(b)-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                    color=(152, 255, 204), thickness=2)
        return img


def convertlocation(people):
    x = people.shape
    location = np.zeros((x[0], x[1]+1), dtype=people.dtype)
    for i in range(len(people)):
        pad_w, pad_h = int(0.15*people[i][2]), int(0.05*people[i][3])
        location[i][0] = people[i][0]+pad_w
        location[i][1] = people[i][1]+pad_h
        location[i][2] = people[i][0]+people[i][2]-pad_w
        location[i][3] = people[i][1]+people[i][3]-pad_h
        location[i][4] = 1
    return location


def detecpeople(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # 16,16适用于比较小的图片
    found, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)
    # 32,32适用于width=800的图片
    # found, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)
    return found


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
    mot_tracker = Sort()
    while True:
        (grabbed, image) = camera.read()
        # img numpy array, 裁剪后的img
        if not grabbed:
            break
        newimage = imutils.resize(image, width=min(800, image.shape[1]))
        detection = detecpeople(newimage)
        people = detection  # people location ,numpy, 根据裁剪后的img找到的人
        # 如果有人就进行追踪
        if len(people) > 0:
            # convert people location
            detections = convertlocation(people)

            # update SORT
            track_bbs_ids = mot_tracker.update(detections)
            # 标识和追踪名字
            frame = draw_detections(newimage, track_bbs_ids)
        else:
            frame = newimage
        # print frame, type(frame)
        if frame is None:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("Security Feed", frame)
    camera.release()
    cv2.destroyAllWindows()
    # imgfile = os.path.join(filedir, 'img1')
    # for path, subdirs, files in os.walk(imgfile):
    #     for filename in files:
    #         # imgfile name
    #         f = os.path.join(path, filename)
    #         detection = detecpeople(f)
    #         img = detection[0]  # img numpy array
    #         people = detection[1]  # people location ,numpy
    #         # convert people location
    #         detections = convertlocation(people)
    #
    #         # update SORT
    #         track_bbs_ids = mot_tracker.update(detections)
    #         # 标识和追踪名字
    #         frame = draw_detections(img, track_bbs_ids)
    #         cv2.imshow("Security Feed", frame)
    #         ch = cv2.waitKey(0)
    # cv2.destroyAllWindows()



