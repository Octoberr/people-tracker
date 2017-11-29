# coding:utf-8

import cv2
import numpy as np
import os

filedir = os.path.dirname(os.path.realpath(__file__))


# 将视频保存在当前目录的文件名下
def savepicture(videoname):
    videopath = os.path.join(filedir, videoname)
    camera = cv2.VideoCapture(videopath)
    name = videoname.split('.')[0]
    # 创建名字开头的文件夹
    savename = os.path.join(filedir, name)
    if not os.path.isdir(savename):
        os.makedirs(savename)
    count = 1
    while True:
        (grabbed, image) = camera.read()
        if not grabbed:
            break
        newimage = image
        if newimage is None:
            continue
        else:
            filename = os.path.join(savename, str(count)+'.png')
            print(filename)
            cv2.imwrite(filename, newimage)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()
    # 返回将视频保存为图片的文件目录
    return savename