# coding:utf-8


from sort import *


import os
import cv2
filedir = os.path.realpath(os.path.dirname(__file__))


def draw_detections(img, rects):
    # 左下和右上
    for x, y, a, b, z in rects:
        # when not tracker,x,y,a,b为位置参数，z为预测的概率
        # when use tracker,x,y,a,b不改变， z为追踪的人的标识
        cv2.rectangle(img, (int(x), int(y)), (int(a), int(b)), (0, 0, 255), thickness=3)
        cv2.putText(img, str(z), (int(x), int(b)-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                    color=(0, 0, 255), thickness=2)
    return img


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--det', type=str, help='The video path.', default='onpersontest.txt')
    parser.add_argument('--imgfile', type=str, help='The img file path .', default='onepersontest')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dettextfile = args.det
    imgfilename = args.imgfile
    detpath = os.path.join(filedir, dettextfile)
    imgpath = os.path.join(filedir, imgfilename)
    mot_tracker = Sort()
    seq_dets = np.loadtxt(detpath, delimiter=',')  # load detections
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 自己录的
    # video = cv2.VideoWriter('output.avi', fourcc, 15.0, (640, 360))
    # 例子
    video = cv2.VideoWriter('notsortone.avi', fourcc, 15.0, (1920, 1080))
    for frame in range(int(seq_dets[:, 0].max())):
        # detection and frame numbers begin at 0 frame = 0
        frame += 1  # sort自带是从1开始
        dets = seq_dets[seq_dets[:, 0] == frame, 1:6]
        # imgfile
        fn = os.path.join(imgpath, str(frame)+'.png')
        # fn = os.path.join(imgpath, '%06d.jpg'%(frame))
        # 上传追踪坐标
        # trackers = mot_tracker.update(dets)
        img = cv2.imread(fn)
        if dets[0][0] == dets[0][1] == dets[0][2] == dets[0][3]:
            # 如果没有人就直接使用原图
            newframe = img
        else:
            newframe = draw_detections(img, dets)
        video.write(newframe)
        # keepdata = os.path.join(filedir, 'sorttxt.txt')
        # num = np.insert(trackers, 0, frame, axis=1)
        # with open(keepdata, "a") as f:
        #     np.savetxt(f, num, fmt='%.4f', delimiter=',')
        print 'write success', frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


