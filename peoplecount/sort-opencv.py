"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

# from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import cv2
from ulits.lineUtils import LineUlit


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                             0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [
                             0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=12, min_hits=6):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def get_people_img(img, det, folderanme):
    det = det.astype(np.int32)
    # print(det[1], (det[3] - det[1]), det[0], (det[2] - det[0]))
    # crop_img = img[det[1]:(det[3] - det[1]), det[0]:(det[2] - det[0])]
    crop_img = img[det[1]:det[3],det[0]:det[2]]
    
    # print(crop_img.shape,len(img),len(img[0]))
    
    if crop_img is None or len(crop_img) == 0 or len(crop_img[0]) == 0:
        return

    # cv2.imshow('crop', crop_img)
    # if cv2.waitKey(33) & 0xFF == ord('q'):
    #     return

    folder = 'output/%s'%folderanme
    # print(os.path.exists(folder) , os.path.isdir(folder))
    if os.path.exists(folder) == False and os.path.isdir(folder) == False:
        os.mkdir(folder)
        print(folder)
    cv2.imwrite('output/%s/%s.jpg'%(folderanme,det[4]),crop_img)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display',
                        help='Display online tracker output (slow) [False]', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    # sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof','ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17','ADL-Rundle-6','ADL-Rundle-8','Venice-2']
    sequences = ['test_video2']
    args = parse_args()
    display = args.display
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display

    IN = 0
    OUT = 0
    T = {}

    # if(display):
    #     # if not os.path.exists('mot_benchmark'):
    #     #   print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
    #     #   exit()
    #     plt.ion()
    #     fig = plt.figure()

    if not os.path.exists('output'):
        os.makedirs('output')

    line = [[0, 600], [750, 0]]
    lineUtils = LineUlit([0, 0], [0, 0], isReverseY=False)

    for seq in sequences:
        mot_tracker = Sort()  # create instance of the SORT tracker
        seq_dets = np.loadtxt('data/%s/det.txt' %
                              (seq), delimiter=',')  # load detections
        cap = cv2.VideoCapture('data/%s/video.mp4' % (seq))
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        imageId = 0
        frame = 0
        with open('output/%s.txt' % (seq), 'w') as out_file:
            print("Processing %s." % (seq))
            while cap.isOpened():
                # frame in range(int(seq_dets[:,0].max())):
                ret, videoFrame = cap.read()
                if not ret:
                    break
                # imageId+=1
                height = videoFrame.shape[0]
                width = videoFrame.shape[1]
                line[0][1] = int(height * 0.3)
                line[1][0] = width
                line[1][1] = int(height * 0.3)
                if lineUtils.p1[0] == 0:
                    lineUtils.p1 = line[0]
                    lineUtils.p2 = line[1]
                    lineUtils.setTotalY(height)

                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                # if(display):
                # ax1 = fig.add_subplot(111, aspect='equal')
                # fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
                # im =io.imread(fn)
                # ax1.imshow(videoFrame)
                # plt.title(seq + ' Tracked Targets %s' % (frame))

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                if(display):
                    cv2.line(videoFrame, (lineUtils.p1[0], lineUtils.p1[1]),
                             (lineUtils.p2[0], lineUtils.p2[1]), (255, 0, 0), 2)

                    cv2.putText(videoFrame, 'IN: %d' % IN, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    cv2.putText(videoFrame, 'OUT: %d' % OUT, (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame,
                                                                    d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)

                    center = [int((d[2] + d[0]) / 2),
                              int((d[3] + d[1]) / 2)]
                    dkey = str(d[4])
                    lastvalue = 0
                    if T.keys().__contains__(dkey):
                        lastvalue = T[dkey]
                    tempvalue = lineUtils.getPosition(center)
                    if tempvalue != 0:
                        T[dkey] = tempvalue
                    if lastvalue != T[dkey] and tempvalue != 0:
                        if lastvalue == -1:
                            # IN -= 1
                            OUT += 1
                        elif lastvalue == 1:
                            IN += 1
                            get_people_img(videoFrame, d, seq)
                            # OUT -= 1

                    if(display):
                        d = d.astype(np.int32)
                        # ax1.add_patch(patches.Rectangle(
                        #     (d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
                        # ax1.set_adjustable('box-forced')
                        # plt.text(d[0], d[1], str(d[4]))
                        cc = (colours[d[4] % 32, :] * 255)
                        color = (int(cc[0]), int(cc[1]), int(cc[2]))

                        cv2.rectangle(
                            videoFrame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color, 2, 0)
                        cv2.circle(
                            videoFrame, (center[0], center[1]), 3, color, -1)

                        cv2.putText(videoFrame, str(d[4]), (int(d[0]), int(
                            d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                    # print('%d at %d' %
                    #         (d[4], tempvalue))

                if(display):
                    #     # plt.plot([line[0][0] + 20, line[0][1]],
                    #     #          [line[1][0], line[1][1] - 20])
                    #     # fig.canvas.flush_events()
                    #     # plt.draw()
                    #     # ax1.cla()
                    cv2.imshow('frame', videoFrame)
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

        print('total out: %d\ntotal in: %d' % (OUT, IN))

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))
    if(display):
        print("Note: to get real runtime results run without the option: --display")
