#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': (
    'voc_2007_trainval+voc_2012_trainval',)}


def print_dets(dets, imageId, output, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print('score is too low')
        return

    inds = list(set(inds))

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score < thresh:
            continue

        arr_res = [imageId, -1, bbox[0], bbox[1],
                   bbox[2] - bbox[0], bbox[3] - bbox[1], score]
        # print(arr_res)
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f' % (
            arr_res[0], arr_res[1], arr_res[2], arr_res[3], arr_res[4], arr_res[5], arr_res[6]), file=output)


def deal_dets(sess, net, im, imageId, output):

    print(imageId)
    scores, boxes = im_detect(sess, net, im)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        if cls == 'person':
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            print_dets(dets, imageId, output, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                           NETS[demonet][0])
    timer = Timer()
    timer.tic()

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 21,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    video_name = 'video'

    video_file = os.path.join('data', 'video', video_name + '.mp4')
    output_file = os.path.join('output', 'sort', video_name + '.txt')

    if os.path.isfile(video_file) == False:
        print('can not read file %s', video_file)

    cap = cv2.VideoCapture(video_file)

    imageId = 0
    with open(output_file, 'w') as f:
        print('start cap video')
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            imageId += 1
            deal_dets(sess, net, frame, imageId, f)

    cap.release()
    cv2.destroyAllWindows()
    timer.toc()
    print('Detection took {:.3f}s for object proposals'.format(timer.total_time))
