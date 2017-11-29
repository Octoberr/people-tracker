# coding:utf-8


from sort import *


import os
filedir = os.path.realpath(os.path.dirname(__file__))



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--det', type=str, help='The video path.', default='newdet.txt')
    parser.add_argument('--imgfile', type=str, help='The img file path .', default='sorttest')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dettextfile = args.det
    imgfilename = args.imgfile
    detpath = os.path.join(filedir, dettextfile)
    imgpath = os.path.join(filedir, imgfilename)
    colours = np.random.rand(32, 3) #used only for display
    plt.ion()
    fig = plt.figure()
    mot_tracker = Sort()  # create instance of the SORT tracker
    seq_dets = np.loadtxt(detpath, delimiter=',')  # load detections
    for frame in range(int(seq_dets[:, 0].max())):
        # detection and frame numbers begin at 0 frame = 0
        frame+=1
        dets = seq_dets[seq_dets[:, 0] == frame, 1:6]
        ax1 = fig.add_subplot(111, aspect='equal')
        fn = os.path.join(imgpath, '%06d.jpg'%(frame))
        im = io.imread(fn)
        ax1.imshow(im)
        plt.title(' Tracked Targets')

        trackers = mot_tracker.update(dets)
        for d in trackers:
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
            ax1.set_adjustable('box-forced')
        fig.canvas.flush_events()
        plt.draw()
        ax1.cla()
        frame += 1
