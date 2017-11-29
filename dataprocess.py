# coding:utf-8

import numpy as np
dettextfile = 'dets.txt'
seq_dets = np.loadtxt(dettextfile, delimiter=',')
print(seq_dets)
