import chainer.links as L
import chainer.functions as F
from chainer import cuda, serializers, Chain, Variable
import chainer
from os.path import dirname
import json
import numpy as np
from numpy.random import RandomState

import sys


class Model(Chain):

    def __init__(self, b_size = {'video':5}):
        self.b_size = b_size
        print(b_size['video'])
        super(Model, self).__init__(
            l1 = L.ConvolutionND(2, 1, 20, 5),    #L.Convolution(ndim, in_channel, out_chennel, stride)
            l2 = L.ConvolutionND(2, 20, 20, 5),
            l3 = L.Linear(2500, 1000),
            l4 = L.Linear(1000, 1000),
            l5 = L.Linear(1000, 300),
        )

    def __call__(self, x_seg):
        '''
        input: np.array(5xN, 4096)
        '''
        b_size = self.b_size['video']
        with cuda.get_device(x_seg.data):
            print(x_seg.shape)
            y0 = F.max_pooling_nd(F.local_response_normalization(F.relu(self.l1(x_seg))), 3, stride=2)
            y1 = F.max_pooling_nd(F.local_response_normalization(F.relu(self.l2(y0))), 3, stride=2)
            y2 = F.tanh(self.l3(y1))
            y3 = F.dropout(F.relu(self.l4(y2)))
            y4 = F.tanh(self.l5(y3))
            h = F.reshape(y4, (int(y4.shape[0] / b_size), b_size, 300))
            return F.sum(h, axis=1) / b_size