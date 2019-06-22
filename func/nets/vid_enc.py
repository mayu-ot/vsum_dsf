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

        super(Model, self).__init__(
            fc_v1=L.Linear(4096, 1000),
            fc_v2=L.Linear(1000, 1000),
            fc_v3=L.Linear(1000, 300),
        )

    def __call__(self, x_seg):
        '''
        input: np.array(5xN, 4096)
        '''
        b_size = self.b_size['video']
        with cuda.get_device(x_seg.data):
            y0 = F.tanh(self.fc_v1(x_seg))
            y1 = F.tanh(self.fc_v2(y0))
            y2 = F.dropout(F.relu(self.fc_v3(y1)))
            h = F.reshape(y2, (int(y2.shape[0] / b_size), b_size, 300))
            return F.sum(h, axis=1) / b_size