import chainer.links as L
import chainer.functions as F
from chainer import cuda, serializers, Chain, Variable
import chainer
import torch.nn as nn
from os.path import dirname
import json
import numpy as np
from numpy.random import RandomState

import sys


class Model(Chain):

    ########################################################################

    def __init__(self, b_size={'video': 5}):
        self.b_size = b_size

        super(Model, self).__init__(
            conv2=L.ConvolutionND(3, 1, 20, 5),
            conv3=L.ConvolutionND(3, 20, 20, 5),
            fc1=L.Linear(4096, 1300),
            fc4=L.Linear(1300,1300),
            fc5=L.Linear(1300, 300),
        )

    def __call__(self, x_seg):
        '''
        input: np.array(5xN, 4096)
        '''
        b_size = self.b_size['video']
        with cuda.get_device(x_seg.data):
            y0 = F.tanh(self.fc1(x_seg))
            y1 = F.tanh(self.fc4(y0))
            y2 = F.tanh(self.fc5(y1))
            h = F.dropout(F.relu(self.fc1(y2)))
            h = self.fc4(h)
            h = F.reshape(h, (int(h.shape[0] / b_size), b_size, 300))
            return F.sum(h, axis=1) / b_size

    ##########################################################################