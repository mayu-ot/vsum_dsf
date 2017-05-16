import chainer.functions as F
from chainer import Chain


class Model(Chain):

    def __init__(self):
        self.b_size = {'video': 5}

    def __call__(self, x_seg):
        '''
        input: np.array(5xN, 4096)
        '''
        b_size = self.b_size['video']
        out_size = x_seg.shape[1]
        return F.sum(F.reshape(x_seg, (x_seg.shape[0] / b_size, b_size, out_size)), axis=1) / b_size
