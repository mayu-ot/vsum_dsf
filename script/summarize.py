import sys
import os
import json
import numpy as np
from scipy.io import savemat


def get_flabel(frames, fnum, fps, seg_l):
    s_i = [int(seg_fn[0][:-4]) for seg_fn in frames]
    e_i = [s + fps * seg_l for s in s_i]
    e_i = map(round, e_i)
    e_i = map(int, e_i)

    label = np.zeros((fnum, 1))
    for s, e in zip(s_i, e_i):
        label[s:e] = 1
    return label

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from func.sampling.vsum import VSUM
    from func.nets import vid_enc
    import chainer
    from chainer import serializers
    from chainer import configuration
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', '-d', type=str, default='summe',
                        help='dataset name')
    parser.add_argument('--feat_type', '-f', type=str, default='smt_feat',
                        help='feat_type: smt_feat or vgg')
    args = parser.parse_args()

    # settings
    seg_l = 5
    feat_type = args.feat_type

    d_name = args.dname
    dataset_root = 'data/{}/'.format(d_name)
    out_dir = 'results/{:}/{:}/'.format(d_name, feat_type)
    print('save to: ', out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset = json.load(open(dataset_root + 'dataset.json'))
    video_id = [d['videoID'] for d in dataset]

    # load embedding model

    ##########################################################################
    if feat_type == 'smt_feat':
        model = vid_enc.Model()
        serializers.load_npz('data/trained_model/model_par', model)
    elif feat_type == 'vgg':
        from func.nets.vid_enc_vgg19 import Model
        model = Model()
    else:
        raise RuntimeError('[invalid feat_type] use smt_feat or vgg')
    ##########################################################################


    for v_id in video_id:

        with configuration.using_config('train', False):
            with chainer.no_backprop_mode():
                vsum = VSUM(v_id, model, dataset=d_name, seg_l=seg_l)

        _, frames, _ = vsum.summarizeRep(seg_l=seg_l, weights=[1.0, 0.0])

        # get 0/1 label for each frame
        fps = vsum.dataset.data['fps']
        fnum = vsum.dataset.data['fnum']
        label = get_flabel(frames, fnum, fps, seg_l)

        np.save(out_dir + v_id, label)
