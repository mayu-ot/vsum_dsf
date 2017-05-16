import numpy as np


def uniform_sampling(data, out_dir, vid_id, total_l=0.15, seg_l=5):
    print vid_id
    v_id = data['videoID']
    fps = data['fps']
    length = data['length']  # length in sec.
    sum_l = length * total_l
    seg_n = np.floor(sum_l / float(seg_l))  # num of segments in a summary
    cent = np.linspace(0, length, seg_n + 2)[1:-1]
    start = cent - seg_l / 2.0

    label = np.zeros((data['fnum'], 1))
    for s in start:
        label[int(s * fps):int(np.floor(s * fps + seg_l * fps))] = 1

    return label


if __name__ == '__main__':
    import json
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', '-d', type=str, default='summe',
                        help='dataset name')
    parser.add_argument('--length', '-l', type=int, default=5,
                        help='segment length')
    args = parser.parse_args()
    d_name = args.dname

    out_dir = 'results/{}/uniform/'.format(d_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    datasetRoot = 'data/{}/'.format(d_name)
    dataset = json.load(open(datasetRoot + 'dataset.json'))

    for data in dataset:
        label = uniform_sampling(data, out_dir, data[
            'videoID'], seg_l=args.length)

        np.save(out_dir + data['videoID'], label)
