from scipy.io import loadmat
import numpy as np
import json
from sklearn.metrics import f1_score


header = '''|                         | Min. HUMAN | Avg. HUMAN | Max. HUMAN |  Summarization  |
|------------------------:|:----------:|:----------:|:----------:|:---------------:|'''

entry = '|{:>25}|{:^12.3}|{:^12.3}|{:^12.3}|{:^17.3}|'

res_sum_mean = '|                     mF1 |{:^12.3}|{:^12.3}|{:^12.3}|{:^17.3}|'
res_sum_rel_to_avr = '|   relative to avg.human |{:^12.4}|{:^12.4}|{:^12.4}|{:^17.4}|'
res_sum_rel_to_max = '|   relative to best.human|{:^12.4}|{:^12.4}|{:^12.4}|{:^17.4}|'


def eval_f1(pred_summary, gt_sumamry):
    pred_summary = (pred_summary > 0).astype(np.int)
    gt_sumamry = (gt_sumamry > 0).astype(np.int)
    f1 = map(lambda y_true: f1_score(y_true, pred_summary), gt_sumamry)
    return sum(f1) / len(f1)


def eval_summary_(dataset_name, res_base, gt_base):
    res_base = res_base

    data = json.load(open('data/{}/dataset.json'.format(dataset_name)))

    res = {}
    for d in data:
        v_id = d['videoID']
        pred_summary = np.load(res_base + '%s.npy' % v_id)

        gt_data = loadmat(gt_base + '/%s.mat' % v_id)
        user_score = gt_data.get('user_score')
        user_score = user_score.T

        f1 = eval_f1(pred_summary, user_score)
        res[v_id] = f1

    return res


def eval_human_summary(dataset_name, gt_base):
    res_base = gt_base
    data = json.load(open('data/{}/dataset.json'.format(dataset_name)))

    res_min, res_avr, res_max = {}, {}, {}
    for d in data:
        v_id = d['videoID']

        gt_data = loadmat(gt_base + '/%s.mat' % v_id)
        user_score = gt_data.get('user_score')
        user_score = user_score.T

        fscore_all = []
        for u_id in range(len(user_score)):
            pred_summary = user_score[u_id]
            gt_sumamry = np.delete(user_score, u_id, 0)
            f1 = eval_f1(pred_summary, gt_sumamry)
            fscore_all.append(f1)

        res_min[v_id] = min(fscore_all)
        res_avr[v_id] = sum(fscore_all) / len(fscore_all)
        res_max[v_id] = max(fscore_all)

    return res_min, res_avr, res_max

if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('resbase', type=str, default='',
                        help='path to resulting .npy files')
    parser.add_argument('--dname', '-d', type=str, default='summe',
                        help='dataset name (summe)')
    args = parser.parse_args()

    dataset_name = args.dname
    if dataset_name == 'summe':
        gt_base = 'data/summe/GT'
    else:
        raise NotImplementedError

    res_base = args.resbase if args.resbase[
        -1] == os.sep else args.resbase + os.sep
    res = eval_summary_(dataset_name, res_base, gt_base)
    hum_min, hum_avr, hum_max = eval_human_summary(dataset_name, gt_base)

    print header
    for k in res:
        print entry.format(k, hum_min[k], hum_avr[k], hum_max[k], res[k])

    score_mean = [sum(r.values()) / len(r)
                  for r in [hum_min, hum_avr, hum_max, res]]

    print '|' + '-' * 82 + '|'
    print res_sum_mean.format(*score_mean)
    print res_sum_rel_to_avr.format(*map(lambda x: x / score_mean[1], score_mean))
    print res_sum_rel_to_max.format(*map(lambda x: x / score_mean[2], score_mean))
