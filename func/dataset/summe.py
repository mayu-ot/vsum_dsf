import json
import numpy as np

datasetRoot = 'data/summe/'


class SUMME():

    def __init__(self, video_id, feat_type='vgg'):

        dataset = json.load(open(datasetRoot + 'dataset.json'))
        print 'load ' + video_id
        data = filter(lambda x: x['videoID'] == video_id, dataset)
        self.data = data[0]
        self.feat = np.load(datasetRoot + 'feat/' + feat_type +
                            '/' + video_id + '.npy').astype(np.float32)

    def sampleFrame(self):
        fps = self.data['fps']
        fnum = self.data['fnum']

        idx = np.arange(fps, fnum, fps)
        idx = np.floor(idx)
        idx = idx.tolist()
        idx = map(int, idx)

        img = [self.data['image'][i] for i in idx]
        img_id = [self.data['imgID'][i] for i in idx]
        score = [self.data['score'][i] for i in idx]

        return img, img_id, score
