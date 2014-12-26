# -*- coding: utf-8 -*-

import numpy as np

from data.data import Data
from classifier.kmeans import KMeans
from classifier.cnn import CNN
from score import online_score

CATEGORIES = (
    'artifacts',
    'chaetognaths',
    'copepods',
    'ctenophores',
    'detritus',
    'diatoms',
    'echinoderms',
    'fish',
    'gastropods',
    'gelatinous zooplankton',
    'hydromedusae',
    'invert larvae',
    'pelagic tunicates',
    'protists',
    'shrimp',
    'siphonophores',
    'trichodesmium',
)

NB_CLUSTERS = len(CATEGORIES)

if __name__ == '__main__':

    d = Data(size=28)
    d2 = d  # change to another size of pictures
    kclf = KMeans(clusters=NB_CLUSTERS).train(d.train_X)

    # TODO: Is it a good idea to train the CNN based on data that the Kmeans
    # classifies without having seen it ?
    train_X = [[] for i in xrange(NB_CLUSTERS)]
    train_Y = [[] for i in xrange(NB_CLUSTERS)]
    for X, y in zip(d2.train_X, d2.train_Y):
        idx = kclf.predict(X)
        train_X[idx].append(X)
        train_Y[idx].append(y)

    # creating a CNN for each
    cnns = [CNN(batch_size=100, train_X=train_X[i], train_Y=train_Y[i]).train()
            for i in xrange(NB_CLUSTERS)]

    # Benchmark the algo:
    predictions = []
    for X in d2.valid_X:
        idx = kclf.predict(X)
        predictions.append(cnns[idx].predict(X))
    online_score(predictions, d2.valid_Y)
