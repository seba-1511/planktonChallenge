# -*- coding: utf-8 -*-

import numpy as np

from data.data import Data
from classifier.kmeans import KMeans

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

if __name__ == '__main__':
    d = Data()
    kclf = KMeans().train(d.train_X)
    for i in xrange(25):
        kclf.predict(d.train_X[i])
