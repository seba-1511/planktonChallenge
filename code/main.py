# -*- coding: utf-8 -*-

import numpy as np
from data.data import get_train_data

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
    data, t = get_train_data()
    print np.shape(data)
    print np.shape(t)
