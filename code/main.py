# -*- coding: utf-8 -*-

import numpy as np
from data.data import Data

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
    data = d.train_X
    print np.shape(data)
