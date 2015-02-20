# -*- coding: utf-8 -*-

import numpy as np
from itertools import izip_longest

NB_CLASSES = 121
IMG_SIZE = 60
SAVE = False
NB_EPOCHS = 2500
BATCH_SIZE = 128
NB_EPOCHS = 2500
DROPOUT = True
DROPOUT_PROB = 0.8


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks, from itertools recipes"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def convert_one_hot(data):
    return np.array([[(1 if y == c else 0)
                      for c in xrange(NB_CLASSES)] for y in data])


def convert_categorical(data):
    return np.argmax(data, axis=1)
