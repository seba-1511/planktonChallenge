# -*- coding: utf-8 -*-

import gzip
import cPickle as pk
import numpy as np
from data.data import Data, RotationalDDM
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from os.path import isfile

from utils import (
        IMG_SIZE,
        )
from utils import (
        convert_one_hot
        )
#   mnist = fetch_mldata('MNIST original')
#   mnist.data = (mnist.data.astype(float) / 255)


DATA_FILE = 'data.pkl.gz'

def get_dataset():
    print 'Loading Data...'
    filename = str(IMG_SIZE) + DATA_FILE
    if isfile(filename):
        print 'Loading from gzip...'
        f = gzip.open(filename, 'rb')
        d = pk.load(f)
        f.close()
    else:
        print 'Reading images...'
        d = Data(size=IMG_SIZE, train_perc=0.8, valid_perc=0.1, test_perc=0.1)
        # f = gzip.open(DATA_FILE, 'wb')
        # pk.dump(d, f, protocol=pk.HIGHEST_PROTOCOL)
        # f.close()
        f = gzip.open(filename, 'wb')
        pk.dump(d, f, protocol=pk.HIGHEST_PROTOCOL)
        f.close()

    # train = RotationalDDM(X=d.train_X, y=convert_one_hot(d.train_Y))
    # valid = RotationalDDM(X=d.valid_X, y=convert_one_hot(d.valid_Y))
    # test = RotationalDDM(X=d.test_X, y=convert_one_hot(d.test_Y))
    train = DenseDesignMatrix(X=d.train_X - 0.5, y=convert_one_hot(d.train_Y))
    valid = DenseDesignMatrix(X=d.valid_X - 0.5, y=convert_one_hot(d.valid_Y))
    test = DenseDesignMatrix(X=d.test_X - 0.5, y=convert_one_hot(d.test_Y))
    return train, valid, test


from pdb import set_trace as debug
def get_gabe_planktons():
    """
        Does not work yet.
    """
    f = gzip.open('plankton.pkl.gz', 'rb')
    d = pk.load(f)
    f.close()
    debug()
    train_X = np.array(d[0][0:25000])
    valid_X = np.array(d[0][25000:30000])
    test_X = np.array(d[0][30000:31000])
    train_Y = np.array(d[1][0:25000])
    valid_Y = np.array(d[1][25000:30000])
    test_Y = np.array(d[1][30000:31000])
    train = DenseDesignMatrix(X=train_X - 0.5, y=train_Y)
    valid = DenseDesignMatrix(X=valid_X - 0.5, y=valid_Y)
    test = DenseDesignMatrix(X=test_X - 0.5, y=test_Y)
    return train, valid, test


