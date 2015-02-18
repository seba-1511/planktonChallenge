# -*- coding: utf-8 -*-

import gzip
import cPickle as pk
from data.data import Data, RotationalDDM
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from utils import (
    convert_one_hot
)

#   mnist = fetch_mldata('MNIST original')
# debug()
#    mnist.data = (mnist.data.astype(float) / 255)
# data = Data(size=IMG_SIZE, train_perc=0.8, valid_perc=0.1, test_perc=0.1)
# f = gzip.open('data.pkl.gz', 'wb')
# pk.dump(data, f, protocol=pk.HIGHEST_PROTOCOL)
# f.close()
# train()
#


def get_dataset():
    print 'Loading Data...'
    f = gzip.open('data.pkl.gz', 'rb')
    d = pk.load(f)
    f.close()
    # train = RotationalDDM(X=d.train_X, y=convert_one_hot(d.train_Y))
    # valid = RotationalDDM(X=d.valid_X, y=convert_one_hot(d.valid_Y))
    # test = RotationalDDM(X=d.test_X, y=convert_one_hot(d.test_Y))
    train = DenseDesignMatrix(X=d.train_X - 0.5, y=convert_one_hot(d.train_Y))
    valid = DenseDesignMatrix(X=d.valid_X - 0.5, y=convert_one_hot(d.valid_Y))
    test = DenseDesignMatrix(X=d.test_X - 0.5, y=convert_one_hot(d.test_Y))
    return train, valid, test
