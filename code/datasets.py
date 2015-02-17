# -*- coding: utf-8 -*-

from utils import (
    convert_one_hot
)
# train = RotationalDDM(X=d.train_X, y=convert_one_hot(d.train_Y))
# valid = RotationalDDM(X=d.valid_X, y=convert_one_hot(d.valid_Y))
# test = RotationalDDM(X=d.test_X, y=convert_one_hot(d.test_Y))
train = DenseDesignMatrix(X=d.train_X - 0.5, y=convert_one_hot(d.train_Y))
valid = DenseDesignMatrix(X=d.valid_X - 0.5, y=convert_one_hot(d.valid_Y))
test = DenseDesignMatrix(X=d.test_X - 0.5, y=convert_one_hot(d.test_Y))
