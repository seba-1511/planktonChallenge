# -*- coding: utf-8 -*-

class Super(object):
    def __init__(self, train_X, train_Y, test_X, test_Y,
                 valid_X=None, Valid_Y=None):
        self.train_X = train_X