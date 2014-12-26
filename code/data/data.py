# -*- coding: utf-8 -*-

import os
import inspect
import cPickle as pickle
import numpy as np

from skimage.io import imread
from skimage.transform import resize

TRAIN_PERCENT = 0.1
VALID_PERCENT = 0.1
TEST_PERCENT = 0.1


class Data:

    def __init__(self, size=25):
        print 'loading data'
        data, targets = self.get_data(size)
        data, targets = self.shuffle_data(data, targets)
        nb_train = TRAIN_PERCENT * len(targets)
        nb_valid = VALID_PERCENT * len(targets)
        nb_test = TEST_PERCENT * len(targets)
        self.train_X = data[:nb_train]
        self.train_Y = targets[:nb_train]
        self.valid_X = data[nb_train:nb_valid]
        self.valid_Y = targets[nb_train:nb_valid]
        self.test_X = data[nb_valid:nb_test]
        self.test_Y = targets[nb_valid:nb_test]

    def create_thumbnail(self, size=25, img=None):
        print 'processing raw images'
        if img:
            return resize(img, (size, size))
        curr_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        folders = os.walk(os.path.join(curr_dir, '../../data/train/'))
        images = []
        classes = []
        targets = []
        for class_id, folder in enumerate(folders):
            classes.append(folder[0][17:])
            for img in folder[2]:
                if img.index('.jpg') == -1:
                    continue
                image = imread(folder[0] + '/' + img)
                image = resize(image, (size, size))
                image = np.array(image).ravel()
                images.append(image)
                targets.append(class_id)
        train = (images, targets)
        pickle.dump(
            train, open(curr_dir + '/train' + str(size) + '.pkl', 'wb'))
        return train

    def shuffle_data(self, X, y):
        shp = np.shape(X)
        shuffle = np.zeros((shp[0], shp[1] + 1))
        shuffle[:, :-1] = X
        shuffle[:, -1] = y
        np.random.shuffle(shuffle)
        return (shuffle[:, :-1], shuffle[:, -1])

    def get_data(self, size=25):
        curr_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        filename = os.path.join(curr_dir, 'train' + str(size) + '.pkl')
        if not os.path.exists(filename):
            return self.create_thumbnail(size)
        return pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    Data().create_thumbnail(25)
