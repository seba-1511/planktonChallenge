# -*- coding: utf-8 -*-

import os
import inspect
import cPickle as pickle
import numpy as np

from os.path import exists
from skimage.io import imread
from skimage.transform import resize

TRAIN_PERCENT = 0.3
VALID_PERCENT = 0.1
TEST_PERCENT = 0.1


class Data(object):

    def __init__(self, size=28):
        print 'loading data'
        data, targets = self.get_data(size)
        nb_train = int(TRAIN_PERCENT * len(targets))
        nb_valid = int(VALID_PERCENT * len(targets))
        nb_test = int(TEST_PERCENT * len(targets))
        total = nb_train + nb_valid + nb_test
        total_perc = TRAIN_PERCENT + VALID_PERCENT + TEST_PERCENT
        data, targets = self.shuffle_data(data[:total], targets[:total])
        self.train_X = data[:nb_train]
        self.train_Y = targets[:nb_train]
        self.valid_X = data[nb_train:nb_train + nb_valid]
        self.valid_Y = targets[nb_train:nb_train + nb_valid]
        self.test_X = data[nb_train + nb_valid:total]
        self.test_Y = targets[nb_train + nb_valid:total]
        saved = (data[:total], targets[:total])
        pickle.dump(
            saved, open('train' + str(size) + '_' + str(total_perc) + '.pkl', 'wb'))

    def create_thumbnail(self, size, img=None):
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

    def get_data(self, size):
        curr_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        filename = os.path.join(curr_dir, 'train' + str(size) + '.pkl')
        total = TRAIN_PERCENT + VALID_PERCENT + TEST_PERCENT
        previous_file = os.path.join(
            curr_dir, 'train' + str(size) + '_' + str(total) + '.pkl')
        if exists(previous_file):
            print 'loaded from smaller dump'
            return pickle.load(open(previous_file, 'rb'))
        if not exists(filename):
            return self.create_thumbnail(size)
        return pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    Data().create_thumbnail(25)
