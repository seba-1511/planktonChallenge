# -*- coding: utf-8 -*-

import os
import inspect
import cPickle as pickle
import numpy as np

from os.path import exists
from skimage.io import imread
from skimage.transform import resize

TRAIN_PERCENT = 0.10
VALID_PERCENT = 0.0
TEST_PERCENT = 0.2


CLASS_NAMES = ('Protists', 'Crustaceans', 'PelagicTunicates', 'Artifacts', 'Chaetognaths', 'Planktons', 'Copepods', 'Ctenophores', 'ShrimpLike', 'Detritus', 'Diotoms', 'Echinoderms', 'GelatinousZoolankton', 'Fish', 'Gastropods', 'Hydromedusae', 'InvertebrateLarvae', 'Siphonophores', 'Trichodesmium', 'Unknowns')
    
CLASSES = {
    'Protists': {0, 1, 2, 82, 83, 84, 85, 86, 90, 91},
    'Crustaceans': {3, 27, 106},
    'PelagicTunicates': {4, 5, 6, 7, 113, 114, 115, 116, 117},
    'Artifacts': {8, 9},
    'Chaetognaths': {10, 11, 12},
    'Planktons': {13, 81},
    'Copepods': {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26},
    'Ctenophores': {28, 29, 30, 31},
    'ShrimpLike': {32, 47, 48, 92, 93, 94, 95},
    'Detritus': {33, 34, 35, 49},
    'Diotoms': {36, 37},
    'Echinoderms': {38, 39, 40, 41, 42, 43, 44, 45},
    'GelatinousZoolankton': {46, 80},
    'Fish': {50, 51, 52, 53, 54, 55},
    'Gastropods': {56, 87, 88, 89},
    'Hydromedusae': {57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77},
    'InvertebrateLarvae': {78, 79, 107, 112},
    'Siphonophores': {96, 97, 98, 99, 100, 101, 102, 103, 104, 105},
    'Trichodesmium': {108, 109, 110, 111},
    'Unknowns': {118, 119, 120},
}


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
            
    def create_categories(self):
        train_classes_X = dict()
        train_classes_Y = dict()
        test_classes_X = dict()
        test_classes_Y = dict()
        for name in CLASS_NAMES:
            train_classes_X[name] = []
            train_classes_Y[name] = []
        
            

    def convertBinaryValues(self, image_set=None, threshold=0.5):
        return (image_set > threshold).astype(int)

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
    d = Data(size=28)    
