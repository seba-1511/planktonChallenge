# -*- coding: utf-8 -*-

import os
import inspect
import cPickle as pickle
import numpy as np

from os.path import exists
from skimage.io import imread
from skimage.transform import resize, rotate, swirl
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from random import randint
from math import sqrt

TRAIN_PERCENT = 0.7
VALID_PERCENT = 0.1
TEST_PERCENT = 0.2
SAVE = True

CLASS_NAMES = (
    'Protists',
    'Crustaceans',
    'PelagicTunicates',
    'Artifacts',
    'Chaetognaths',
    'Planktons',
    'Copepods',
    'Ctenophores',
    'ShrimpLike',
    'Detritus',
    'Diotoms',
    'Echinoderms',
    'GelatinousZoolankton',
    'Fish',
    'Gastropods',
    'Hydromedusae',
    'InvertebrateLarvae',
    'Siphonophores',
    'Trichodesmium',
    'Unknowns'
)

CLASSES = {
    'Protists': (0, 1, 2, 82, 83, 84, 85, 86, 90, 91),
    'Crustaceans': (3, 27, 106),
    'PelagicTunicates': (4, 5, 6, 7, 113, 114, 115, 116, 117),
    'Artifacts': (8, 9),
    'Chaetognaths': (10, 11, 12),
    'Planktons': (13, 81),
    'Copepods': (14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26),
    'Ctenophores': (28, 29, 30, 31),
    'ShrimpLike': (32, 47, 48, 92, 93, 94, 95),
    'Detritus': (33, 34, 35, 49),
    'Diotoms': (36, 37),
    'Echinoderms': (38, 39, 40, 41, 42, 43, 44, 45),
    'GelatinousZoolankton': (46, 80),
    'Fish': (50, 51, 52, 53, 54, 55),
    'Gastropods': (56, 87, 88, 89),
    'Hydromedusae': (57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77),
    'InvertebrateLarvae': (78, 79, 107, 112),
    'Siphonophores': (96, 97, 98, 99, 100, 101, 102, 103, 104, 105),
    'Trichodesmium': (108, 109, 110, 111),
    'Unknowns': (118, 119, 120),
}


def print_images(images, folder='cropped/'):
    import imageio as imio
    width = sqrt(images.shape[1])
    for i, img in enumerate(images):
        img = img.reshape(width, width)
        imio.imsave(folder + str(i) + '.jpg', img)


class RotationalDDM(DenseDesignMatrix):

    def __init__(self, X, y, y_labels=None):
        self.original_X = X
        super(RotationalDDM, self).__init__(X=X, y=y, y_labels=y_labels)

    def rotation(self, x):
        width = sqrt(x.shape[0])
        angle = randint(0, 359)
        img = x.reshape(width, width)
        return rotate(img, angle, mode='nearest').ravel()

    def parallel_rotate(self, X):
        return [self.rotation(x) for x in X]

    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False):
        self.X = self.parallel_rotate(self.original_X)
        self.X = np.array(self.X)
        print 'Rotated'
        return super(RotationalDDM, self).iterator(
            mode=mode,
            batch_size=batch_size,
            num_batches=num_batches,
            rng=rng,
            data_specs=data_specs,
            return_tuple=return_tuple
        )


class Data(object):

    def __init__(self, size=28, train_perc=TRAIN_PERCENT, valid_perc=VALID_PERCENT, test_perc=TEST_PERCENT, augmentation=0):
        print 'loading data'
        self.size = size
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.test_perc = test_perc
        self.augmentation = augmentation
        data, targets = self.get_data(size)
        nb_train = int(self.train_perc * len(targets))
        nb_valid = int(self.valid_perc * len(targets))
        nb_test = int(self.test_perc * len(targets))
        total = nb_train + nb_valid + nb_test
        total_perc = self.train_perc + self.valid_perc + self.test_perc
        data = np.around(data, 4)
        data, targets = self.shuffle_data(data[:total], targets[:total])
        self.train_X = data[:nb_train]
        self.train_Y = targets[:nb_train]
        self.valid_X = data[nb_train:nb_train + nb_valid]
        self.valid_Y = targets[nb_train:nb_train + nb_valid]
        self.test_X = data[nb_train + nb_valid:total]
        self.test_Y = targets[nb_train + nb_valid:total]
        name = 'train' + str(size) + '_' + str(total_perc)
        self.save_set(name, data[:total], targets[:total])

    def create_categories(self):
        train_classes_X = dict()
        train_classes_Y = dict()
        test_classes_X = dict()
        test_classes_Y = dict()
        for name in CLASS_NAMES:
            train_classes_X[name] = []
            train_classes_Y[name] = []
            test_classes_X[name] = []
            test_classes_Y[name] = []
        for img_i, img in enumerate(self.train_X):
            label = self.train_Y[img_i]
            for name in CLASS_NAMES:
                if label in CLASSES[name]:
                    train_classes_X[name].append(img)
                    train_classes_Y[name].append(label)
                    break
        for img_i, img in enumerate(self.test_X):
            label = self.test_Y[img_i]
            for name in CLASS_NAMES:
                if label in CLASSES[name]:
                    test_classes_X[name].append(img)
                    test_classes_Y[name].append(label)
                    break
        total = self.train_perc + self.valid_perc + self.test_perc
        for name in CLASS_NAMES:
            filename = name + str(self.size) + '_' + str(total)
            dir = 'classes/'
            X = train_classes_X[name]
            y = train_classes_Y[name]
            self.save_set('train_' + filename, X, y, dir)
            X = test_classes_X[name]
            y = test_classes_Y[name]
            self.save_set('test_' + filename, X, y, dir)
        self.train_cat_X = train_classes_X
        self.train_cat_Y = train_classes_Y
        self.test_cat_X = test_classes_X
        self.test_cat_Y = test_classes_Y

    def create_parent_labels(self):
        new_train_labels = []
        for y in self.train_Y:
            for new_label, name in enumerate(CLASS_NAMES):
                if y in CLASSES[name]:
                    new_train_labels.append(new_label)
                    break
        new_test_labels = []
        for y in self.test_Y:
            for new_label, name in enumerate(CLASS_NAMES):
                if y in CLASSES[name]:
                    new_test_labels.append(new_label)
                    break
        self.train_parent_Y = new_train_labels
        self.test_parent_Y = new_test_labels

    def save_set(self, name, X, y,  directory=''):
        if SAVE:
            curr_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
            filename = os.path.join(curr_dir, directory + name + '.pkl')
            f = open(filename, 'wb')
            pickle.dump((X, y), f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    def convertBinaryValues(self, image_set=None, threshold=0.5):
        binary = np.array(image_set) > threshold
        return binary.astype(int)

    def augment_data(self, image, target):
        images = [image.ravel(), ]
        targets = [target, ]
        image_modifiers = (
            lambda x: rotate(x, 90),
            lambda x: rotate(x, 180),
            lambda x: rotate(x, 270),
            lambda x: rotate(x, 45),
            lambda x: swirl(x)
        )
        for i in xrange(self.augmentation):
            img = image_modifiers[i](image)
            images.append(img.ravel())
            targets.append(target)
        return images, targets

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
                # Important to put -1, to have it 0-based.
                target = class_id - 1
                new_images, new_targets = self.augment_data(image, target)
                images.extend(new_images)
                targets.extend(new_targets)
        train = (images, targets)
        self.save_set('train' + str(size), images, targets)
        # f = open(curr_dir + '/train' + str(size) + '.pkl', 'wb')
        # pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
        # f.close()
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
        total = self.train_perc + self.valid_perc + self.test_perc
        previous_file = os.path.join(
            curr_dir, 'train' + str(size) + '_' + str(total) + '.pkl')
        if exists(previous_file):
            print 'loaded from smaller dump'
            f = open(previous_file, 'rb')
            content = pickle.load(f)
            f.close()
            return content
        if not exists(filename):
            return self.create_thumbnail(size)
        return pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    import time
    start = time.time()
    d = Data(size=28, train_perc=0.1, valid_perc=0.0,
             test_perc=0.1, augmentation=4)
    end = time.time()
    print 'Augmented:' + str(end - start)
    print np.shape(d.train_X)
    start = time.time()
    d = Data(size=28, train_perc=0.1, valid_perc=0.0,
             test_perc=0.1, augmentation=0)
    end = time.time()
    print 'Not Augmented:' + str(end - start)
    print np.shape(d.train_X)
