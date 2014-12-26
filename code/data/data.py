# -*- coding: utf-8 -*-

import os
import inspect
import cPickle as pickle
import numpy as np

from skimage.io import imread
from skimage.transform import resize


def create_thumbnail(size=25, img=None):
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
    pickle.dump(train, open('train' + str(size) + '.pkl', 'wb'))


def get_train_data(size=25):
    curr_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    filename = os.path.join(curr_dir, 'train' + str(size) + '.pkl')
    if not os.path.exists(filename):
        create_thumbnail(size)
    return pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    create_thumbnail(25)
