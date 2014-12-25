# -*- coding: utf-8 -*-

from skimage.transform import resize


def create_thumbnail(size=25, img=None):
    if img:
        return resize(img, (size, size))
    print 'test'

if __name__ == '__main__':
    create_thumbnail(25)
