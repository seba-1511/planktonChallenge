# -*- coding: utf-8 -*-
import theano
import numpy as np

from layers import *
from utils import *
from submit import *
from train import *

from pylearn2.models import mlp

if __name__ == '__main__':
    in_space = Conv2DSpace(
        shape=[IMG_SIZE, IMG_SIZE],
        num_channels=1,
        # axes=['c', 0, 1, 'b']
    )
    net = mlp.MLP(
        layers=[conv, conv2, rect, rect1, smax],
        input_space=in_space,
        # nvis=784,
    )
    trainer.setup(net, train)
