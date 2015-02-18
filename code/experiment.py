# -*- coding: utf-8 -*-
import theano
import numpy as np

from layers import (
    conv0,
    conv1,
    rect0,
    rect1,
    smax,
)
from utils import (
    IMG_SIZE,
)
from submit import (
    submit,
)
from train import (
    get_trainer,
)
from datasets import (
    get_datasets,
)

from pylearn2.models import mlp

if __name__ == '__main__':
    train, valid, test = get_datasets()
    trainer = get_trainer(train, valid, test)

    in_space = Conv2DSpace(
        shape=[IMG_SIZE, IMG_SIZE],
        num_channels=1,
        # axes=['c', 0, 1, 'b']
    )
    net = mlp.MLP(
        layers=[conv0, conv1, rect0, rect1, smax],
        input_space=in_space,
        # nvis=784,
    )

    net = train(trainer, model)
    submit()
