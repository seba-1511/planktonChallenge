# -*- coding: utf-8 -*-
import warnings

from pylearn2.space import Conv2DSpace

from layers import (
    conv0,
    conv1,
    conv2,
    rect0,
    rect1,
    mout0,
    mout1,
    sigmoid,
    smax,
)
from utils import (
    IMG_SIZE,
)
from submit import (
    submit,
    get_predict_fn,
)
from train import (
    get_trainer,
    loop,
)
from datasets import (
    get_dataset,
    get_gabe_planktons,
)

from pylearn2.models import mlp


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train, valid, test = get_dataset()
    trainer = get_trainer(train, valid, test)

    in_space = Conv2DSpace(
        shape=[IMG_SIZE, IMG_SIZE],
        num_channels=1,
        # axes=['c', 0, 1, 'b']
    )

    net = mlp.MLP(
        layers=[conv0, conv1, conv2, rect0, rect1, smax],
        input_space=in_space,
        # nvis=784,
    )

    net = loop(trainer, net)
    submit(net)
