# -*- coding: utf-8 -*-

import warnings
import theano
import pylearn2
import numpy as np
import cPickle as pk

from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score, classification_report

from data.data import Data, RotationalDDM
from pdb import set_trace as debug
from itertools import izip_longest

from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2 import termination_criteria, monitor
from pylearn2.train_extensions import best_params
from pylearn2.models import mlp
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.training_algorithms import bgd, sgd, learning_rule
from pylearn2.costs.mlp import dropout

warnings.filterwarnings("ignore")

NB_CLASSES = 121


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks, from itertools recipes"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def convert_one_hot(data):
    return np.array([[1 if y == c else 0 for c in xrange(NB_CLASSES)] for y in data])


def convert_categorical(data):
    return np.argmax(data, axis=1)


def classify(inp, model, batch_size, vec_space):
    if len(np.shape(inp)) == 1:
        inp = np.array([inp, ])
    in_space = model.get_input_space()
    inp = vec_space.np_format_as(inp, in_space)
    return np.argmax(model.fprop(theano.shared(inp, name='inputs')).eval())


def score(dataset, model, batch_size):
    nr_correct = 0
    features, targets = np.array(dataset, dtype=float)
    vec_space = VectorSpace(len(features[0]))
    for X, y in zip(grouper(features, batch_size), grouper(targets, batch_size)):
        if classify(X, model, batch_size, vec_space) == np.argmax(y, axis=1):
            nr_correct += 1
    # for features, label in zip(dataset[0], dataset[1]):
    #     if classify(features, model, batch_size, vec_space) == np.argmax(label):
    #         nr_correct += 1
    return float(nr_correct) / float(len(dataset))


def train(d):
    print 'Creating dataset'
    # load mnist here
    # X = d.train_X
    # y = d.train_Y
    # test_X = d.test_X
    # test_Y = d.test_Y
    # nb_classes = len(np.unique(y))
    # train_y = convert_one_hot(y)
    # train_set = DenseDesignMatrix(X=X, y=y)
    train = RotationalDDM(X=d.train_X, y=convert_one_hot(d.train_Y))
    valid = RotationalDDM(X=d.valid_X, y=convert_one_hot(d.valid_Y))
    test = RotationalDDM(X=d.test_X, y=convert_one_hot(d.test_Y))
    # train = mnist.MNIST(
    #     which_set='train',
    #     start=0,
    #     stop=50000,
    # )
    # valid = mnist.MNIST(
    #     which_set='train',
    #     start=50000,
    #     stop=60000,
    # )
    # test = mnist.MNIST(which_set='test')

    print 'Setting up'
    batch_size = 512
    conv = mlp.ConvRectifiedLinear(
        layer_name='c0',
        output_channels=20,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        # W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    mout = MaxoutConvC01B(
        layer_name='m0',
        num_pieces=4,
        num_channels=128,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[3, 3],
        pool_stride=[2, 2],
        W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    mout2 = MaxoutConvC01B(
        layer_name='m1',
        num_pieces=4,
        num_channels=96,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    sigmoid = mlp.Sigmoid(
        layer_name='Sigmoid',
        dim=2000,
        sparse_init=500,
    )
    sigmoid2 = mlp.Sigmoid(
        layer_name='s2',
        dim=750,
        sparse_init=225,
    )
    smax = mlp.Softmax(
        layer_name='y',
        n_classes=NB_CLASSES,
        irange=0.
    )
    in_space = Conv2DSpace(
        shape=[28, 28],
        num_channels=1,
        axes=['c', 0, 1, 'b']
    )
    net = mlp.MLP(
        layers=[mout, mout2, sigmoid, sigmoid2, smax],
        input_space=in_space,
        # nvis=784,
    )
    # Momentum:
    mom_init = 0.3
    mom_final = 0.99
    mom_start = 1
    mom_saturate = 100
    mom_adjust = learning_rule.MomentumAdjustor(
        mom_final,
        mom_start,
        mom_saturate,
    )
    mom_rule = learning_rule.Momentum(mom_init)

    # Learning Rate:
    lr_init = 1
    lr_saturate = 100
    lr_decay_factor = 0.1
    lr_adjust = sgd.LinearDecayOverEpoch(lr_init, lr_saturate, lr_decay_factor)

    # Monitor:
    monitor_save_best = best_params.MonitorBasedSaveBest('test_y_nll',
                                                         'best_model.pkle')

    trainer = bgd.BGD(
        batch_size=batch_size,
        line_search_mode='exhaustive',
        conjugate=1,
        updates_per_batch=10,
        monitoring_dataset={
            'train': train,
            'valid': valid,
            'test': test
        },
        termination_criterion=termination_criteria.MonitorBased(
            channel_name='valid_y_misclass')
    )
    trainer = sgd.SGD(
        learning_rate=0.1,
        learning_rule=mom_rule,
        cost=dropout.Dropout(),
        batch_size=batch_size,
        monitoring_dataset={
            'train': train,
            'valid': valid,
            'test': test
        },
        termination_criterion=termination_criteria.MonitorBased(
            channel_name='valid_y_misclass')
    )
    trainer.setup(net, train)
    epoch = 0
    test_monitor = []
    prev_nll = 10
    while True:
        print 'Training...', epoch
        trainer.train(dataset=train)
        net.monitor()
        monitor_save_best.on_monitor(net, valid, trainer)
        nll = monitor.read_channel(net, 'test_y_nll') + 0
        test_monitor.append(
            (nll, monitor.read_channel(net, 'test_y_misclass'))
        )
        if nll < prev_nll:
            f = open('best.pkle', 'wb')
            pk.dump(net, f, protocol=pk.HIGHEST_PROTOCOL)
            f.close()
        f = open('monitor.pkle', 'wb')
        pk.dump(test_monitor, f, protocol=pk.HIGHEST_PROTOCOL)
        f.close()
        print 'Custom test score', score((test.X, test.y), net, batch_size)
        mom_adjust.on_monitor(net, valid, trainer)
        lr_adjust.on_monitor(net, valid, trainer)
        epoch += 1

"""
    TODO: Get above .98 with momentum and maxout and dropout. And then add several ones.
"""

if __name__ == '__main__':
 #   mnist = fetch_mldata('MNIST original')
    # debug()
#    mnist.data = (mnist.data.astype(float) / 255)
    data = Data(size=28, train_perc=0.75, valid_perc=0.1, test_perc=0.15)
    train(d=data)
    # train()
