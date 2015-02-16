# -*- coding: utf-8 -*-

import warnings
import theano
import pylearn2
import numpy as np
import cPickle as pk

from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score, classification_report

from data.data import Data, RotationalDDM
from submission import submit
from pdb import set_trace as debug
from itertools import izip_longest

from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2 import termination_criteria, monitor
from pylearn2.train_extensions import best_params
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.models import mlp
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.training_algorithms import bgd, sgd, learning_rule
from pylearn2.costs.mlp import dropout, WeightDecay, L1WeightDecay, Default
from pylearn2.costs.cost import SumOfCosts

warnings.filterwarnings("ignore")

NB_CLASSES = 121
IMG_SIZE = 28
SAVE = False


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks, from itertools recipes"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def convert_one_hot(data):
    return np.array([[1 if y == c else 0 for c in xrange(NB_CLASSES)] for y in data])


def convert_categorical(data):
    return np.argmax(data, axis=1)

def predict(data, model, batch_size, vec_space):
    data = np.asarray(data)
    # data.shape = (1, 784)
    res = []
    for X in grouper(data, batch_size):
        X = vec_space.np_format_as(X, model.get_input_space())
        res.append(ann.fprop(theano.shared(X, name='inputs')).eval())
    return res



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
    # train = RotationalDDM(X=d.train_X, y=convert_one_hot(d.train_Y))
    # valid = RotationalDDM(X=d.valid_X, y=convert_one_hot(d.valid_Y))
    # test = RotationalDDM(X=d.test_X, y=convert_one_hot(d.test_Y))
    train = DenseDesignMatrix(X=d.train_X - 0.5, y=convert_one_hot(d.train_Y))
    valid = DenseDesignMatrix(X=d.valid_X - 0.5, y=convert_one_hot(d.valid_Y))
    test = DenseDesignMatrix(X=d.test_X - 0.5, y=convert_one_hot(d.test_Y))

    print 'Setting up'
    batch_size = 256
    conv = mlp.ConvRectifiedLinear(
            layer_name='c0',
            output_channels=96,
            irange=0.235,
            kernel_shape=(4, 4),
            kernel_stride=(1, 1),
            pool_shape=(3, 3),
            pool_stride=(2, 2),
            border_mode='valid',
            # W_lr_scale=0.25,
            # max_kernel_norm=1.9365
            )
    conv2 = mlp.ConvRectifiedLinear(
            layer_name='c2',
            output_channels=128,
            irange=.235,
            kernel_shape=[4, 4],
            pool_shape=[3, 3],
            pool_stride=[2, 2],
            # W_lr_scale=0.25,
            max_kernel_norm=1.9365
            )
    conv3 = mlp.ConvRectifiedLinear(
            layer_name='c3',
            output_channels=128,
            irange=.235,
            kernel_shape=[5, 5],
            pool_shape=[4, 4],
            pool_stride=[2, 2],
            # W_lr_scale=0.25,
            max_kernel_norm=1.9365
            )
    mout = MaxoutConvC01B(
            layer_name='m0',
            num_pieces=6,
            num_channels=96,
            irange=.235,
            kernel_shape=[4, 4],
            pool_shape=[3, 3],
            pool_stride=[2, 2],
            # W_lr_scale=0.25,
            )
    mout2 = MaxoutConvC01B(
            layer_name='m1',
            num_pieces=6,
            num_channels=128,
            irange=.05,
            kernel_shape=[5, 5],
            pool_shape=[4, 4],
            pool_stride=[2, 2],
            W_lr_scale=0.25,
            max_kernel_norm=1.9365
            )
    sigmoid = mlp.Sigmoid(
            layer_name='Sigmoid',
            dim=10000,
            sparse_init=2000,
            )
    sigmoid2 = mlp.Sigmoid(
            layer_name='s2',
            dim=2000,
            sparse_init=500,
            )
    rect = mlp.RectifiedLinear(
            layer_name='r0',
            dim=1560,
            irange=0.070,
            # sparse_init=200,
            # W_lr_scale=0.25,
            )
    rect1 = mlp.RectifiedLinear(
            layer_name='r1',
            dim=512,
            sparse_init=200,
            irange=0.235,
            )
    smax = mlp.Softmax(
            layer_name='y',
            n_classes=NB_CLASSES,
            irange=0.235,
            )
    in_space = Conv2DSpace(
            shape=[IMG_SIZE, IMG_SIZE],
            num_channels=1,
            # axes=['c', 0, 1, 'b']
            )
    net = mlp.MLP(
            layers=[conv, rect, smax],
            input_space=in_space,
            # nvis=784,
            )
    # Momentum:
    mom_init = 0.5
    mom_final = 0.99
    mom_start = 1
    mom_saturate = 35
    mom_adjust = learning_rule.MomentumAdjustor(
            mom_final,
            mom_start,
            mom_saturate,
            )
    mom_rule = learning_rule.Momentum(mom_init)

    # Learning Rate:
    lr_init = 5
    lr_saturate = 35
    lr_decay_factor = 0.1
    lr_adjust = sgd.LinearDecayOverEpoch(lr_init, lr_saturate, lr_decay_factor)

    # Monitor:
    if SAVE:
        monitor_save_best = best_params.MonitorBasedSaveBest('test_y_nll',
                'best_model.pkl')

    # trainer = bgd.BGD(
    #     batch_size=batch_size,
    #     line_search_mode='exhaustive',
    #     conjugate=1,
    #     updates_per_batch=10,
    #     monitoring_dataset={
    #         'train': train,
    #         'valid': valid,
    #         'test': test
    #     },
    #     termination_criterion=termination_criteria.MonitorBased(
    #         channel_name='valid_y_misclass')
    # )
    trainer = sgd.SGD(
            learning_rate=0.05,
            learning_rule=mom_rule,
            cost=SumOfCosts(
                costs=[
                    Default(),
                    # dropout.Dropout(),
                    WeightDecay([1e-2, 1e-2, 1e-3]),
                ]
            ),
            batch_size=batch_size,
            monitoring_dataset={
                'train': train,
                'valid': valid,
                'test': test
                },
            termination_criterion=EpochCounter(32),
            # termination_criterion=MonitorBased(channel_name='valid_y_nll'),
            )
    trainer.setup(net, train)
    epoch = 0
    test_monitor = []
    prev_nll = 10
    while True:
        print 'Training...', epoch
        trainer.train(dataset=train)
        net.monitor()
        if not trainer.continue_learning(net):
            break
        if SAVE:
            monitor_save_best.on_monitor(net, valid, trainer)
            nll = monitor.read_channel(net, 'test_y_nll') + 0
            test_monitor.append(
                    (nll, monitor.read_channel(net, 'test_y_misclass'))
                    )
            if nll < prev_nll:
                f = open('best.pkl', 'wb')
                pk.dump(net, f, protocol=pk.HIGHEST_PROTOCOL)
                f.close()
            f = open('monitor.pkl', 'wb')
            pk.dump(test_monitor, f, protocol=pk.HIGHEST_PROTOCOL)
            f.close()
        # print 'Custom test score', score((test.X, test.y), net, batch_size)
        mom_adjust.on_monitor(net, valid, trainer)
        lr_adjust.on_monitor(net, valid, trainer)
        epoch += 1
    submit(predict, net, IMG_SIZE)

"""
    TODO: Get above .98 with momentum and maxout and dropout. And then add several ones.
"""

if __name__ == '__main__':
    #   mnist = fetch_mldata('MNIST original')
    # debug()
#    mnist.data = (mnist.data.astype(float) / 255)
    import gzip
    # data = Data(size=IMG_SIZE, train_perc=0.8, valid_perc=0.1, test_perc=0.1)
    # f = gzip.open('data.pkl.gz', 'wb')
    # pk.dump(data, f, protocol=pk.HIGHEST_PROTOCOL)
    # f.close()
    f = gzip.open('data.pkl.gz', 'rb')
    data = pk.load(f)
    f.close()
    train(d=data)
    # train()
