# -*- coding: utf-8 -*-
from pylearn2.models import mlp
from pylearn2.models.maxout import MaxoutConvC01B
from utils import (
    NB_CLASSES,
)

conv = mlp.ConvRectifiedLinear(
    layer_name='c0',
    output_channels=96,
    irange=0.070,
    kernel_shape=(2, 2),
    kernel_stride=(1, 1),
    pool_shape=(2, 2),
    pool_stride=(1, 1),
    border_mode='valid',
    # W_lr_scale=0.25,
    # max_kernel_norm=1.9365
)
conv2 = mlp.ConvRectifiedLinear(
    layer_name='c2',
    output_channels=126,
    irange=0.070,
    kernel_shape=(3, 3),
    kernel_stride=(1, 1),
    pool_shape=(3, 3),
    pool_stride=(1, 1),
    border_mode='valid',
    # W_lr_scale=0.25,
    # max_kernel_norm=1.9365
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
    dim=2000,
    irange=0.070,
    # sparse_init=200,
    # W_lr_scale=0.25,
)
rect1 = mlp.RectifiedLinear(
    layer_name='r1',
    dim=2000,
    # sparse_init=200,
    irange=0.054,
)
smax = mlp.Softmax(
    layer_name='y',
    n_classes=NB_CLASSES,
    irange=0.054,
)
