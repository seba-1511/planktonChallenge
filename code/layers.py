# -*- coding: utf-8 -*-
from pylearn2.models import mlp
from pylearn2.models.maxout import MaxoutConvC01B
from utils import (
    NB_CLASSES,
    DROPOUT,
    DROPOUT_PROB
)

dropout_scale = DROPOUT_PROB**2 if DROPOUT else 1.0

conv0 = mlp.ConvRectifiedLinear(
    layer_name='c0',
    output_channels=96,
    irange=0.070,
    kernel_shape=(5, 5),
    kernel_stride=(1, 1),
    pool_shape=(2, 2),
    pool_stride=(1, 1),
    border_mode='valid',
    W_lr_scale=dropout_scale,
    # max_kernel_norm=1.9365
)
conv1 = mlp.ConvRectifiedLinear(
    layer_name='c1',
    output_channels=128,
    irange=0.070,
    kernel_shape=(3, 3),
    kernel_stride=(1, 1),
    pool_shape=(2, 2),
    pool_stride=(1, 1),
    border_mode='valid',
    W_lr_scale=dropout_scale,
    # max_kernel_norm=1.9365
)
conv2 = mlp.ConvRectifiedLinear(
    layer_name='c2',
    output_channels=128,
    irange=0.070,
    kernel_shape=[3, 3],
    kernel_stride= (1, 1),
    pool_shape=[2, 2],
    pool_stride=[1, 1],
    W_lr_scale=dropout_scale,
    max_kernel_norm=1.9365
)
mout0 = MaxoutConvC01B(
    layer_name='m0',
    num_pieces=6,
    num_channels=96,
    irange=0.070,
    kernel_shape=[2, 2],
    kernel_stride=(1, 1),
    pool_shape=[2, 2],
    pool_stride=[1, 1],
    W_lr_scale=dropout_scale,
)
mout1 = MaxoutConvC01B(
    layer_name='m1',
    num_pieces=6,
    num_channels=128,
    irange=.05,
    kernel_shape=[5, 5],
    pool_shape=[4, 4],
    pool_stride=[2, 2],
    W_lr_scale=dropout_scale,
    # max_kernel_norm=1.9365
)
sigmoid = mlp.Sigmoid(
    layer_name='s0',
    dim=10000,
    sparse_init=2000,
)
sigmoid1 = mlp.Sigmoid(
    layer_name='s1',
    dim=2000,
    sparse_init=500,
)
rect0 = mlp.RectifiedLinear(
    layer_name='r0',
    dim=1512,
    irange=0.0757,
    # sparse_init=200,
    W_lr_scale=dropout_scale,
)
rect1 = mlp.RectifiedLinear(
    layer_name='r1',
    dim=1512,
    # sparse_init=200,
    irange=0.054,
)
smax = mlp.Softmax(
    layer_name='softmax',
    n_classes=NB_CLASSES,
    irange=0.054,
)
