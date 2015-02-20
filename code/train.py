# -*- coding: utf-8 -*-
import cPickle as pk

from pylearn2 import monitor
from pylearn2.train_extensions import best_params
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.training_algorithms import bgd, sgd, learning_rule
from pylearn2.costs.mlp import dropout, WeightDecay, L1WeightDecay, Default
from pylearn2.costs.cost import SumOfCosts

from utils import (
    SAVE,
    BATCH_SIZE,
    NB_EPOCHS,
    IMG_SIZE,
    DROPOUT,
    DROPOUT_PROB,
)

# Global Variables
"""
    TODO: Change those variables, this is disgusting.
"""
if SAVE:
    monitor_save_best = best_params.MonitorBasedSaveBest(
        'test_y_nll', 'best_model.pkl')
test_monitor = []
prev_nll = 10
momentum_adjustor = None
momentum_rule = None
lr_adjustor = None


def save_best_history(model, trainer):
    global monitor_save_best
    global test_monitor
    global prev_nll
    monitor_save_best.on_monitor(
        model, trainer.monitoring_dataset['valid'], trainer)
    nll = monitor.read_channel(model, 'test_y_nll') + 0
    test_monitor.append(
        (nll, monitor.read_channel(model, 'test_y_misclass'))
    )
    if nll < prev_nll:
        f = open('best.pkl', 'wb')
        pk.dump(model, f, protocol=pk.HIGHEST_PROTOCOL)
        f.close()
    f = open('monitor.pkl', 'wb')
    pk.dump(test_monitor, f, protocol=pk.HIGHEST_PROTOCOL)
    f.close()


def adjust_parameters(model, trainer, dataset):
    global momentum_adjustor
    global lr_adjustor
    if momentum_adjustor:
        momentum_adjustor.on_monitor(model, dataset, trainer)
    if lr_adjustor:
        lr_adjustor.on_monitor(model, dataset, trainer)


def init_momentum():
    global momentum_adjustor
    global momentum_rule
    mom_init = 0.9
    mom_final = 0.99
    mom_start = 8
    mom_saturate = 35
    momentum_adjustor = learning_rule.MomentumAdjustor(
        mom_final,
        mom_start,
        mom_saturate,
    )
    momentum_rule = learning_rule.Momentum(mom_init)


def init_learning_rate():
    global lr_adjustor
    lr_init = 8
    lr_saturate = 35
    lr_decay_factor = 0.07
    lr_adjustor = sgd.LinearDecayOverEpoch(
        lr_init, lr_saturate, lr_decay_factor)


def get_BGD(train, valid, test):
    trainer = bgd.BGD(
        batch_size=BATCH_SIZE,
        line_search_mode='exhaustive',
        conjugate=1,
        updates_per_batch=10,
        monitoring_dataset={
            'train': train,
            'valid': valid,
            'test': test
        },
        termination_criterion=MonitorBased(channel_name='valid_y_misclass')
    )
    return trainer


def get_SGD(train, valid, test):
    global momentum_rule
    regularizer = dropout.Dropout(DROPOUT_PROB) if DROPOUT else Default()
    trainer = sgd.SGD(
        learning_rate=0.01,
        learning_rule=momentum_rule,
        cost=SumOfCosts(
            costs=[
                regularizer,
                WeightDecay([5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 0.0]),
            ]
        ),
        batch_size=BATCH_SIZE,
        monitoring_dataset={
            'train': train,
            'valid': valid,
            'test': test
        },
        termination_criterion=EpochCounter(NB_EPOCHS),
        # termination_criterion=MonitorBased(channel_name='valid_y_nll'),
    )
    return trainer


def get_trainer(train, valid=None, test=None):
    init_momentum()
    init_learning_rate()
    trainer = get_SGD(train, valid, test)
    return trainer


def loop(trainer, model):
    trainer.setup(model, trainer.monitoring_dataset['train'])
    epoch = 0
    while True:
        print 'Training...', epoch
        trainer.train(dataset=trainer.monitoring_dataset['train'])
        model.monitor()
        if not trainer.continue_learning(model):
            break
        if SAVE:
            save_best_history(model, trainer)
        adjust_parameters(model, trainer, trainer.monitoring_dataset['valid'])
        epoch += 1
    return model
