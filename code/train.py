# -*- coding: utf-8 -*-
import cPickle as pk

from pylearn2 import monitor
from pylearn2.train_extensions import best_params
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.training_algorithms import bgd, sgd, learning_rule
from pylearn2.costs.mlp import dropout, WeightDecay, L1WeightDecay, Default
from pylearn2.costs.cost import SumOfCosts

from pdb import set_trace as debug

from utils import (
    SAVE,
    BATCH_SIZE,
    NB_EPOCHS,
    IMG_SIZE
)

# Momentum:
mom_init = 0.3
mom_final = 0.99
mom_start = 8
mom_saturate = 35
mom_adjust = learning_rule.MomentumAdjustor(
    mom_final,
    mom_start,
    mom_saturate,
)
mom_rule = learning_rule.Momentum(mom_init)

# Learning Rate:
lr_init = 8
lr_saturate = 35
lr_decay_factor = 0.07
lr_adjust = sgd.LinearDecayOverEpoch(lr_init, lr_saturate, lr_decay_factor)


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
    trainer = sgd.SGD(
        learning_rate=0.03,
        learning_rule=mom_rule,
        cost=SumOfCosts(
            costs=[
                Default(),
                # dropout.Dropout(),
                WeightDecay([1e-2, 1e-2, 1e-2, 1e-2, 1e-3]),
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
    trainer = get_SGD(train, valid, test)
    return trainer


def loop(trainer, model):
    # Monitor:
    if SAVE:
        monitor_save_best = best_params.MonitorBasedSaveBest(
            'test_y_nll', 'best_model.pkl')
    trainer.setup(model, trainer.monitoring_dataset['train'])
    epoch = 0
    test_monitor = []
    prev_nll = 10
    while True:
        print 'Training...', epoch
        trainer.train(dataset=trainer.monitoring_dataset['train'])
        model.monitor()
        if not trainer.continue_learning(model):
            break
        if SAVE:
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
        mom_adjust.on_monitor(
            model, trainer.monitoring_dataset['valid'], trainer)
        lr_adjust.on_monitor(
            model, trainer.monitoring_dataset['valid'], trainer)
        epoch += 1
    return model
