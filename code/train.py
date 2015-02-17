# -*- coding: utf-8 -*-

from pylearn2 import monitor
from pylearn2.train_extensions import best_params
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.training_algorithms import bgd, sgd, learning_rule
from pylearn2.costs.mlp import dropout, WeightDecay, L1WeightDecay, Default
from pylearn2.costs.cost import SumOfCosts


from utils import (
    SAVE,
    BATCH_SIZE,
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
    termination_criterion=EpochCounter(3200),
    # termination_criterion=MonitorBased(channel_name='valid_y_nll'),
)
