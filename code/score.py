# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from math import log


def logloss(act, pred):
    """ Vectorised computation of logloss """

    # cap in official Kaggle implementation,
    # per forums/t/1576/r-code-for-logloss
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)

    # compute logloss function (vectorised)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act)
             * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


def online_score(predictions=[[]], targets=[]):
    # return logloss(targets, predictions)

    # One-liner:
    epsilon = 1e-15
    predictions = np.array(predictions)
    targets = np.array(targets)
    targets = np.argmin(targets, axis=0) if len(targets.shape) > 1 else targets
    if np.shape(predictions)[1] == 1:
        return -np.average(
            [log(max(epsilon, p[0][t] / p[0].sum()))
             for p, t in zip(predictions, targets)]
        )
    return -np.average(
        [log(max(epsilon, p[t] / p.sum()))
         for p, t in zip(predictions, targets)]
    )
    # score = 0.0
    # for i, entry in enumerate(predictions):
    #     shp = np.shape(entry)
    #     if shp and shp[0] == 1:
    #         entry = entry[0]
    #     tot = np.sum(entry)
    #     score += log(entry[targets[i]] / tot)
    # return -score / len(predictions)


def score(predictions=[]):
    return online_score()

# print online_score([[0.4999, 0.5001] for i in xrange(10)], [0 for i in
# xrange(10)])
