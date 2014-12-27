# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from math import log


def logloss(act, pred):
    """ Vectorised computation of logloss """
    
    #cap in official Kaggle implementation, 
    #per forums/t/1576/r-code-for-logloss
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    
    #compute logloss function (vectorised)
    ll = sum(   act*sp.log(pred) + 
                sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def online_score(predictions=[[]], targets=[]):
    return logloss(targets, predictions)
    score = 0.0
    for i, entry in enumerate(predictions):
        entry = entry[0]
        tot = np.sum(entry)
        score += log(entry[targets[i]] / tot)
    return -score / len(predictions)


def score(predictions=[]):
    return online_score()

# print online_score([[0.4999, 0.34, 2.0] for i in xrange(10)], [0 for i in xrange(10)])
