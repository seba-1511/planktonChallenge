# -*- coding: utf-8 -*-

import numpy as np
from math import log


def online_score(predictions=[[]], targets=[]):
    score = 0.0
    for i, entry in enumerate(predictions):
        tot = np.sum(entry)
        score += log(entry[targets[i]]/tot)
    return -score / len(predictions)


def score(predictions=[]):
    return online_score()

print online_score([[0.4999, ] for i in xrange(10)], [0 for i in xrange(10)])
