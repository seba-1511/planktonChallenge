# -*- coding: utf-8 -*-

from math import log


def online_score(predictions=[[]], targets=[]):
    score = 0.0
    for i, entry in enumerate(predictions):
        score += log(entry[targets[i]])
    return -score / len(predictions)


def score(predictions=[]):
    return online_score()

print online_score([[1, 2, 3], [1, 23], [1, ]], [1, 1, 0])
