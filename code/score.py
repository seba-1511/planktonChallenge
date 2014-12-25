# -*- coding: utf-8 -*-

from math import log


def online_score(predictions=[[]], targets=[]):
    score = 0.0
    for i, entry in enumerate(predictions):
        for j, prob in enumerate(entry):
            score += log(0.7)
    return -score / len(predictions)


def score(predictions=[]):
    return online_score()

print online_score([[1, 2, 3], [1, 23], [1, ]], [])
