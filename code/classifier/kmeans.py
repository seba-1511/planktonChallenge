# -*- coding: utf-8 -*-

from sklearn import cluster
from multiprocessing import cpu_count

N_PROC = cpu_count()


class KMeans(object):

    def __init__(self):
        self.clf = cluster.KMeans(n_cluster=18, n_jobs=N_PROC)
        return self

    def train(self, X):
        self.clf.fit(X)
        return self

    def predict(self, X):
        return self.clf.predict(X)
