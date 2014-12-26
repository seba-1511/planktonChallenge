# -*- coding: utf-8 -*-

from sklearn import cluster
from multiprocessing import cpu_count

NB_PROC = cpu_count()


class KMeans(object):

    def __init__(self, clusters=18):
        self.clf = cluster.KMeans(n_clusters=clusters, n_jobs=NB_PROC)

    def train(self, X):
        print 'training K-means'
        self.clf.fit(X)
        return self

    def predict(self, X):
        return self.clf.predict(X)
