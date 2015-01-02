# -*- coding: utf-8 -*-

import warnings
import numpy as np
import cPickle as pk

from data.data import Data
from classifier.kmeans import KMeans
from classifier.cnn import CNN
from score import online_score
from nolearn.dbn import DBN

from sklearn.svm import LinearSVC, SVC
from sklearn import neighbors
from sklearn.metrics import log_loss
from MLCompare import MLCompare

warnings.filterwarnings("ignore")

NB_CLUSTERS = 20

CLASS_NAMES = ('Protists', 'Crustaceans', 'PelagicTunicates', 'Artifacts', 'Chaetognaths', 'Planktons', 'Copepods', 'Ctenophores', 'ShrimpLike', 'Detritus',
               'Diotoms', 'Echinoderms', 'GelatinousZoolankton', 'Fish', 'Gastropods', 'Hydromedusae', 'InvertebrateLarvae', 'Siphonophores', 'Trichodesmium', 'Unknowns')

def train_steroids_knn(d=None):
     kclf = KMeans(clusters=NB_CLUSTERS - 1).train(d.train_X)

     #TODO: Is it a good idea to train the CNN based on data that the Kmeans
     #classifies without having seen it ?
     print 'creating the datasets'
     train_X = [[] for i in xrange(NB_CLUSTERS)]
     train_Y = [[] for i in xrange(NB_CLUSTERS)]
     for X, y in zip(d2.train_X, d2.train_Y):
         idx = kclf.predict(X)
         train_X[idx].append(X)
         train_Y[idx].append(y)

     #creating a CNN for each
     print 'creating and training the cnn'
     cnns = [CNN(batch_size=1, train_X=train_X[i], train_Y=train_Y[i], epochs=20).train()for i in xrange(NB_CLUSTERS)]

     #Benchmark the algo:
     print 'benchmarking the algorithm'
     predictions = []
     for X in d2.valid_X:
         idx = kclf.predict(X)
         predictions.append(cnns[idx].predict([X, ]))
     print online_score(predictions, d2.valid_Y)

def train_specialists(d=None):
    d.create_categories()
    cnns = []
    for i, name in enumerate(CLASS_NAMES):
        print 'Training for ' + name
        train_X = d.train_cat_X[name]
        train_y = d.train_cat_Y[name]
        test_X = d.test_cat_X[name]
        test_y = d.test_cat_Y[name]
        svm = neighbors.KNeighborsClassifier()
        svm.fit(train_X, train_y)
        print 'Score for ' + name + ': ' + str(svm.score(test_X, test_y))
        print 'Log loss for ' + name + ': ' + str(online_score(svm.predict(test_X), test_y))
#        cnn = CNN(
#             alpha=0.1,
#             batch_size=100,
#             train_X=train_X,
#             train_Y=train_y,
#             test_X=test_X,
#             test_Y=test_y,
#             epochs=200,
#             instance_id=12000+i)
#        cnn.train()
##        cnns.append[cnn]
#        predictions = []
#        for X in test_X:
#             predictions.append(cnn.predict([X, ]))
#        print 'Score for ' + name + ': ' + str(online_score(predictions, test_y))
        # pk.dump(cnn, open('cnn_' + name + '.pl', 'wb'))
    # pk.dump(cnns, open('cnns.pl', 'wb'))

def train_general(d=None):
    d.create_parent_labels()
    print 'One-Hot labeling'
    train_X = d.convertBinaryValues(d.train_X)
    train_y = d.train_parent_Y
    test_X = d.convertBinaryValues(d.test_X)
    test_y = d.test_parent_Y
    print 'creating CNN'
    cnn = CNN(
         alpha=0.5,
         batch_size=100,
         train_X=train_X,
         train_Y=train_y,
         test_X=test_X,
         test_Y=test_y,
         epochs=50,
         instance_id=None)
    print 'Training CNN'
    cnn.train()
    predictions = []
    print 'Making predictions'
    for X in test_X:
         predictions.append(cnn.predict([X, ]))
    print 'Score for general: ' + str(online_score(predictions, test_y))
#    svm = SVC(probability=True)
#    svm.fit(train_X, train_y)
#    probs = svm.predict_proba(test_X)
#    print log_loss(test_y, probs)

def test_dbn(d=None):
    d.create_parent_labels()
    print 'One-Hot labeling'
    train_X = d.convertBinaryValues(d.train_X)
    train_y = d.train_parent_Y
    test_X = d.convertBinaryValues(d.test_X)
    test_y = d.test_parent_Y
    print 'creating CNN'
    dbn = DBN(
        [train_X.shape[1], 300, 10],
        learn_rates = 0.3,
        learn_rate_decays = 0.9,
        epochs = 10,
        verbose = 1)
    print 'Training DBN'
    dbn.fit(train_X, train_y)
    print 'Scoring DBN'
    print 'Score: ' + str(dbn.score(test_X, test_y))
    print 'Log loss: ' + str(online_score(dbn.predict(test_X), test_y))



if __name__ == '__main__':
    d = Data(size=60, train_perc=0.8, test_perc=0.2, valid_perc=0.0)
    test_dbn(d)
#    train_specialists(d=d)
#    train_general(d=d)

