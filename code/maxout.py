# -*- coding: utf-8 -*-

import warnings
import theano
import numpy as np
import cPickle as pk

from data.data import Data
from classifier.kmeans import KMeans
from classifier.cnn import CNN
from score import online_score
from sklearn.neural_network import BernoulliRBM as RBM
from sklearn.metrics import log_loss

from sklearn import neighbors

from pdb import set_trace as debug

from pylearn2.models import mlp, maxout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.costs.mlp import dropout

warnings.filterwarnings("ignore")

NB_CLUSTERS = 20

CLASS_NAMES = ('Protists', 'Crustaceans', 'PelagicTunicates', 'Artifacts', 'Chaetognaths', 'Planktons', 'Copepods', 'Ctenophores', 'ShrimpLike', 'Detritus',
               'Diotoms', 'Echinoderms', 'GelatinousZoolankton', 'Fish', 'Gastropods', 'Hydromedusae', 'InvertebrateLarvae', 'Siphonophores', 'Trichodesmium', 'Unknowns')


def train_steroids_knn(d=None):
    kclf = KMeans(clusters=NB_CLUSTERS - 1).train(d.train_X)

    # TODO: Is it a good idea to train the CNN based on data that the Kmeans
    # classifies without having seen it ?
    print 'creating the datasets'
    train_X = [[] for i in xrange(NB_CLUSTERS)]
    train_Y = [[] for i in xrange(NB_CLUSTERS)]
    for X, y in zip(d.train_X, d.train_Y):
        idx = kclf.predict(X)
        train_X[idx].append(X)
        train_Y[idx].append(y)

    # creating a CNN for each
    print 'creating and training the cnn'
    cnns = [CNN(batch_size=1, train_X=train_X[i], train_Y=train_Y[
                i], epochs=20).train()for i in xrange(NB_CLUSTERS)]

    # Benchmark the algo:
    print 'benchmarking the algorithm'
    predictions = []
    for X in d.valid_X:
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
#        predictions = []
#        for X in test_X:
#             predictions.append(cnn.predict([X, ]))
#        print 'Score for ' + name + ': ' + str(online_score(predictions, test_y))
        # pk.dump(cnn, open('cnn_' + name + '.pl', 'wb'))
    # pk.dump(cnns, open('cnns.pl', 'wb'))


def train_general(d=None):
    d.create_parent_labels()
    print 'One-Hot labeling'
    train_X = d.train_X
    train_y = d.train_parent_Y
    test_X = d.test_X
    test_y = d.test_parent_Y
    print 'creating RBM'
    rbm = RBM(n_components=3600)
    train_X = rbm.fit_transform(train_X, train_y)
    test_X = rbm.transform(test_X)
    print 'creating CNN'
    cnn = CNN(
        alpha=0.1,
        batch_size=100,
        train_X=train_X,
        train_Y=train_y,
        test_X=test_X,
        test_Y=test_y,
        epochs=50,
        instance_id=None)
    print 'Training CNN'
    cnn.train()
    print 'Making predictions'
    predictions = []
    for X in test_X:
        predictions.append(cnn.predict([X, ]))
    print 'Score for general: ' + str(online_score(predictions, test_y))
#    svm = SVC(probability=True)
#    svm.fit(train_X, train_y)
#    probs = svm.predict_proba(test_X)
#    print log_loss(test_y, probs)


def train_general(d=None):
    d.create_parent_labels()
    print 'One-Hot labeling'
    train_X = d.train_X
    train_y = d.train_parent_Y
    test_X = d.test_X
    test_y = d.test_parent_Y
    print 'creating RBM'
    rbm = RBM(n_components=3600)
    train_X = rbm.fit_transform(train_X, train_y)
    test_X = rbm.transform(test_X)
    print 'creating CNN'
    cnn = CNN(
        alpha=0.1,
        batch_size=100,
        train_X=train_X,
        train_Y=train_y,
        test_X=test_X,
        test_Y=test_y,
        epochs=50,
        instance_id=None)
    print 'Training CNN'
    cnn.train()
    print 'Making predictions'
    predictions = []
    for X in test_X:
        predictions.append(cnn.predict([X, ]))
    print 'Score for general: ' + str(online_score(predictions, test_y))
#    svm = SVC(probability=True)
#    svm.fit(train_X, train_y)
#    probs = svm.predict_proba(test_X)
#    print log_loss(test_y, probs)


def train_pylearn_general(d=None):
    d.create_parent_labels()
    train_X = np.array(d.train_X)
    train_y = np.array(d.train_parent_Y)
    test_X = np.array(d.test_X)
    test_y = np.array(d.test_parent_Y)
    train_y = [
        [1 if y == c else 0 for c, _ in enumerate(CLASS_NAMES)] for y in train_y]
    train_y = np.array(train_y)
    train_set = DenseDesignMatrix(
        X=train_X, y=train_y, y_labels=len(CLASS_NAMES))
    print 'Setting up'
    h = maxout.MaxoutConvC01B(
        layer_name='h',
        num_channels=64,
	num_pieces=4,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        # max_kernel_norm=1.9365
    )
    h0 = maxout.MaxoutConvC01B(
        layer_name='h0',
        num_channels=64,
	num_pieces=4,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        # max_kernel_norm=1.9365
    )
    h1 = maxout.MaxoutConvC01B(
        layer_name='h1',
        num_channels=64,
	num_pieces=4,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        # max_kernel_norm=1.9365
    )
    out = mlp.Softmax(
        n_classes=len(CLASS_NAMES),
        layer_name='output',
        irange=.235,
        # istdev=0.05
    )
    epochs = EpochCounter(200)
    layers = [h, h0, h1, out]
    in_space = Conv2DSpace(
        shape=[d.size, d.size],
        num_channels=1,
	axes=['c', 0, 1, 'b'],
    )
    vec_space = VectorSpace(d.size ** 2)
    nn = mlp.MLP(layers=layers, input_space=in_space, batch_size=None)
    trainer = sgd.SGD(
        learning_rate=.005,
        cost=dropout.Dropout(),
        batch_size=10,
        termination_criterion=epochs,
        learning_rule=learning_rule.Momentum(init_momentum=0.5),
    )
    trainer.setup(nn, train_set)
    print 'Learning'
    test_X = vec_space.np_format_as(test_X, nn.get_input_space())
    train_X = vec_space.np_format_as(train_X, nn.get_input_space())
    # test_X = theano.shared(test_X)
    i = 0
    X = nn.get_input_space().make_theano_batch()
    Y = nn.fprop(X)
    predict = theano.function([X], Y)
    while trainer.continue_learning(nn):
        print '--------------'
        print 'Training Epoch ' + str(i)
        trainer.train(dataset=train_set)
        predictions = [predict([f, ])[0] for f in train_X]
        print np.min(predictions), np.max(predictions)
        print 'Logloss on train: ' + str(online_score(predictions, train_y))
        predictions = [predict([f, ])[0] for f in test_X]
        print np.min(predictions), np.max(predictions)
        print 'Logloss on test: ' + str(online_score(predictions, test_y))
        i += 1
        print ' '

if __name__ == '__main__':
    d = Data(size=100, train_perc=0.9, test_perc=0.1, valid_perc=0.0)
#    test_dbn(d)
#    train_specialists(d=d)
    train_pylearn_general(d=d)
