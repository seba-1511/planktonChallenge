#-*- coding: utf-8 -*-
"""
TODO:
- Test with a larger dataset.
- Add choice for regression vs classification.
    (separate methods or function params)
- Add more algorithms.
"""

import time
import numpy as np
from sklearn import datasets, svm, neighbors, linear_model, naive_bayes, tree, ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, cross_val_score

from multiprocessing import Process, Manager

# Show the debug messages ?
DEBUG = True

# Computer Settings
NB_THREADS = 4

# Application settings
TEST_SET_PERCENTAGE = 0.1
GRID_SEARCH_CV = 4
KFOLD_NB_FOLDS = 15


class Learner():

    """
    A helper class, to train and compare classifiers in parallel threads.
    """

    def __init__(self, algorithm, features, targets):
        self.name = algorithm[0]
        self.algorithm = algorithm[1]
        self.parameters = algorithm[2]
        (self.train_data,
         self.train_targets,
         self.test_data,
         self.test_targets) = self.buildSets(features, targets)
        self.optimal_algo = self.algorithm()
        self.optimal_params = {}
        self.averaged_score = 0.0
        self.shared = Manager().Namespace()
        self.train_time = 0
        self.prediciton_time = 0
        self.train = Process(
            target=self.parallelTrain, args=(self.train_data, self, self.shared))
        self.score = Process(target=self.scoreLearner, args=())

    def startTraining(self):
        self.shared.optimal_algo = None
        self.shared.optimal_params = None
        self.shared.averaged_score = None

        self.train.start()
        if DEBUG:
            print 'learning started'

    def waitTraining(self):
        self.train.join()
        self.optimal_algo = self.shared.optimal_algo
        self.optimal_params = self.shared.optimal_params
        self.averaged_score = self.shared.averaged_score
        if DEBUG:
            print '%s learning joined' % self.name

    def parallelTrain(self, data, Learner, shared):
        shared.optimal_algo, shared.optimal_params = Learner.findBestParams()
        shared.averaged_score = Learner.findAverageCVResults(
            Learner.train_data, Learner .train_targets)

    def findBestParams(self):
        clf = GridSearchCV(
            self.algorithm(), self.parameters, n_jobs=NB_THREADS, cv=GRID_SEARCH_CV)
        clf.fit(self.train_data, self.train_targets)
        if DEBUG:
            print ''
            message = 'The optimal algorithm for %s is: %s'
            print message % (self.name, clf.best_estimator_)
            print ''
        return clf.best_estimator_, clf.best_params_

    def findAverageCVResults(self, features, targets):
        cross_validator = KFold(len(features), n_folds=KFOLD_NB_FOLDS)
        scores = cross_val_score(
            self.optimal_algo, features, targets, n_jobs=NB_THREADS, cv=cross_validator)
        scores = np.mean(scores)
        if DEBUG:
            print ''
            message = 'Fitted %s with training scores %s'
            print message % (self.name, scores)
        return scores

    def buildSets(self, features, targets):
        # Shuffling the data
        shuffled = np.asarray(features)
        shuffled = np.hstack(
            (shuffled, np.zeros((shuffled.shape[0], 1), dtype=shuffled.dtype)))
        shuffled[:, -1] = targets
        np.random.shuffle(shuffled)
        features = shuffled[:, :-1]
        targets = shuffled[:, -1]
        # Creating test and train sets
        test_length = int(TEST_SET_PERCENTAGE * len(targets))
        train_data, train_targets = features[
            0:-test_length], targets[0:-test_length]
        test_data, test_targets = features[
            -test_length:], targets[-test_length:]
        if DEBUG:
            print ''
            print 'Train set size: %s examples.' % len(train_data)
            print 'Test set size: %s inputs.' % len(test_data)
        return (train_data, train_targets, test_data, test_targets)

    def scoreLearner(self):
        self.train_time = time.time()
        self.optimal_algo.fit(self.train_data, self.train_targets)
        self.train_time = time.time() - self.train_time
        self.prediciton_time = time.time()
        score = self.optimal_algo.score(self.test_data, self.test_targets)
        self.prediciton_time = time.time() - self.prediciton_time
        message = '%s:      Test: %.2f     CV: %.2f     Train Time: %.3fs     Score Time: %.3fs     Parameters: %s'
        var_mess = (self.name, score, self.averaged_score,
                    self.train_time, self.prediciton_time, self.optimal_params)
        print ''
        print ''
        print message % var_mess
        print ''
        print ''


def MLCompare(features, targets):
    algorithms = [
        # ('SVM', svm.SVC,
        #  {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        #   'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        #   'degree': [2, 3, 4, 5, 6, 7, 8],
        #   'gamma': [0.0, 1.0, 10.0, 100.0]
        #   }),

        ('SGD', linear_model.SGDClassifier,
         {
         'loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
          'penalty': ['l1', 'l2', 'elasticnet'],
          'alpha': [0.0001, 0.001, 0.01, 0.1],
          'fit_intercept': [True, False],
          'n_jobs': [NB_THREADS]
          }),


        ('kNN', neighbors.KNeighborsClassifier,
         {'n_neighbors': [1, 5, 10, 25, 50, 100, 200, 500, 1000],
          'weights': ['uniform', 'distance'],
          'algorithm': ['auto'],
          }),


        ('Logistic Classifier', linear_model.LogisticRegression,
         {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
          'penalty': ['l1', 'l2'],
          'fit_intercept': [True, False]
          }),


        ('MultinomialNB', naive_bayes.MultinomialNB,
         {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
          'fit_prior': [True, False]
          }),


        ('GaussianNB', naive_bayes.GaussianNB, {}),


        ('Decision Tree', tree.DecisionTreeClassifier,
         {'criterion': ['gini', 'entropy'],
          'max_features': [None, 'sqrt', 'log2'],
          'max_depth': [None, 2, 4, 8, 16, 32, 64],
          'min_samples_split': [1, 2, 4, 8, 16, 32],
          'min_samples_leaf': [1, 2, 4, 8, 16, 32],
          }),


        ('Random Forest', ensemble.RandomForestClassifier,
         {'n_estimators': [2, 5, 10, 15],
          'criterion': ['gini', 'entropy'],
          'max_features': [None, 'sqrt', 'log2'],
          'max_depth': [None, 2, 4, 8, 16, 32, 64],
          'min_samples_split': [1, 2, 4, 8, 16, 32],
          'min_samples_leaf': [1, 2, 4, 8, 16, 32],
          'bootstrap': [True, False],
          'n_jobs': [NB_THREADS],
          }),


        ('AdaBoost', ensemble.AdaBoostClassifier,
         {'n_estimators': [2, 5, 10, 25, 50, 100, 200],
          'base_estimator': [tree.DecisionTreeClassifier()],
          'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
          'algorithm': ['SAMME.R', 'SAMME']
          }),
    ]

    learners = [Learner(x, features, targets) for x in algorithms]

    [l.startTraining() for l in learners]

    [l.waitTraining() for l in learners]

    print ''
    print ''
    print 'RESULTS:'

    [l.score.start() for l in learners]

    [l.score.join() for l in learners]

    print ''
    print ''
    print 'Evaluation Ended.'

# Uncomment to see an example:
# data = datasets.load_iris()
# MLCompare(data.data, data.target)
#data = datasets.load_digits()
#timer = time.time()
#MLCompare(data.data, data.target)
#print 'Total execution: %.3fs' % (time.time() - timer)
