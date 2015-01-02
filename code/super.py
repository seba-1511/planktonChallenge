# -*- coding: utf-8 -*-

import numpy as np

from data.data import Data, CLASS_NAMES, CLASSES
from multiprocessing import cpu_count
from classifier.cnn import CNN
from sklearn import datasets, svm, neighbors, linear_model, naive_bayes, tree, ensemble
from sklearn.ensemble import RandomForestClassifier
from score import online_score

NB_CPUS = cpu_count()

class Super(object):
    def __init__(self, d):
        print 'Creation of Super'
        self.data = d
        self.train_X = d.convertBinaryValues(d.train_X)
        self.train_Y = d.train_Y
        self.test_X = d.convertBinaryValues(d.test_X)
        self.test_Y = d.test_Y
        self.valid_X = d.valid_X if d.valid_X else d.test_X
        self.valid_Y = d.valid_Y if d.valid_Y else d.test_Y
        self.specialists = self.create_specialists()
        self.general = self.create_general()

    def train(self):
        print 'Started training'
        self.train_specialists()
        self.train_general()

    def create_specialists(self):
        print 'Creating specialized models'
        specialists = [
            # Protists
            neighbors.KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                algorithm='auto'),
            # Crustaceans
            tree.DecisionTreeClassifier(),
            # Pelagic Tunicates
            neighbors.KNeighborsClassifier(),
            # Artifacts
            neighbors.KNeighborsClassifier(),
            # Chactognaths
            RandomForestClassifier(
                max_features='sqrt',
                n_estimators=15,
                n_jobs=NB_CPUS),
            # Planktons
            RandomForestClassifier(
                max_features='sqrt',
                n_estimators=15,
                n_jobs=NB_CPUS),
            # Copepods
            RandomForestClassifier(
                max_features='sqrt',
                n_estimators=15,
                n_jobs=NB_CPUS),
            # Ctenophores
            neighbors.KNeighborsClassifier(),
            # Shrimp-like
            neighbors.KNeighborsClassifier(),
            # Detritus
            neighbors.KNeighborsClassifier(),
            # Diotoms
            neighbors.KNeighborsClassifier(),
            # Echinoderms
            neighbors.KNeighborsClassifier(),
            # Gelatinous Zooplanktons
            neighbors.KNeighborsClassifier(),
            # Fish
            neighbors.KNeighborsClassifier(),
            # Gastropods
            naive_bayes.MultinomialNB(),
            # Hydromedusae
            neighbors.KNeighborsClassifier(),
            # invertebrate larvae
            RandomForestClassifier(
                max_features='sqrt',
                n_estimators=15,
                n_jobs=NB_CPUS),
            # Siphonophores
            RandomForestClassifier(
                max_features='sqrt',
                n_estimators=15,
                n_jobs=NB_CPUS),
            # Trichodesmium
            svm.SVC(),
            # Unknowns
            RandomForestClassifier(
                max_features='sqrt',
                n_estimators=15,
                n_jobs=NB_CPUS),
        ]
        return specialists
#        [RandomForestClassifier(
#                    max_features='sqrt',
#                    n_estimators=15,
#                    n_jobs=NB_CPUS) for i in CLASS_NAMES]

    def create_general(self):
        print 'Creating general model'
        self.data.create_parent_labels()
        train_y = self.data.train_parent_Y
        test_y = self.data.test_parent_Y
        cnn = CNN(
             alpha=0.1,
             batch_size=100,
             train_X=self.train_X,
             train_Y=train_y,
             test_X=self.test_X,
             test_Y=test_y,
             epochs=100,
             instance_id=None)
        return cnn

    def train_specialists(self):
        print 'Training Specialists'
        self.data.create_categories()
        for i, name in enumerate(CLASS_NAMES):
            print 'Training for ' + name
            train_X = self.data.convertBinaryValues(self.data.train_cat_X[name])
            train_y = self.data.train_cat_Y[name]
            test_X = self.data.convertBinaryValues(self.data.test_cat_X[name])
            test_y = self.data.test_cat_Y[name]
            clf = self.specialists[i]
            clf.fit(train_X, train_y)
            print 'Score for ' + name + ': ' + str(clf.score(test_X, test_y))
#            print 'Log loss for ' + name + ': ' + str(
#                log_loss(test_y, clf.predict_proba(test_X))
#            )
            self.specialists[i] = clf

    def train_general(self):
        print 'Training General'
        self.general.train()
        predictions = []
        print 'Making predictions'
        for X in self.test_X:
             predictions.append(self.general.predict([X, ]))
        print 'Score for general: ' + str(
            online_score(predictions, self.data.test_parent_Y)
        )

    def score(self):
        print 'Scoring Super'
        predictions = []
        for X in self.test_X:
            predictions.append(self.predict(X))
        return online_score(predictions, self.test_Y)

    def predict(self, X):
        top_best = 3
        prediction = np.zeros(121, dtype=float)
        gen_pred = self.general.predict([X, ])[0]
        best_indices = self.get_best_indices(gen_pred, top_best)
        for idx in best_indices:
            pred_score = self.specialists[idx].predict_proba(X)[0]
            name = CLASS_NAMES[idx]
            cls = CLASSES[name]
            for i, score in enumerate(pred_score):
                prediction[cls[i]] += (score * gen_pred[idx])
        return prediction

    def get_best_indices(self, array, top=3):
        bests = np.array([-10 for i in xrange(top)])
        indices = [-1 for i in xrange(top)]
        array = np.array(array)
        for idx, nb in enumerate(array):
            if nb > np.amin(bests):
                index = np.argmin(bests)
                bests[index] = i
                indices[index] = idx
        return indices

if __name__ == '__main__':
    d = Data(size=60, train_perc=0.8, test_perc=0.2, valid_perc=0.0)
    a = Super(d)
    a.train()
    print a.score()
