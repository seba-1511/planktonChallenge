# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
 from cnn import CNN
 from sklearn.ensemble import RandomForestClassifier

CLASS_NAMES = (
    'Protists',
    'Crustaceans',
    'PelagicTunicates',
    'Artifacts',
    'Chaetognaths',
    'Planktons',
    'Copepods',
    'Ctenophores',
    'ShrimpLike',
    'Detritus',
    'Diotoms',
    'Echinoderms',
    'GelatinousZoolankton',
    'Fish',
    'Gastropods',
    'Hydromedusae',
    'InvertebrateLarvae',
    'Siphonophores',
    'Trichodesmium',
    'Unknowns'
)

NB_CPUS = cpu_count()

class Super(object):
    def __init__(self, d):
        print 'Creation of Super'
        self.data = d
        self.train_X = d.train_X
        self.train_Y = d.train_Y
        self.test_X = d.test_X
        self.test_Y = d.test_Y
        self.valid_X = d.valid_X if d.valid_X else d.test_X
        self.valid_Y = d.valid_Y if d.valid_Y else d.test_Y

    def train(self):
        print 'Started training'
        self.specialists = self.create_specialists()
        self.general = self.create_general()
        self.train_specialists()
        self.train_general()

    def create_specialists(self):
        print 'Creating specialized models'
        return [RandomForestClassifier(
                    mex_features='sqrt',
                    n_estimators=15,
                    n_jobs=NB_CPUS) for i in CLASS_NAMES]

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
             epochs=200,
             instance_id=None)
        return cnn

    def train_specialists(self):
        print 'Training Specialists'
        self.data.create_categories()

    def train_general(self):
        pass
