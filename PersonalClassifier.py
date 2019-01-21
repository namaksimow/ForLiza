import numbers
import numpy as np
import sklearn
import pandas as pd

from six import with_metaclass
from abc import ABCMeta
from sklearn.base import ClassifierMixin
from sklearn.ensemble.base import BaseEnsemble
from sklearn.base import clone
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

class PersonalClassifier(with_metaclass(ABCMeta, BaseEnsemble, ClassifierMixin)):
    """Base class for oneshot personal clasifier ensemble 
    
    """

    def __init__(self,
                 id_index,
                 base_classifier = RandomForestClassifier(n_estimators=40),
                 n_upsample = 1,
                 general_base_classifier = False):
        self.base_classifier = base_classifier
        self.id_index = id_index
        self.personal_classifiers = dict()
        self.n_upsample = n_upsample
        self.general_base_classifier = general_base_classifier


    def _make_classifier(self):
        classifier = deepcopy(self.base_classifier)#clone(self.base_classifier)
        return classifier

    def fit(self, X, y, sample_weight=None):
        #Fit general population classifier
        if self.general_base_classifier == True:
            self.base_classifier.fit(X,y)

        #Fit persinal classifiers
        all_voters = pd.DataFrame(X[:,self.id_index]).drop_duplicates()
        for voter in all_voters.iterrows():
            voter_classifier = self._make_classifier()
            X_v = X[X[:,self.id_index] == voter[1][0]]
            y_v = y[X[:,self.id_index] == voter[1][0]]
            combined = np.c_[X_v,y_v]
            combined_upsample = resample(combined, replace=True, n_samples=self.n_upsample*X_v.shape[0], random_state=0)
            X_v = combined_upsample[:,0:X_v.shape[1]]
            y_v = combined_upsample[:,-1]
            if self.general_base_classifier == True:
                voter_classifier.partial_fit(X_v, y_v, [1,2,3])
            else:
                voter_classifier.fit(X_v, y_v)
            self.personal_classifiers[voter[1][0]] = voter_classifier
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        prediction = np.zeros(n_samples)
        for i in range(0, n_samples):
            voterID = X[i,self.id_index]
            if self.personal_classifiers.keys().__contains__(voterID):
                voter_classifier = self.personal_classifiers[voterID]
                prediction[i] = voter_classifier.predict(np.reshape(X[i,:],(1,X.shape[1])))[0]
            else:
                prediction[i] = 1 #predict q if we didn't had the voter is training set.
        return prediction