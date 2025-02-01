import numbers
import numpy as np
import sklearn

from sklearn.utils import check_X_y, check_array, column_or_1d
from sklearn.utils.multiclass import check_classification_targets
import sklearn.exceptions
import joblib

from joblib import Parallel, delayed  # Исправленный импорт
from sklearn.utils.validation import has_fit_parameter, check_is_fitted


class DecisionTreeBaseline():
    """Base class for ordinal meta-classifier.

    """

    def __init__(self):
        return self
    
    def fit(self, X, y, sample_weight=None):
        return self

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=['csr', 'csc'])

        # ---------------------------------------------Our CODE
        n_samples = X.shape[0]
        prediction = np.zeros((n_samples, 1))
        
        for i in range(0, n_samples):
            if X[i,"Scenario"] == "C":
                if X[i,"VoterType"] == "LB":
                    prediction[i] = 2 #Q' vote
                else:
                    prediction[i] = 1 #Q vote  
            else:
                if X[i,"Scenario"] in ["E","F"]:
                    if X[i,"VoterType"] == "TRT":
                        prediction[i] = 1 #Q vote   
                    else:
                        prediction[i] = 2 #Q' vote
                        
                else:
                    prediction[i] = 1 #Q vote


        return prediction



