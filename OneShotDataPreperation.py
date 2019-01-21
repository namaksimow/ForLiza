import numbers
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import kurtosis
from scipy.stats import skew

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from keras.layers import Input, Dense
from keras.models import Model

from datetime import datetime
# Model and feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2
# Classification metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
from PersonalClassifier import PersonalClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import logistic


def _data_conversion(data_df, is_target, le):
    if is_target:
        data_df = data_df.astype("category")
        data_df = le.fit_transform(data_df)
    else:
        for c in data_df.columns:
            if data_df[c].dtype in (object, str, np.object, bool):
                if not (data_df[c].dtype in (int, float)):
                    data_df[c] = le.fit_transform(data_df[c])
    return data_df

class OneShotDataPreparation():
    """ Class for One Shot data preparation

    """
    @staticmethod
    def _prepare_dataset(features_df):
        le = sklearn.preprocessing.LabelEncoder()
        features_encoded_df = pd.DataFrame(
            preprocessing.normalize(preprocessing.scale(_data_conversion(features_df, False, le).as_matrix()), axis=0,
                                    norm='max'))

        # target_le = sklearn.preprocessing.LabelEncoder()
        # target_df = _data_conversion(target_df, True, target_le)

        return features_encoded_df#, target_df















