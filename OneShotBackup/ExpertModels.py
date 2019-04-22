import numbers
import numpy as np
import sklearn
import pandas as pd
from OneShotDataPreperation import OneShotDataPreparation as osp
import MDLP.MDLP as mdlp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy

from sklearn.utils import check_X_y, check_array, column_or_1d
from sklearn.utils.multiclass import check_classification_targets

from sklearn.externals.joblib import Parallel, delayed #For parallel computing TODO: check if we need to be parallel or not
from sklearn.utils.validation import has_fit_parameter, check_is_fitted


def _extract_num_votes(V_i):
    return V_i.NumVotes[0]

def _get_max_threshold(V_i):
    numVotes = _extract_num_votes(V_i)
    return int(numVotes*0.7)


def _determine_split_preferences(v_ij, split_feature):
    below = 0
    above = 0
    if split_feature == "GAP12_poll":
        below = 1
        above = 2
    if split_feature == "GAP13_poll":
        below = 1
        above = 3
    return below, above

def _threshold_range_accuracy(r, V_i, split_feature):
    for v_ij in V_i:
        preference_below, preference_above = _determine_split_preferences(v_ij, split_feature)
        if v_ij.Action == preference_above & v_ij.loc[0, split_feature] <= r.range[0]:
            r.errors_below = r.errors_below + 1
        elif v_ij.Action == preference_below & v_ij.loc[0, split_feature] >= r.range[1]:
            r.errors_above = r.errors_above + 1
        elif v_ij.Action == preference_below & v_ij.loc[0, split_feature] <= r.range[0]:
            r.correct_below = r.correct_below + 1
        elif v_ij.Action == preference_above & v_ij.loc[0, split_feature] >= r.range[1]:
            r.correct_above = r.correct_above + 1
    return r


def _total_ranges_accuracy(R):
    #R is list\array of ranges r
    r_total = ThresholdRange()
    for r in R:
        r_total.errors_below = r_total.errors_below + r.errors_below
        r_total.errors_above = r_total.errors_above + r.errors_above
        r_total.correct_below = r_total.correct_below + r.correct_below
        r_total.correct_above = r_total.correct_above + r.correct_above
    return r_total


def _most_likely_threshold_range_for_voter_i(V_i, split_feature):
    numVotes = _extract_num_votes(V_i)
    gaps = list({0, V_i.split_feature, _get_max_threshold(V_i)})
    gaps.sort()
    min_error = np.inf
    r_best = None
    for gapIndex in range(0, len(gaps) - 1):
        cur_range = range(gaps[gapIndex], gaps[gapIndex + 1])
        r = ThresholdRange(range = cur_range)
        r.numVotes = numVotes
        r = _threshold_range_accuracy(r, V_i, split_feature)
        total_error = r.errors_below + r.errors_above
        if total_error < min_error:
            min_error = total_error
            r_best = r
    return r_best


def _most_likely_threshold_ranges(X, y, split_feature):
    R = set()
    voters = pd.DataFrame(X[["VoterID", "SessionIDX"]].drop_duplicates())
    for voter in voters:
        V_i = pd.concat([X.loc[X['VoterID'] == voter.VoterID,] , y], axis=1, join='inner')
        r_best = _most_likely_threshold_range_for_voter_i(V_i, split_feature)
        r_best.voter = voter
        R.add(r_best)

    return R


def _threshold_probability_estimation(t, R):
    prob = (1/len(R))*(np.sum([(1/(np.max(r.range)-np.min(r.range))) for r in R]))
    return prob

def _sample_probability_estimation(V_i, Z_floor, Z_ceiling, t, split_feature):
    sample_prob = 1
    for v_ij in V_i:
        gap = v_ij.split_feature
        below, above = _determine_split_preferences(v_ij, split_feature)
        if gap > t:
            if v_ij.Action == above:
                sample_prob = sample_prob*Z_ceiling
            else:
                sample_prob = sample_prob*(1 - Z_ceiling)
        else:
            if v_ij.Action == above:
                sample_prob = sample_prob*Z_floor
            else:
                sample_prob = sample_prob*(1 - Z_floor)

    return sample_prob

def _threshold_likelihood_estimation(V_i, R_without_i, Z_floor, Z_ceiling, n_votes, split_feature):
    L = list()
    for t in range(0, _get_max_threshold(n_votes)):
        threshold_prob = _threshold_probability_estimation(t, R_without_i)
        sample_prob = _sample_probability_estimation(V_i, Z_floor, Z_ceiling, t, split_feature)
        L.append(threshold_prob*sample_prob)
    return L

def _voters_threshold_likelihoods_estimation(X, X_train, y_train, S, split_feature):
    V_train = pd.concat([X_train.loc[X_train['Scenario'] == S & X_train['Is_Random'] == False & X_train["VoterType"] != "TRT",], y_train], axis=1, join='inner')
    R = _most_likely_threshold_ranges(X_train, y_train, split_feature)
    voters = pd.DataFrame(X[["VoterID", "SessionIDX"]].drop_duplicates())
    for voter in voters:
        V_i = V_train.loc[V_train["VoterID"] == voter.VoterID]
        R_without_i = R[[r.voter.VoterID != voter.VoterID for r in R]]
        R_accuracy = _total_ranges_accuracy(R_without_i)
        Z_floor = (R_accuracy.errors_below)/(R_accuracy.errors_below + R_accuracy.correct_below)
        Z_ceiling = (R_accuracy.errors_above)/(R_accuracy.errors_above + R_accuracy.correct_above)
        numVotes = _extract_num_votes(V_i)
        L = _threshold_likelihood_estimation(V_i, R_without_i, Z_floor, Z_ceiling, numVotes, split_feature)
        X = pd.concat([X.loc[X["VoterID"] == voter.VoterID], L, Z_floor, Z_ceiling], axis=1, join='inner')

    return X


def _voters_action_likelihoods_estimation(X,S, split_feature):
    V = _voters_threshold_likelihoods_estimation(X, S, split_feature)
    V_s = V.loc[V.Scenario == S]

    for v in V_s:
        below, above = _determine_split_preferences(v, split_feature)
        L_prob_below = 0
        L_prob_above = 0
        Z_floor = v.Z_floor
        Z_celing = v.Z_celing
        for t in range(0, _get_max_threshold(v.NumVotes)):
            pass


class ThresholdRange():
    def __init__(self,
                 voter=None, range=None, numVotes=None):
        self.voter = voter
        self.range = range
        self.numVotes = numVotes
        self.errors_below = 0
        self.errors_above = 0
        self.correct_below = 0
        self.correct_above = 0



class DecisionTreeBaseline():
    """Base class for ordinal meta-classifier.

    """

    def __init__(self):
        pass
    
    def fit(self, X, y, sample_weight=None):
        return self

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        return self

    def predict(self, X):
        #X = check_array(X, accept_sparse=['csr', 'csc'])

        # ---------------------------------------------Our CODE
        n_samples = X.shape[0]
        prediction = np.zeros((n_samples, 1))
        
        for i in range(0, n_samples):
            if X.iloc[i].Scenario == 3:
                if X.iloc[i].VoterType == "LB":
                    prediction[i] = 2 #Q' vote
                else:
                    prediction[i] = 1 #Q vote  
            else:
                if X.iloc[i].Scenario in [5,6]:
                    if X.iloc[i].VoterType == "TRT":
                        prediction[i] = 1 #Q vote   
                    else:
                        prediction[i] = 2 #Q' vote
                        
                else:
                    prediction[i] = 1 #Q vote


        return prediction

class BayesRuleClassifier(DecisionTreeBaseline):
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
            if X[i, "Scenario"] == "C":
                if X[i, "VoterType"] == "LB":
                    prediction[i] = 2  # Q' vote
                else:
                    prediction[i] = 1  # Q vote
            else:
                if X[i, "Scenario"] in ["E", "F"]:
                    if X[i, "VoterType"] == "TRT":
                        prediction[i] = 1  # Q vote
                    else:
                        prediction[i] = 2  # Q' vote

                else:
                    prediction[i] = 1  # Q vote

        return prediction

class LHClassifier(DecisionTreeBaseline):
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
            if X[i, "Scenario"] == "C":
                if X[i, "VoterType"] == "LB":
                    prediction[i] = 2  # Q' vote
                else:
                    prediction[i] = 1  # Q vote
            else:
                if X[i, "Scenario"] in ["E", "F"]:
                    if X[i, "VoterType"] == "TRT":
                        prediction[i] = 1  # Q vote
                    else:
                        prediction[i] = 2  # Q' vote

                else:
                    prediction[i] = 1  # Q vote

        return prediction

class MLHClassifier(DecisionTreeBaseline):
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
        prediction = super().predict(X) #Baseline prediciton

        #TODO: compelete this
        _voters_action_likelihoods_estimation(X, "C", "GAP12_poll")

        return prediction

class ScenarioClassifier(DecisionTreeBaseline):
    """Base class for ordinal meta-classifier.

    """

    def __init__(self, clf):
        self.clf = clf
        pass

    def _set_columns(self, columns):
        id_features = [columns.get_loc("VoterID")]
        gap_features = [columns.get_loc("GAP12_pref_poll"), columns.get_loc("GAP13_pref_poll")]
        scenario_features = [columns.get_loc("Scenario")]

        self.id_features = id_features
        self.gap_features = gap_features
        self.scenario_features = scenario_features
        pass

    def fit(self, X, y, sample_weight=None):
        return self._fit(X,y)

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        #data_df.iloc[:, data_df.columns != "Action"]
        #data_df["Action"]
        # X_df = osp._prepare_dataset(X)
        y = np.asarray(y)
        id_features = self.id_features
        gap_features = self.gap_features
        scenario_feature = self.scenario_features[0]

        index_135 = [(X[i,scenario_feature] in [1,3,5,6]) & (y[i] in [1,2]) for i in range(0, len(X))]
        X_135_train, y_135_train = X[index_135][:,id_features + [gap_features[0]]], y[index_135]

        index_24 = [(X[i,scenario_feature] in [2,4,6]) & (y[i] in [1,3]) for i in range(0, len(X))]
        X_24_train, y_24_train = X[index_24][:,id_features + [gap_features[1]]], y[index_24]

        index_6_1 = [(X[i,scenario_feature] in [4,6]) & (y[i] in [1,2,3]) for i in range(0, len(X))]
        y_6_1_train = y[index_6_1]
        y_6_1_train[[y_6_1_train[i] in [1,2] for i in range(0, len(y_6_1_train))]] = 2
        y_6_1_train[[y_6_1_train[i] in [3] for i in range(0, len(y_6_1_train))]] = 1
        X_6_1_train = X[index_6_1][:, id_features + [gap_features[1]]]

        index_6_2 = [(X[i,scenario_feature] in [1,3,5,6]) & (y[i] in [1,2]) for i in range(0, len(X))]
        X_6_2_train, y_6_2_train = X[index_6_2][:, id_features + [gap_features[0]]], y[index_6_2]

        X_train, y_train = X[:,id_features + gap_features], y[:]

        self.discretizer_135 = mdlp.MDLP_Discretizer(features=[1])
        self.discretizer_24 = mdlp.MDLP_Discretizer(features=[1])
        self.discretizer_6_1 = mdlp.MDLP_Discretizer(features=[1])
        self.discretizer_6_2 = mdlp.MDLP_Discretizer(features=[1])

        # X_135_train, self.le = np.asarray(osp._prepare_dataset(X_135_train, self.le))
        # X_24_train, self.le = np.asarray(osp._prepare_dataset(X_24_train, self.le))
        # X_6_1_train, self.le = np.asarray(osp._prepare_dataset(X_6_1_train, self.le))
        # X_6_2_train, self.le = np.asarray(osp._prepare_dataset(X_6_2_train, self.le))
        # X_train, self.le = np.asarray(osp._prepare_dataset(X_train, self.le))

        self.discretizer_135.fit(X_135_train, y_135_train)
        self.discretizer_24.fit(X_24_train, y_24_train)
        self.discretizer_6_1.fit(X_6_1_train, y_6_1_train)
        self.discretizer_6_2.fit(X_6_2_train, y_6_2_train)

        X_135_discretized = X_135_train#self.discretizer_135.transform(X_135_train)
        X_24_discretized = X_24_train#self.discretizer_24.transform(X_24_train)
        X_6_1_discretized =X_6_1_train# self.discretizer_6_1.transform(X_6_1_train)
        X_6_2_discretized = X_6_2_train#self.discretizer_6_2.transform(X_6_2_train)

        self.clf_135 = deepcopy(self.clf)
        self.clf_24 = deepcopy(self.clf)
        self.clf_6_1 = deepcopy(self.clf)
        self.clf_6_2 = deepcopy(self.clf)
 #       self.general_clf = clf()#GradientBoostingClassifier(n_estimators=300)

        self.clf_135.fit(X_135_discretized, y_135_train)
        self.clf_24.fit(X_24_discretized, y_24_train)
        self.clf_6_1.fit(X_6_1_discretized, y_6_1_train)
        self.clf_6_2.fit(X_6_2_discretized, y_6_2_train)
#       self.general_clf.fit(X_train, y_train)

        # apply same discretization to test set
        #X_test_discretized = discretizer.transform(X_test)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        prediction = np.zeros((n_samples, 1))
        id_features = self.id_features
        gap_features = self.gap_features
        scenario_feature = self.scenario_features[0]

        for i in range(0, n_samples):
            if X[i,scenario_feature] in [1, 3 ,5]:
                x_i = np.asarray([X[i, id_features + [gap_features[0]]]])
                prediction[i] = int(self.clf_135.predict(x_i)[0]) # self.discretizer_135.transform(x_i)
            else:
                if X[i,scenario_feature] in [2 ,4]:
                        x_i = np.asarray([X[i, id_features + [gap_features[1]]]])
                        prediction[i] = int(self.clf_24.predict(x_i)[0]) #self.discretizer_24.transform(x_i)
                else: #6
                    x_i  = np.asarray([X[i, id_features + [gap_features[1]]]])
                    first_predict = int(self.clf_6_1.predict(x_i)[0]) #self.discretizer_6_1.transform(x_i)
                    if first_predict == 1: #Leader=1 , not leader=2
                        prediction[i] = 3
                    else:
                        x_i = np.asarray([X[i, id_features + [gap_features[0]]]])
                        prediction[i] = int(self.clf_6_2.predict(x_i)[0]) #self.discretizer_6_2.transform(x_i)


        return prediction#self.general_clf.predict(X[:, id_features + gap_features])