# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:16:03 2018

@author: Adam
"""
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

def _convert_prediction(X):
    X.loc[X['Prediction']==1,"VotePrediction"] = X.loc[X['Prediction']==1,"Pref1"]
    X.loc[X['Prediction']==2,"VotePrediction"] = X.loc[X['Prediction']==2,"Pref2"]
    X.loc[X['Prediction']==3,"VotePrediction"] = X.loc[X['Prediction']==3,"Pref3"]

    return X

def _generate_action_name(X):
    # Generate action name
    # Action mapping table
    d = {'scenario': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'E', 'E', 'E', 'F', 'F', 'F'],
         'action': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
         'action_name': ['TRT', 'DOM', 'DOM', 'TRT', 'DOM', 'DOM', 'TRT', 'WLB', 'DOM', 'TRT', 'DOM', 'SLB', 'TRT',
                         'CMP\WLB', 'DOM', 'TRT', 'CMP', 'SLB']}
    action_map_df = pd.DataFrame(data=d)

    X['Action_name'] = [(action_map_df.loc[(action_map_df.scenario == str(x[1]['Scenario'])) & (
        action_map_df.action == int(X.loc[x[0], 'Action'])), 'action_name']).values[0] for x in
                        X.iterrows()]

    return X

def _data_cleaning(data_df,is_target ,le):
    if is_target:
        data_df = data_df.astype("category")
        data_df = le.fit_transform(data_df)
    else:
        for c in data_df.columns:
            if data_df[c].dtype in (object, str, np.object, bool):
                if not (data_df[c].dtype in (int, float)):
                    data_df[c] = le.fit_transform(data_df[c])
    return data_df

def _generate_A_ratios(X, X_train, y_train, voter):
    """Generate A ratios - That is TRT-ratio, CMP-ratio, WLB-ratio, SLB-ratio, DOM-ratio
        Action is in {TRT,DLB,SLB,WLB,CMP,DOM}
        Scenario is in {A,B,C,D,E,F}
    """
    start_ind_col = len(X.columns)

    availability_counter = np.count_nonzero([x[1].Scenario in ['A','B','C','D','E','F'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()])
    X.loc[X['VoterID'] == voter.VoterID, 'TRT-ratio'] = np.count_nonzero((['TRT' in x[1]['Action_name'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()]))/availability_counter if availability_counter>0 else 0

    availability_counter = np.count_nonzero([x[1].Scenario in ['C','E'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()])
    X.loc[X['VoterID'] == voter.VoterID, 'WLB-ratio'] =  np.count_nonzero((['WLB' in x[1]['Action_name'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()]))/availability_counter if availability_counter>0 else 0

    availability_counter = np.count_nonzero([x[1].Scenario in ['D','F'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()])
    X.loc[X['VoterID'] == voter.VoterID, 'SLB-ratio'] =  np.count_nonzero((['SLB' in x[1]['Action_name'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()]))/availability_counter if availability_counter>0 else 0
    #X.loc[X['VoterID'] == voter.VoterID, 'LB-ratio'] =  np.count_nonzero((['LB' in x[1]['Action_name'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()]))/np.count_nonzero([x[1].Scenario in ['C','D','E','F'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()])
    availability_counter = np.count_nonzero([x[1].Scenario in ['E','F'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()])
    X.loc[X['VoterID'] == voter.VoterID, 'CMP-ratio'] =  np.count_nonzero((['CMP' in x[1]['Action_name'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()]))/availability_counter if availability_counter>0 else 0

    availability_counter = np.count_nonzero([x[1].Scenario in ['A','B','C','D','E','F'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()])
    X.loc[X['VoterID'] == voter.VoterID, 'DOM-ratio'] =  np.count_nonzero((['DOM' in x[1]['Action_name'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()]))/availability_counter if availability_counter>0 else 0

    X.loc[X['VoterID'] == voter.VoterID, 'DOM-counter'] =  np.count_nonzero((['DOM' in x[1]['Action_name'] for x in X_train.loc[X_train['VoterID'] == voter.VoterID].iterrows()]))

    X['TRT-ratio'] = (X['TRT-ratio']).astype(float)
    X['WLB-ratio'] = (X['WLB-ratio']).astype(float)
    X['SLB-ratio'] = (X['SLB-ratio']).astype(float)
    X['CMP-ratio'] = (X['CMP-ratio']).astype(float)
    X['DOM-ratio'] = (X['DOM-ratio']).astype(float)
    X['DOM-counter'] = (X['DOM-counter']).astype(float)

    end_ind_col = len(X.columns)
    return X, list(range(start_ind_col, end_ind_col))

def _generate_is_random_voter(X):
    """Identify random voters using the rule of DOM-counter >= 2 (excluding SLB actions)"""
    X['Is_Random'] = [x >= 2 for x in X['DOM-counter']]

    return X

def _generate_voter_type(X):
    """Generate Voter Type using thresholds over the A-ratio values"""
    X['VoterType'] = 'Other'
    #X.loc[ [int(x[1]['CMP-ratio'])>=0.7 for x in X.iterrows()], 'VoterType'] = 'CMP'
    X.loc[ [float(x[1]['WLB-ratio'])>0.8 for x in X.iterrows()], 'VoterType'] = 'LB'
    X.loc[ [float(x[1]['TRT-ratio'])>0.9 for x in X.iterrows()], 'VoterType'] = 'TRT'

    return X

def _generate_pref_gaps(X):
   X["VotesPref1PreVote"] = [x[1]["VotesCand"+str(x[1]["Pref1"])+"PreVote"] for x in X.iterrows()]
   X["VotesPref2PreVote"] = [x[1]["VotesCand"+str(x[1]["Pref2"])+"PreVote"] for x in X.iterrows()]
   X["VotesPref3PreVote"] = [x[1]["VotesCand"+str(x[1]["Pref3"])+"PreVote"] for x in X.iterrows()]

   return X

def _generate_gaps(X):
    """Generate Gaps features"""

    X['VotesLeader_poll'] = X[['VotesCand1PreVote','VotesCand2PreVote','VotesCand3PreVote']].max(axis = 1)
    X['VotesRunnerup_poll'] = X[['VotesCand1PreVote','VotesCand2PreVote','VotesCand3PreVote']].apply(np.median, axis=1)
    X['VotesLoser_poll'] = X[['VotesCand1PreVote','VotesCand2PreVote','VotesCand3PreVote']].min(axis = 1)


    X['GAP12_poll'] = X['VotesLeader_poll'] - X['VotesRunnerup_poll']
    X['GAP23_poll'] =  X['VotesRunnerup_poll'] - X['VotesLoser_poll']
    X['GAP13_poll'] = X['VotesLeader_poll'] - X['VotesLoser_poll']

    #Preference based gaps - I think more suitable for ML for it's more synchronized across the scenarios

    # for x in X.iterrows():
    #     vote = x[1]
    #     candPref1 = 'VotesCand' + str(vote.Pref1) + 'PreVote'
    #     candPref2 = 'VotesCand' + str(vote.Pref2) + 'PreVote'
    #     candPref3 = 'VotesCand' + str(vote.Pref3) + 'PreVote'
    #     X.loc[x[0], 'VotesPref1_poll'] = X.loc[x[0], candPref1]
    #     X.loc[x[0], 'VotesPref2_poll'] = X.loc[x[0], candPref2]
    #     X.loc[x[0], 'VotesPref3_poll'] = X.loc[x[0], candPref3]
    #
    #
    X['GAP12_pref_poll'] = X['VotesPref1PreVote'] - X['VotesPref2PreVote']
    X['GAP23_pref_poll'] = X['VotesPref2PreVote'] - X['VotesPref3PreVote']
    X['GAP13_pref_poll'] = X['VotesPref1PreVote'] - X['VotesPref3PreVote']


    return X

def _generate_scenario_type(X):
    #initialize
    X['TRT'] = 0
    X['WLB'] = 0
    X['SLB'] = 0
    X['CMP'] = 0

    X.loc[[x[1].Scenario in ['A','B','C','D','E','F'] for x in X],'TRT'] = 1
    X.loc[[x[1].Scenario in ['C','E'] for x in X],'WLB'] = 1
    X.loc[[x[1].Scenario in ['D','F'] for x in X],'SLB'] = 1
    X.loc[[x[1].Scenario in ['E','F'] for x in X],'CMP'] = 1

    return X

def _generate_feature_aggregation_class_dependant(X, X_train, y_train, voter, feature_name, aggregation_func):
    action1_list = [ float(x[1][feature_name]) for x in X_train.loc[(X_train['VoterID'] == voter.VoterID) & (y_train == 1)].iterrows()]
    if len(action1_list)>0:
        X.loc[X['VoterID'] == voter.VoterID, feature_name + '_action1_' + aggregation_func.__name__] =  aggregation_func(action1_list)

    action2_list = [ float(x[1][feature_name]) for x in X_train.loc[(X_train['VoterID'] == voter.VoterID) & (y_train == 2)].iterrows()]
    if len(action2_list)>0:
        X.loc[X['VoterID'] == voter.VoterID, feature_name + '_action2_' + aggregation_func.__name__] =  aggregation_func(action2_list)

    action3_list = [ float(x[1][feature_name]) for x in X_train.loc[(X_train['VoterID'] == voter.VoterID) & (y_train == 3)].iterrows()]
    if len(action3_list)>0:
        X.loc[X['VoterID'] == voter.VoterID, feature_name + '_action3_' + aggregation_func.__name__] =  aggregation_func(action3_list)


    return X

def _generate_action_aggregation_features(X, X_train, y_train, voter):
      aggregators = [np.average, np.std, np.median]
      feature_name = "Action"

      for aggregation_func in aggregators:
          X.loc[X['VoterID'] == voter.VoterID, feature_name + "_" + aggregation_func.__name__] =  aggregation_func([float(y_train[x[0]]) for x in X_train.loc[(X_train['VoterID'] == voter.VoterID) & (X_train['Scenario'].isin(['C','D','E','F']))].iterrows()])

      return X


def _generate_gaps_features(X, X_train, y_train, voter):
    start_ind_col = len(X.columns)

    feature12 = 'GAP12_pref_poll'
    feature23 = 'GAP23_pref_poll'
    feature13 = 'GAP13_pref_poll'

    features = [feature12, feature23, feature13]
    aggregators = [np.average, np.std,  np.median, np.min, np.max, skew, kurtosis]
    scenarios = ['C','D','E','F']

    for aggregator in aggregators:
        for feature in features:
            X = _generate_feature_aggregation_class_dependant(X, X_train.loc[X_train['Scenario'].isin(scenarios)], y_train, voter, feature, aggregator)


    end_ind_col = len(X.columns)

    return X, list(range(start_ind_col, end_ind_col))

def _generate_gap_dif_features(X):

    start_ind_col = len(X.columns)

    feature12 = 'GAP12_pref_poll'
    feature23 = 'GAP23_pref_poll'
    feature13 = 'GAP13_pref_poll'


    X[feature12 + '_action1_' + np.average.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action1_' + np.average.__name__ ]
    X[feature12 + '_action2_' + np.average.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action2_' + np.average.__name__ ]
    X[feature12 + '_action3_' + np.average.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action3_' + np.average.__name__ ]

    X[feature23 + '_action1_' + np.average.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action1_' + np.average.__name__ ]
    X[feature23 + '_action2_' + np.average.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action2_' + np.average.__name__ ]
    X[feature23 + '_action3_' + np.average.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action3_' + np.average.__name__ ]

    X[feature13 + '_action1_' + np.average.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action1_' + np.average.__name__ ]
    X[feature13 + '_action2_' + np.average.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action2_' + np.average.__name__ ]
    X[feature13 + '_action3_' + np.average.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action3_' + np.average.__name__ ]

    X[feature12 + '_action1_' + np.median.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action1_' + np.median.__name__ ]
    X[feature12 + '_action2_' + np.median.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action2_' + np.median.__name__ ]
    X[feature12 + '_action3_' + np.median.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action3_' + np.median.__name__ ]

    X[feature23 + '_action1_' + np.median.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action1_' + np.median.__name__ ]
    X[feature23 + '_action2_' + np.median.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action2_' + np.median.__name__ ]
    X[feature23 + '_action3_' + np.median.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action3_' + np.median.__name__ ]

    X[feature13 + '_action1_' + np.median.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action1_' + np.median.__name__ ]
    X[feature13 + '_action2_' + np.median.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action2_' + np.median.__name__ ]
    X[feature13 + '_action3_' + np.median.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action3_' + np.median.__name__ ]

    X[feature12 + '_action1_' + np.max.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action1_' + np.max.__name__ ]
    X[feature12 + '_action2_' + np.max.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action2_' + np.max.__name__ ]
    X[feature12 + '_action3_' + np.max.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action3_' + np.max.__name__ ]

    X[feature23 + '_action1_' + np.max.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action1_' + np.max.__name__ ]
    X[feature23 + '_action2_' + np.max.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action2_' + np.max.__name__ ]
    X[feature23 + '_action3_' + np.max.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action3_' + np.max.__name__ ]

    X[feature13 + '_action1_' + np.max.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action1_' + np.max.__name__ ]
    X[feature13 + '_action2_' + np.max.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action2_' + np.max.__name__ ]
    X[feature13 + '_action3_' + np.max.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action3_' + np.max.__name__ ]

    X[feature12 + '_action1_' + np.min.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action1_' + np.min.__name__ ]
    X[feature12 + '_action2_' + np.min.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action2_' + np.min.__name__ ]
    X[feature12 + '_action3_' + np.min.__name__ + '_dif'] = X[feature12] - X[feature12 + '_action3_' + np.min.__name__ ]

    X[feature23 + '_action1_' + np.min.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action1_' + np.min.__name__ ]
    X[feature23 + '_action2_' + np.min.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action2_' + np.min.__name__ ]
    X[feature23 + '_action3_' + np.min.__name__ + '_dif'] = X[feature23] - X[feature23 + '_action3_' + np.min.__name__ ]

    X[feature13 + '_action1_' + np.min.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action1_' + np.min.__name__ ]
    X[feature13 + '_action2_' + np.min.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action2_' + np.min.__name__ ]
    X[feature13 + '_action3_' + np.min.__name__ + '_dif'] = X[feature13] - X[feature13 + '_action3_' + np.min.__name__ ]


    end_ind_col = len(X.columns)

    return X, list(range(start_ind_col, end_ind_col))

def _static_feature_generation(X):
    X = _generate_pref_gaps(X)
    X = _generate_gaps(X)
    #X = _generate_scenario_type(X)

    return X

def _dynamic_feature_generation(X, X_train, y_train):
    all_voters = pd.DataFrame(X[["VoterID", "SessionIDX"]].drop_duplicates())
    for voter in all_voters.iterrows():
        X, added_columns = _generate_A_ratios(X, X_train, y_train, voter[1])
        if len(added_columns)>0:
            a_ratio_columns = added_columns
        X, added_columns = _generate_gaps_features(X, X_train, y_train, voter[1])
        if len(added_columns)>0:
            gaps_columns = added_columns
        X = _generate_action_aggregation_features(X, X_train, y_train, voter[1])

    #Gaps features encoding
    X = X.fillna(X.mean()) #X.fillna(1000) #fill na with some high value (maybe maximum) because the voters with na values didn't choose the action (say q'', 3) in all gaps they incounterd.
    X, gaps_dif_columns = _generate_gap_dif_features(X)
    total_gaps_columns = a_ratio_columns + gaps_columns + gaps_dif_columns
    total_gaps_columns.append(X.columns.get_loc("GAP12_pref_poll"))
    total_gaps_columns.append(X.columns.get_loc("GAP23_pref_poll"))
    total_gaps_columns.append(X.columns.get_loc("GAP13_pref_poll"))

    normalized_gap_fs = pd.DataFrame(preprocessing.normalize(X.iloc[:,total_gaps_columns]))
    encoded_gap_fs = pd.DataFrame(_autoencode(normalized_gap_fs))

    X = pd.concat([X, encoded_gap_fs], axis=1, join='inner')

    X = X.drop(X.columns[gaps_columns + gaps_dif_columns], axis=1)

    X = _generate_is_random_voter(X)
    X = _generate_voter_type(X)



    return X

def _evaluation(X, clf, target):
    #tests

    #static features generation
    X = _static_feature_generation(X)
    # Encoders definitions
    le = sklearn.preprocessing.LabelEncoder()
    target_le = sklearn.preprocessing.LabelEncoder()

    # Split into features and target
    features_df, target_df = X.drop([target], axis=1),X[target]


    n_folds = 10
    results_df = pd.DataFrame(columns=['Measure', 'Result'])
    # Initialize metrics:
    results_df.loc[0] = ['PRECISION', 0]
    results_df.loc[1] = ['RECALL', 0]
    results_df.loc[2] = ['F_MEASURE', 0]


    # 10 fold cross validation
    kf = KFold(n_folds, shuffle=True, random_state=1)  # 10 fold cross validation
    for train_indices, test_indices in kf.split(features_df, target_df):
        # Feature Generation
        features_train = features_df.loc[[ii for ii in train_indices],]
        targets_train = target_df[[ii for ii in train_indices]]
        features_ext_df = _dynamic_feature_generation(features_df, features_train, targets_train)
        features_ext_df = features_ext_df.drop(["Action_name"], axis=1)

        # encoding the dataframes
        features_encoded_df = pd.DataFrame(preprocessing.normalize(preprocessing.scale(_data_cleaning(features_ext_df, False, le))))
        target_encoded_df = _data_cleaning(target_df, True, target_le)


        # make training and testing datasets
        features_train = features_encoded_df.loc[[ii for ii in train_indices],]
        features_test = features_encoded_df.loc[[ii for ii in test_indices],]
        targets_train = target_encoded_df[[ii for ii in train_indices]]
        targets_test = target_encoded_df[[ii for ii in test_indices]]

        # Train
        clf.fit(X = features_train.as_matrix(), y = targets_train)
        # Test
        predicated = clf.predict(features_test.as_matrix())

        # Measures
        results_df.iloc[0, 1] = results_df.iloc[0, 1] + precision_score(targets_test, predicated, average='weighted')
        results_df.iloc[1, 1] = results_df.iloc[1, 1] + recall_score(targets_test, predicated, average='weighted')
        results_df.iloc[2, 1] = results_df.iloc[2, 1] + f1_score(targets_test, predicated, average='weighted')

    results_df.Result = results_df.Result.apply(lambda x: x / n_folds)
    return results_df

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def _read_roy_folds(folds_file):
    lines = folds_file.read().split('\n')
    folds = list()
    for index in range(0,len(lines)):
        line = lines[index]
        if not ('fold' in line) and line != '':
            folds.append([int(ii) for ii in line[1:len(line)-1].split(',')])


    return folds

def _get_loo_folds(X):
    folds = list()
    for x in X.iterrows():
        fold = [x[0]]
        folds.append(fold)

    return folds


def _autoencode(features):
    #test

    encoding_dim = int(len(features.columns)/5)
    input_votes = Input(shape = (len(features.columns),))
    encoded = Dense(encoding_dim, activation='relu')(input_votes)
    decoded = Dense(len(features.columns), activation='tanh')(encoded)
    autoencoder = Model(input_votes, decoded)
    encoder = Model(input_votes, encoded)

 #   encoded_input = Input(shape=(encoding_dim,))
#    decoder_layer = autoencoder.layers[-1]
 #   decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='MSE')

    autoencoder.fit(features, features,
                epochs=150,
                batch_size=256,
                shuffle=True,verbose=False)

    encoded_votes = encoder.predict(features)

    return encoded_votes


def _evaluation_roy_splits(raw_data, clfs, target, folds, scenario_filter):
    data = raw_data.copy()
    #static features generation
    data = _static_feature_generation(data)
    # Encoders definitions
    le = sklearn.preprocessing.LabelEncoder()
    target_le = sklearn.preprocessing.LabelEncoder()


    n_folds = 10


    results_df = pd.DataFrame(columns=['Classifier', 'PRECISION', 'RECALL','F_MEASURE','ACCURACY'])

    prediction = pd.DataFrame(np.matrix([]))
    feature_importances = pd.DataFrame()
    features_train = pd.DataFrame()
    # 10 fold cross validation
    for i in range(0,len(folds)):
        # Split into features and target
        features_df, target_df = data.drop([target], axis=1),data[target]

        test_indices = data.index[[x[1].RoundIndex in folds[i] for x in data.iterrows()]].tolist()
        train_indices =  data.index[[not (x[1].RoundIndex in folds[i] or x[1].Scenario == scenario_filter) for x in data.iterrows()]].tolist()
         # Feature Generation
        features_train = features_df.loc[[ii for ii in train_indices],]
        targets_train = target_df[[ii for ii in train_indices]]
        features_ext_df = _dynamic_feature_generation(features_df, features_train , targets_train)
        features_ext_df = features_ext_df.drop(["Action_name"], axis=1)
        features_ext_df = features_ext_df.drop(["Vote"], axis=1)
        # encoding the dataframes
        features_encoded_df = pd.DataFrame(preprocessing.normalize(preprocessing.scale(_data_cleaning(features_ext_df, False, le).as_matrix()), axis=0, norm='max'))#
        target_encoded_df = target_df #_data_cleaning(target_df, True, target_le)


        # make training and testing datasets
        features_train = features_encoded_df.loc[[ii for ii in train_indices],]
        features_test = features_encoded_df.loc[[ii for ii in test_indices],]
        targets_train = target_encoded_df[[ii for ii in train_indices]]
        targets_test = target_encoded_df[[ii for ii in test_indices]]


#        training = pd.concat([features_train, pd.DataFrame(targets_train)], axis=1, join='inner')
#        testing = pd.concat([features_test, pd.DataFrame(targets_test)], axis=1, join='inner')
#        training.to_csv("training_fold_"+str(i)+".csv")
#        testing.to_csv("testing_fold_"+str(i)+".csv")

        for j in range(0,len(clfs)):
            clf = clfs[j]
            clf_name = str(clf).split("(")[0]
            if i == 0:
                #Initialize metrics
                results_df.loc[j] = [str(clf), 0, 0, 0, 0]

            if clf != "baseline":

                # Train
                clf.fit(X = features_train.as_matrix(), y = targets_train)
                # Test
                predicated = clf.predict(features_test.as_matrix())

                # #feature importance
                # current_feature_importances = pd.DataFrame(clf.feature_importances_,
                #                                            index=features_ext_df.columns,
                #                                            columns=['importance']).sort_values('importance',
                #                                                                                ascending=False)
                # if len(feature_importances) == 0:
                #     feature_importances = current_feature_importances
                # else:
                #     feature_importances['importance'] = feature_importances['importance'] + current_feature_importances['importance']
                #
                # print(feature_importances)

            else:
                features_ext_df_test = features_ext_df.loc[[ii for ii in test_indices],]
                n_samples = len(features_ext_df_test)
                predicated = np.zeros((n_samples),dtype=int)

                features_ext_df_test["Prediction"] = 1
                features_ext_df_test.loc[(features_ext_df_test["Scenario"] == "C") & (features_ext_df_test["VoterType"] == "LB"), "Prediction"] = 2
                features_ext_df_test.loc[(features_ext_df_test['Scenario'].isin(["E","F"])) & (features_ext_df_test["VoterType"] != "TRT"), "Prediction"] = 2

                predicated = features_ext_df_test["Prediction"]

                features_ext_df_test = pd.concat([features_ext_df_test, pd.DataFrame(predicated)], axis=1)




            #aggregate results
            if len(prediction) == 0:
                prediction =  pd.DataFrame(predicated)
            else:
                prediction = pd.concat([prediction, pd.DataFrame(predicated)])

            raw_data.loc[[ii for ii in test_indices],"Prediction" + "_" + clf_name] =  predicated

            print(str(clf) +": F_score = " + str(f1_score(targets_test, predicated, average='weighted')))
            # Measures
            results_df.iloc[j, 1] = results_df.iloc[j, 1] + precision_score(targets_test, predicated, average='weighted')
            results_df.iloc[j, 2] = results_df.iloc[j, 2] + recall_score(targets_test, predicated, average='weighted')
            results_df.iloc[j, 3] = results_df.iloc[j, 3] + f1_score(targets_test, predicated, average='weighted')
            results_df.iloc[j, 4] = results_df.iloc[j, 4] + accuracy_score(targets_test, predicated)

    results_df.Result = results_df.Result.apply(lambda x: x / n_folds)
    return results_df, prediction, raw_data, feature_importances, features_train


def _build_data_by_folds(data, folds):
    transformed_data = pd.DataFrame()
    for i in range(0,len(folds)):
        # Split into features and target
        fold_indices = data.index[[x[1].RoundIndex in folds[i] for x in data.iterrows()]].tolist()
        fold_df = data.iloc[fold_indices,:]

        if len(transformed_data) == 0:
            transformed_data = fold_df
        else:
            transformed_data = pd.concat([transformed_data, fold_df])
    return transformed_data

#
# data = pd.read_excel("datasets/oneshot/full_data_no_timeout_no_first_round.xlsx")
#
# data_with_action_name = _generate_action_name(data)
# data_with_action_name.to_csv("datasets\\oneshot\\full_data_no_timeout_no_first_round_with_action_name.csv")
#

# #Load rawdata
# oneshot_df = pd.read_csv("datasets/oneshot/full_data_no_timeout_no_first_round_with_action_name.csv")#read_excel("datasets/oneshot/NewData.xlsx")
#
# #sorted_data = _build_data_by_folds(oneshot_df)
# #sorted_data.to_csv("Results\\sorted_fulldata.csv")
# #
# #print("F_score = " + str(f1_score(sorted_data.Action, nn_prediction[1:], average='weighted')))
# #
#
# raw_data = oneshot_df
#
#
#
# raw_data = raw_data.drop(['GameConfiguration','FileName','PlayerID','DateTime','CandName1', 'CandName2','CandName3','WinnerPostVote', 'VotesCand1', 'VotesCand2', 'VotesCan3', 'PointsPostVote','ResponseTime'], axis=1)


classifier =  RandomForestClassifier(n_estimators=100)#MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)

#
# #Model Phase
# #rf_results_df = _evaluation_roy_splits(data, RandomForestClassifier(n_estimators=100), 'Action')
#
#
# nn_results_df, nn_prediction, data_with_pred = _evaluation_roy_splits(raw_data, classifier, 'Action')
#
# nn_results_df.to_csv("Results\\rf_new_performance_df.csv")
# nn_prediction.to_csv("Results\\rf_new_prediction.csv")
# data_with_pred.to_csv("Results\\rf_new_data_with_pred.csv")
#

#baseline_results_df, baseline_prediction, baseline_full_pred = _evaluation_roy_splits(data, "baseline", 'Action')
#
#
#baseline_results_df.to_csv("Results\\baseline_performance_df.csv")
#baseline_prediction.to_csv("Results\\baseline_prediction.csv")
#baseline_full_pred.to_csv("Results\\baseline_full_pred.csv")


#
# # #other data sets
# data = pd.read_excel("datasets/oneshot/tal.xlsx")
# data_with_action_name = _generate_action_name(data)
# data_with_action_name.to_csv("datasets\\oneshot\\tal_no_timeout_with_action_name.csv")
# folds = _read_roy_folds(open("datasets/oneshot/tal_folds.txt", "r"))
# #
#
# oneshot_df = pd.read_csv("datasets/oneshot/tal_no_timeout_with_action_name.csv")#read_excel("datasets/oneshot/NewData.xlsx")
# oneshot_df = oneshot_df.fillna(oneshot_df.mean())
#
#
# raw_data = oneshot_df
#
# raw_data = raw_data.drop(['DateTime','Util4','LabExperiment','ExperimentComments','CandName1', 'CandName2','CandName3','CanName4','GameID', 'PrefsObserved','Pref4', 'VotesCand1','ResponseTime', 'VotesCand2', 'VotesCand3'], axis=1)
#
# #Model Phase
# #rf_results_df = _evaluation_roy_splits(data, RandomForestClassifier(n_estimators=100), 'Action')
#
#
# tal_nn_performance_df, tal_nn_prediction, tal_nn_pred = _evaluation_roy_splits(raw_data, classifier, 'Action', folds)
#
# tal_nn_performance_df.to_csv("Results\\tal_rf_performance_df.csv")
# tal_nn_prediction.to_csv("Results\\tal_rf_prediction.csv")
# tal_nn_pred.to_csv("Results\\tal_rf_pred.csv")
#
#
# print("F_score = " + str(f1_score(tal_nn_pred.Action, tal_nn_pred.nn_predction.astype(int), average='weighted')))
#
# #scharm
# scharm_data = pd.read_excel("datasets/oneshot/schram.xlsx")
#
# data_with_action_name = _generate_action_name(scharm_data)
# data_with_action_name.to_csv("datasets\\oneshot\\scharm_data_with_action_name.csv")
# #other data sets
# scharm_df = pd.read_csv("datasets/oneshot/scharm_data_with_action_name.csv")#read_excel("datasets/oneshot/NewData.xlsx")
# scharm_df = scharm_df.fillna(scharm_df.mean())
# folds = _read_roy_folds(open("datasets/oneshot/schram_folds.txt", "r"))
#
# raw_data = scharm_df
#
# #raw_data = raw_data.drop(['DateTime','Util4','LabExperiment','ExperimentComments','CandName1', 'CandName2','CandName3','CanName4','GameID', 'PrefsObserved','Pref4', 'VotesCand1','ResponseTime', 'VotesCand2', 'VotesCand3'], axis=1)
#
# #Model Phase
# #rf_results_df = _evaluation_roy_splits(data, RandomForestClassifier(n_estimators=100), 'Action')
#
#
# scharm_nn_performance_df, scharm_nn_prediction, scharm_nn_pred = _evaluation_roy_splits(raw_data, classifier, 'Action', folds)
#
# scharm_nn_performance_df.to_csv("Results\\scharm_rf_performance_df.csv")
# scharm_nn_prediction.to_csv("Results\\scharm_rf_prediction.csv")
# scharm_nn_pred.to_csv("Results\\scharm_rf_pred.csv")
#
# print("F_score = " + str(f1_score(scharm_nn_pred.Action, scharm_nn_pred.nn_predction.astype(int), average='weighted')))



# # d32
# d32_data = pd.read_excel("datasets/oneshot/d32_updated.xlsx")
#
#
# data_with_action_name = _generate_action_name(d32_data)
# data_with_action_name.to_csv("datasets\\oneshot\\d32_data_with_action_name.csv")
# #other data sets
d32_df = pd.read_csv("datasets/oneshot/d32_data_with_action_name.csv")#read_excel("datasets/oneshot/NewData.xlsx")
d32_df = d32_df.fillna(d32_df.mean())
folds = _read_roy_folds(open("datasets/oneshot/d32_folds.txt", "r"))


#Model Phase
#rf_results_df = _evaluation_roy_splits(data, RandomForestClassifier(n_estimators=100), 'Action')

for scenario in ['NONE']:#['A','B','C','D','E','F','NONE']:
    raw_data = d32_df.copy()
    raw_data = raw_data.drop(
        ['DateTime', 'CandName1', 'CandName2', 'CandName3', 'VotesCand1', 'ResponseTime', 'VotesCand2', 'VotesCan3',
         'PointsPostVote', 'WinnerPostVote', 'AU', 'KP', 'LD', 'LDLB', 'CV', 'BW', 'HF', 'heuristic', 'parameter'],
        axis=1)

#    loo_folds = _get_loo_folds(raw_data)
    personal_rf_clf = PersonalClassifier(id_index=raw_data.columns.get_loc("VoterID"), n_upsample=3)#RandomForestClassifier(n_estimators=100)  # MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
    personal_nn_clf = PersonalClassifier(id_index=raw_data.columns.get_loc("VoterID"), base_classifier=MLPClassifier(hidden_layer_sizes = (92), max_iter = 500), n_upsample=10, general_base_classifier=True)#RandomForestClassifier(n_estimators=100)  # MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
    neural_net_cf = MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
    rf_clf = RandomForestClassifier(n_estimators=100)
    dt_clf = DecisionTreeClassifier()
    adaboost_clf = AdaBoostClassifier(n_estimators=200)
    svm_clf = SVC()
    logistics_clf = logistic.LogisticRegression()

    classifiers = [personal_rf_clf, personal_nn_clf, neural_net_cf, rf_clf, dt_clf, adaboost_clf, svm_clf, logistics_clf]

    d32_performance_df,d32_prediction, d32_pred, feature_importances, features_train = _evaluation_roy_splits(raw_data, classifiers, 'Action', folds, scenario)

    d32_performance_df.to_csv("Results\\d32_performance_df_"+scenario+".csv")
#    d32_prediction.to_csv("Results\\d32_rf_prediction_"+scenario+".csv")
    d32_pred.to_csv("Results\\d32_pred_"+scenario+".csv")
#    feature_importances.to_csv("Results\\d32_feature_importance_"+scenario+".csv")
#    features_train.to_csv("Results\\d32_feature_train_"+scenario+".csv")

    print("F_score = " + str(f1_score(d32_pred.Action, d32_pred.Prediction.astype(int), average='weighted')))


#d36
# d36_data = pd.read_excel("datasets/oneshot/d36_updated.xlsx")
#
# data_with_action_name = _generate_action_name(d36_data)
# data_with_action_name.to_csv("datasets\\oneshot\\d36_data_with_action_name.csv")
# #other data sets
# d36_df = pd.read_csv("datasets/oneshot/d36_data_with_action_name.csv")#read_excel("datasets/oneshot/NewData.xlsx")
# d36_df = d36_df.fillna(d36_df.mean())
# folds = _read_roy_folds(open("datasets/oneshot/d36_folds.txt", "r"))
#
#
#
# #Model Phase
# #rf_results_df = _evaluation_roy_splits(data, RandomForestClassifier(n_estimators=100), 'Action')
#
# for scenario in ['A','B','C','D','E','F','NONE']:
#     classifier = RandomForestClassifier(n_estimators=100)  # MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
#     raw_data = d36_df.copy()
#
#     raw_data = raw_data.drop(['DateTime', 'CandName1', 'CandName2', 'CandName3', 'VotesCand1', 'ResponseTime', 'VotesCand2', 'VotesCan3',
#      'PointsPostVote', 'WinnerPostVote', 'AU', 'KP', 'LD', 'LDLB', 'CV', 'BW', 'HF', 'heuristic', 'parameter'],
#     axis=1)
#
#     d36_nn_performance_df,d36_nn_prediction, d36_nn_pred,feature_importances,features_train = _evaluation_roy_splits(raw_data, classifier, 'Action', folds,scenario)
#
#     d36_nn_performance_df.to_csv("Results\\d36_rf_performance_df_"+scenario+".csv")
#     d36_nn_prediction.to_csv("Results\\d36_rf_prediction_"+scenario+".csv")
#     d36_nn_pred.to_csv("Results\\d36_rf_pred_"+scenario+".csv")
#     feature_importances.to_csv("Results\\d36_feature_importance_"+scenario+".csv")
#     features_train.to_csv("Results\\d36_feature_train_"+scenario+".csv")
#
#     d36_nn_pred = _convert_prediction(d36_nn_pred)
#
#     d36_nn_pred.to_csv("Results\\V_d36_rf_pred_"+scenario+".csv")
#
#
#     print("F_score = " + str(f1_score(d36_nn_pred.Action, d36_nn_pred.Prediction.astype(int), average='weighted')))
#
#

