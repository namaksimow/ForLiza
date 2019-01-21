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
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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
from OneShotFeatureGenerator import OneShotStaticFeatureGenerator
from OneShotFeatureGenerator import OneShotDynamicFeatureGenerator
from OneShotDataPreperation import OneShotDataPreparation
from OrdinalClassifier import OrdinalClassifier
from BaselineModel import DecisionTreeBaseline
from BayesRuleModel import BayesRuleClassifier
from LikelihoodModel import LHClassifier
from MaximumLikelihoodModel import MLHClassifier

def _convert_prediction(X, column_name, n_candidates):
    X.loc[X[column_name]==1,"Vote_"+column_name] = X.loc[X[column_name]==1,"Pref1"]
    X.loc[X[column_name]==2,"Vote_"+column_name] = X.loc[X[column_name]==2,"Pref2"]
    X.loc[X[column_name]==3,"Vote_"+column_name] = X.loc[X[column_name]==3,"Pref3"]
    if n_candidates == 4:
        X.loc[X[column_name] ==4, "Vote_" + column_name] = X.loc[X[column_name] == 4, "Pref4"]

    return X

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
        fold = [x[1].RoundIndex]
        folds.append(fold)

    return folds

def _get_k_folds(X,k):
    folds = list()
    kf = KFold(k, shuffle=True, random_state=1)  # 10 fold cross validation
    for train_indices, test_indices in kf.split(X):
        folds.append(test_indices)
    return folds

def _evaluation(raw_data, clfs, target, folds, scenario_filter, action_table_df, scenarios_df, n_candidates = 3):
    data = raw_data.copy()

    oneshot_static_fg = OneShotStaticFeatureGenerator(action_table_df, scenarios_df, n_candidates)
    oneshot_dyn_fg = OneShotDynamicFeatureGenerator(action_table_df, scenarios_df, n_candidates)

    #static features generation
    data = oneshot_static_fg._static_feature_generation(data)

    n_folds = len(folds)


    results_df = pd.DataFrame(columns=['Classifier','FOLD','PRECISION','RECALL','F_MEASURE','ACCURACY'])

    prediction = pd.DataFrame(np.matrix([]))
    feature_importances = pd.DataFrame()
    features_train = pd.DataFrame()
    # 10 fold cross validation
    for i in range(0,len(folds)):

        print(str(100*(i/len(folds)))+"%")
        # Split into features and target
        features_df, target_df = data.drop([target], axis=1),data[target]

        test_indices = data.index[[x[1].RoundIndex in folds[i] for x in data.iterrows()]].tolist()
        train_indices =  data.index[[not (x[1].RoundIndex in folds[i] or x[1].Scenario == scenario_filter) for x in data.iterrows()]].tolist()
         # Feature Generation
        features_train = features_df.loc[[ii for ii in train_indices],]
        targets_train = target_df[[ii for ii in train_indices]]
        features_ext_df = oneshot_dyn_fg._dynamic_feature_generation(features_df, features_train , targets_train)
        features_ext_df = features_ext_df.drop(["Vote"], axis=1)
        # encoding the dataframes
        features_encoded_df = OneShotDataPreparation._prepare_dataset(features_ext_df)
        target_encoded_df = target_df
        # make training and testing datasets
        features_train = features_encoded_df.loc[[ii for ii in train_indices],]
        features_test = features_encoded_df.loc[[ii for ii in test_indices],]
        targets_train = target_encoded_df[[ii for ii in train_indices]]
        targets_test = target_encoded_df[[ii for ii in test_indices]]

        for j in range(0,len(clfs)):
            clf = clfs[j]
            clf_name = str(clf).split("(")[0]
            if i == 0:
                #Initialize metrics
                results_df.loc[j] = [str(clf), i + 1,0, 0, 0, 0]

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

            #aggregate results
            if len(prediction) == 0:
                prediction =  pd.DataFrame(predicated)
            else:
                prediction = pd.concat([prediction, pd.DataFrame(predicated)])

            raw_data.loc[[ii for ii in test_indices],"Prediction" + "_" + clf_name] =  predicated

            raw_data = _convert_prediction(raw_data, "Prediction" + "_" + clf_name, n_candidates)

            print(str(clf) +": F_score = " + str(f1_score(targets_test, predicated, average='weighted')))
            # Measures
            results_df.iloc[j + i, 1] = results_df.iloc[j + i, 1] + precision_score(targets_test, predicated, average='weighted')
            results_df.iloc[j + i, 2] = results_df.iloc[j + i, 2] + recall_score(targets_test, predicated, average='weighted')
            results_df.iloc[j + i, 3] = results_df.iloc[j + i, 3] + f1_score(targets_test, predicated, average='weighted')
            results_df.iloc[j + i, 4] = results_df.iloc[j + i, 4] + accuracy_score(targets_test, predicated)

            # if i == n_folds - 1:
            #     results_df.iloc[j, 1] = results_df.iloc[j, 1]/n_folds
            #     results_df.iloc[j, 2] = results_df.iloc[j, 2]/n_folds
            #     results_df.iloc[j, 3] = results_df.iloc[j, 3]/n_folds
            #     results_df.iloc[j, 4] = results_df.iloc[j, 4]/n_folds


            #results_df.Result = results_df.Result.apply(lambda x: x / n_folds)
    return results_df, raw_data#, feature_importances

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

def _load_and_run(datasets, load_folds, classifiers, n_candidates, scenarios = ['NONE'], is_loo = False, n_folds = 10):
    actions_table = pd.read_csv("datasets/oneshot/action_table_N"+str(n_candidates)+".csv")
    scenarios_table = pd.read_csv("datasets/oneshot/scenario_table_N"+str(n_candidates)+".csv")

    for dataset in datasets:
        file_path = "datasets/oneshot/" + dataset + ".xlsx"
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            #Get sheet from xlsx
            data = pd.read_excel(file_path, sheet_name=sheet)
            d_df = data.fillna(data.mean())

            #Prepare folds
            if load_folds == True:
                folds = _read_roy_folds(open("datasets/oneshot/"+dataset+"_folds.txt", "r"))
            else:
                if is_loo == True:
                    folds = _get_loo_folds(d_df)
                else:
                    folds = _get_k_folds(d_df, n_folds)

            for scenario in scenarios:  # ['A','B','C','D','E','F','NONE']:
                raw_data = d_df.copy()

                d_performance_df, d_pred = _evaluation(raw_data, classifiers, 'Action', folds, scenario, actions_table, scenarios_table, n_candidates)
                d_performance_df.to_csv("Results\\" + dataset + "_" + sheet + "_performance_df_" + scenario + ".csv")
                d_pred.to_csv("Results\\" + dataset + "_" + sheet + "_pred_" + scenario + ".csv")
    pass


#---------------------------------- Classifiers Definition ------------------------------------#
# personal_rf_clf = PersonalClassifier(id_index=raw_data.columns.get_loc("VoterID"), n_upsample=3)#RandomForestClassifier(n_estimators=100)  # MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
# personal_nn_clf = PersonalClassifier(id_index=raw_data.columns.get_loc("VoterID"), base_classifier=MLPClassifier(hidden_layer_sizes = (92), max_iter = 500), n_upsample=10, general_base_classifier=True)#RandomForestClassifier(n_estimators=100)  # MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
# neural_net_cf = MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
rf_clf = RandomForestClassifier(n_estimators=100)
# dt_clf = DecisionTreeClassifier()
# adaboost_clf = AdaBoostClassifier(n_estimators=200)
# svm_clf = SVC()
# logistics_clf = logistic.LogisticRegression()
#ordinal_clf = OrdinalClassifier(base_classifier = RandomForestClassifier(n_estimators=100))
#baseline_clf = DecisionTreeBaseline()
# bayesrule_clf = BayesRuleClassifier()
# likelihood_clf = LHClassifier()
# maxlikelihood_clf = MLHClassifier()

classifiers = [rf_clf]  # ,personal_nn_clf,neural_net_cf, rf_clf,dt_clf,adaboost_clf, svm_clf,logistics_clf]
#---------------------------------- Classifiers Definition ------------------------------------#
#----------------------------------- Dataset definition ---------------------------------------#
# datasets: ["schram"]#["d36_2_folds","d36_4_folds","d36_6_folds","d32_2_folds","d32_4_folds","d32_6_folds"]
datasets = ["schram"]
n_candidates = 3

_load_and_run(datasets=datasets, load_folds=True, classifiers=classifiers, n_candidates=n_candidates)

