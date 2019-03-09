# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:16:03 2018

@author: Adam
"""

import numpy as np
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE

from datetime import datetime
# Model and feature selection
from sklearn.model_selection import KFold
# Classification metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from PersonalClassifier import PersonalClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import logistic
from OneShotFeatureGenerator import OneShotStaticFeatureGenerator
from OneShotFeatureGenerator import OneShotDynamicFeatureGenerator
from OneShotDataPreperation import OneShotDataPreparation
from OrdinalClassifier import OrdinalClassifier
from ExpertModels import DecisionTreeBaseline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

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
    if k == 1:
        folds.append(X.index.tolist())
    else:
        kf = KFold(k, shuffle=True, random_state=1)  # 10 fold cross validation
        for train_indices, test_indices in kf.split(X):
            folds.append(X.iloc[test_indices].RoundIndex)
    return folds

def _select_features(features_train, targets_train, features_ext_df):
    #feature importance
    feature_importance = pd.DataFrame()

    rf_for_fs = RandomForestClassifier(n_estimators=100)
    rf_for_fs.fit(X=features_train.values, y=targets_train)
    current_feature_importances = pd.DataFrame(rf_for_fs.feature_importances_,
                                               index=features_ext_df.columns,
                                               columns=['importance']).sort_values('importance',
                                                                                   ascending=False)
    if len(feature_importance) == 0:
        feature_importance = current_feature_importances
    else:
        feature_importance['importance'] = feature_importance['importance'] + current_feature_importances['importance']

    feature_importance['importance_percentage'] = feature_importance['importance']/np.max(feature_importance['importance'])
    selected_comlumns = feature_importance.iloc[[feature_importance['importance_percentage']>0.2],].index.tolist()
    return selected_comlumns


def _evaluation(raw_data, clfs, target, folds, scenario_filter, action_table_df, scenarios_df,n_candidates = 3):
    data = raw_data.copy()
    data = data.drop(["Vote"], axis=1)

    oneshot_static_fg = OneShotStaticFeatureGenerator(action_table_df, scenarios_df, n_candidates)
    oneshot_dyn_fg = OneShotDynamicFeatureGenerator(action_table_df, scenarios_df, n_candidates)

    #static features generation
    data = oneshot_static_fg._static_feature_generation(data)

    n_folds = len(folds)

    results_df = pd.DataFrame(columns=['Classifier','FOLD','PRECISION','RECALL','F_MEASURE','ACCURACY'])

    prediction = pd.DataFrame(np.matrix([]))

    features_train = pd.DataFrame()
    # 10 fold cross validation
    for i in range(0,len(folds)):

        print(str(100*(i/len(folds)))+"%")
        # Split into features and target
        features_df, target_df = data.drop([target], axis=1),data[target]

        if n_folds == 1: #Upperbound case
            test_indices = data.index.tolist()
            train_indices = data.index.tolist()
        else:
            test_indices = data.index[[x[1].RoundIndex in folds[i].tolist() for x in data.iterrows()]].tolist()
            train_indices = data.index[[not (x[1].RoundIndex in folds[i].tolist()) for x in data.iterrows()]].tolist()
         # Feature Generation
        features_train = features_df.loc[[ii for ii in train_indices],]
        targets_train = target_df[[ii for ii in train_indices]]
        features_ext_df = oneshot_dyn_fg._dynamic_feature_generation(features_df, features_train, targets_train)
#        features_ext_df = features_ext_df.drop(["Vote"], axis=1)
        # encoding the dataframes
        features_encoded_df = OneShotDataPreparation._prepare_dataset(features_ext_df.copy())
        target_encoded_df = target_df
        # make training and testing datasets
        features_train = features_encoded_df.loc[[ii for ii in train_indices],]
        features_test = features_encoded_df.loc[[ii for ii in test_indices],]
        targets_train = target_encoded_df[[ii for ii in train_indices]]
        targets_test = target_encoded_df[[ii for ii in test_indices]]

        # select features
        #selected_columns = _select_features(features_train, targets_train, features_ext_df)

        for j in range(0,len(clfs)):
            clf = clfs[j]
            clf_name = str(clf).split("(")[0]
            # if i == 0:
            #     #Initialize metrics
            #     results_df.loc[j] = [str(clf), i + 1,0, 0, 0, 0]


            # Train
            clf.fit(X=features_train.values, y=targets_train)

            if "DecisionTreeBaseline" in clf_name:
                features_ext_df.to_csv("datasets/oneshot/test_features.csv")
                targets_test.to_csv("datasets/oneshot/test_target.csv")
                predicated = clf.predict(features_ext_df.loc[[ii for ii in test_indices],])
            else:
                # Test
                predicated = clf.predict(features_test.values)


            #aggregate results
            if len(prediction) == 0:
                prediction =  pd.DataFrame(predicated)
            else:
                prediction = pd.concat([prediction, pd.DataFrame(predicated)])

            raw_data.loc[[ii for ii in test_indices],"Prediction" + "_" + clf_name] =  predicated

            raw_data = _convert_prediction(raw_data, "Prediction" + "_" + clf_name, n_candidates)

            print(str(clf) +": F_score = " + str(f1_score(targets_test, predicated, average='weighted')))
            # Measures

            results_df.loc[i*len(clfs) + j] = [str(clf), i + 1, precision_score(targets_test, predicated, average='weighted'),  recall_score(targets_test, predicated, average='weighted'), f1_score(targets_test, predicated, average='weighted'), accuracy_score(targets_test, predicated)]

            # if i == n_folds - 1:
            #     results_df.iloc[j, 1] = results_df.iloc[j, 1]/n_folds
            #      results_df.iloc[j, 2] = results_df.iloc[j, 2]/n_folds
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

def _get_classifiers(df, n_candidates):
    neural_net_cf = MLPClassifier(hidden_layer_sizes = (50), max_iter = 500, random_state=1)
    two_layer_nn_cf = MLPClassifier(hidden_layer_sizes = (50,30), max_iter = 500, random_state=1)
    three_layer_nn_cf = MLPClassifier(hidden_layer_sizes = (50,30,20), max_iter = 500, random_state=1)
    nn_cf_2 = MLPClassifier(hidden_layer_sizes=(90), max_iter=500, random_state=1)
    nn_cf_3 = MLPClassifier(hidden_layer_sizes=(20), max_iter=500, random_state=1)
    rf_clf1 = RandomForestClassifier(n_estimators=20, random_state=1)
    rf_clf2 = RandomForestClassifier(n_estimators=40, random_state=1)
    rf_clf3 = RandomForestClassifier(n_estimators=60, random_state=1)
    rf_clf4 = RandomForestClassifier(n_estimators=100, random_state=1)
    rf_clf5 = RandomForestClassifier(n_estimators=300, random_state=1)
    rf_clf6 = RandomForestClassifier(n_estimators=400, random_state=1)
    dt_clf = DecisionTreeClassifier()
    adaboost_clf = AdaBoostClassifier(n_estimators=30, random_state=1)
    adaboost_clf2 = AdaBoostClassifier(n_estimators=50, random_state=1)
    adaboost_clf3 = AdaBoostClassifier(n_estimators=80, random_state=1)
    adaboost_clf4 = AdaBoostClassifier(n_estimators=300, random_state=1)
    svm_clf = SVC(kernel="poly", degree=4, random_state=1)
    svm_clf2 = SVC(kernel="sigmoid", degree=4, random_state=1)
    svm_clf3 = SVC(kernel="rbf", degree=4, random_state=1)
    logistics_clf = logistic.LogisticRegression(random_state=1)
    extra_tree_clf = ExtraTreesClassifier(random_state=1)
    gb_clf = GradientBoostingClassifier(random_state=1)
    if n_candidates == 3:
        ordered_class = [1,2,3]
    else:
        ordered_class = [1,2,3,4]

    rfi1_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
                                             n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=20, random_state=1))
    rfi2_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
                                             n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=40, random_state=1))
    rfi3_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
                                             n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=60, random_state=1))
    rfi4_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
                                             n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=80, random_state=1))

    personal_nn_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
                                             base_classifier=MLPClassifier(hidden_layer_sizes=(50), max_iter=500, random_state=1),
                                             n_upsample=10,
                                             general_base_classifier=True)  # RandomForestClassifier(n_estimators=100)  # MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)

    ordinal_clf = OrdinalClassifier(base_classifier = RandomForestClassifier, ordered_class=ordered_class)

    #naive_bayes_clf = sklearn.naive_bayes()
    # bayesrule_clf = BayesRuleClassifier()
    # likelihood_clf = LHClassifier()
    # maxlikelihood_clf = MLHClassifier()
    if n_candidates == 3:
        baseline_clf = DecisionTreeBaseline()
        classifiers = [rf_clf3]#[baseline_clf, extra_tree_clf, gb_clf, rfi1_clf, rfi2_clf, rfi3_clf, rfi4_clf, ordinal_clf ,personal_nn_clf,neural_net_cf,nn_cf_2, nn_cf_3, two_layer_nn_cf, three_layer_nn_cf, rf_clf1,rf_clf2, rf_clf3,rf_clf4,rf_clf5, rf_clf6, dt_clf,adaboost_clf,adaboost_clf2, adaboost_clf3,adaboost_clf4, svm_clf, svm_clf2, svm_clf3,logistics_clf]
    else:
        classifiers = [extra_tree_clf, gb_clf, rfi1_clf, rfi2_clf, rfi3_clf, rfi4_clf, ordinal_clf,
                       personal_nn_clf, neural_net_cf, nn_cf_2, nn_cf_3, two_layer_nn_cf, three_layer_nn_cf, rf_clf1,
                       rf_clf2, rf_clf3, rf_clf4, rf_clf5, rf_clf6, dt_clf, adaboost_clf, adaboost_clf2, adaboost_clf3,
                       adaboost_clf4, svm_clf, svm_clf2, svm_clf3, logistics_clf]

    return classifiers

def _load_and_run(datasets, load_folds, scenarios = ['NONE'], is_loo = False, fold_set = [10]):

    for dataset in datasets:
        file_path = "datasets/oneshot/" + dataset + ".xlsx"
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            #Get sheet from xlsx
            data = pd.read_excel(file_path, sheet_name=sheet)

            #Take sample from data
            data = data.sample(frac=0.05,replace=False, random_state=1)

            d_df = data.fillna(data.mean())

            n_candidates = d_df.iloc[0]["NumberOfCandidates"]
            actions_table = pd.read_csv("datasets/oneshot/action_table_N" + str(n_candidates) + ".csv")
            scenarios_table = pd.read_csv("datasets/oneshot/scenario_table_N" + str(n_candidates) + ".csv")
            classifiers = _get_classifiers(d_df, n_candidates)

            #Prepare folds
            for n_folds in fold_set:
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
                    d_performance_df.to_csv("Results\\" + dataset + "_" + sheet + "_performance_df_" + scenario + "_" + str(n_folds) + ".csv")
                    d_pred.to_csv("Results\\" + dataset + "_" + sheet + "_pred_" + scenario + "_" + str(n_folds) + ".csv")

    pass


#---------------------------------- Classifiers Definition ------------------------------------#

#---------------------------------- Classifiers Definition ------------------------------------#
#----------------------------------- Dataset definition ---------------------------------------#
# datasets: ["schram"]#["d36_2_folds","d36_4_folds","d36_6_folds","d32_2_folds","d32_4_folds","d32_6_folds"]
# datasets = ["schram"]
# n_candidates = 3
#
# _load_and_run(datasets=datasets, load_folds=True, classifiers=classifiers, n_candidates=n_candidates)
#
datasets = ["d36_updated_train"]#["schram_train","tal_train","d36_updated_train","d32_updated_train","N4_first_90_train"] #["N4_first_90", "d32_updated", "d36_updated", "tal", "schram"]#["N4_first_90_sample", "d32_updated_sample", "d36_updated_sample", "tal_sample", "schram_sample"]#["N4_first_90", "d32_updated", "d36_updated", "tal", "schram"]
fold_set = [10]#, 10]
_load_and_run(datasets=datasets, load_folds=False, fold_set=fold_set)
#

# datasets = ["N4_first_90", "d32_updated", "d36_updated", "tal", "schram", "N4_first_90_train", "d32_updated_train", "d36_updated_train", "tal_train", "schram_train"]
# for dataset in datasets:
#     file_path = "datasets/oneshot/PartionedDatasets/Original/" + dataset + ".xlsx"
#     xls = pd.ExcelFile(file_path)
#     for sheet in xls.sheet_names:
#         #Get sheet from xlsx
#         data = pd.read_excel(file_path, sheet_name=sheet)
#         data_train, data_test = train_test_split(data, random_state=1, test_size=0.2)
#         data_train.to_excel("datasets\\oneshot\\PartionedDatasets\\" + dataset + "_train.xlsx")
#         data_test.to_excel("datasets\\oneshot\\PartionedDatasets\\" + dataset + "_test.xlsx")