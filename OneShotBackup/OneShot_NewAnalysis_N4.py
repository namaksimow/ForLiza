# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:16:03 2018

@author: Adam
"""
import pickle
import numpy as np
import pandas as pd
import glob
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
from sklearn.metrics import confusion_matrix
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from ExpertModels import ScenarioClassifier

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

def _read_folds_from_file(folds_file):
    with open(folds_file, 'rb') as fp:
        folds = pickle.load(fp)

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
    #neural_net_cf = MLPClassifier(hidden_layer_sizes = (50), max_iter = 500, random_state=1)
    scenario_clf = ScenarioClassifier(GradientBoostingClassifier(n_estimators=300))
    # two_layer_nn_cf = MLPClassifier(hidden_layer_sizes = (50,30), max_iter = 500, random_state=1)
    # three_layer_nn_cf = MLPClassifier(hidden_layer_sizes = (50,30,20), max_iter = 500, random_state=1)
    # nn_cf_2 = MLPClassifier(hidden_layer_sizes=(90), max_iter=500, random_state=1)
    # nn_cf_3 = MLPClassifier(hidden_layer_sizes=(20), max_iter=500, random_state=1)
    # rf_clf1 = RandomForestClassifier(n_estimators=20, random_state=1)
    # rf_clf2 = RandomForestClassifier(n_estimators=40, random_state=1)
    # rf_clf3 = RandomForestClassifier(n_estimators=60, random_state=1)
    # rf_clf4 = RandomForestClassifier(n_estimators=100, random_state=1)
    # dt_clf = DecisionTreeClassifier()
    # adaboost_clf = AdaBoostClassifier(n_estimators=30, random_state=1)
    # adaboost_clf2 = AdaBoostClassifier(n_estimators=50, random_state=1)
    # adaboost_clf3 = AdaBoostClassifier(n_estimators=80, random_state=1)
    # adaboost_clf4 = AdaBoostClassifier(n_estimators=300, random_state=1)
    # svm_clf = SVC(kernel="poly", degree=4, random_state=1)
    # svm_clf2 = SVC(kernel="sigmoid", degree=4, random_state=1)
    # svm_clf3 = SVC(kernel="rbf", degree=4, random_state=1)
    # logistics_clf = logistic.LogisticRegression(random_state=1)
    # extra_tree_clf = ExtraTreesClassifier(random_state=1)
    rf_clf5 = RandomForestClassifier(n_estimators=300, random_state=1)
    rf_clf6 = RandomForestClassifier(n_estimators=400, random_state=1)
    gb_clf = GradientBoostingClassifier(random_state=1)
    gb_clf_300 = GradientBoostingClassifier(random_state=1, n_estimators=300)
    if n_candidates == 3:
        ordered_class = [1,2,3]
    else:
        ordered_class = [1,2,3,4]

    # rfi1_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
    #                                          n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=20, random_state=1))
    # rfi2_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
    #                                          n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=40, random_state=1))
    # rfi3_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
    #                                          n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=60, random_state=1))
    # rfi4_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
    #                                          n_upsample=10, base_classifier=RandomForestClassifier(n_estimators=80, random_state=1))

    # personal_nn_clf = PersonalClassifier(id_index=df.columns.get_loc("VoterID"), classes=ordered_class,
    #                                          base_classifier=MLPClassifier(hidden_layer_sizes=(50), max_iter=500, random_state=1),
    #                                          n_upsample=10,
    #                                          general_base_classifier=True)  # RandomForestClassifier(n_estimators=100)  # MLPClassifier(hidden_layer_sizes = (92), max_iter = 500)
    #
    # ordinal_clf = OrdinalClassifier(base_classifier = RandomForestClassifier, ordered_class=ordered_class)

    #naive_bayes_clf = sklearn.naive_bayes()
    # bayesrule_clf = BayesRuleClassifier()
    # likelihood_clf = LHClassifier()
    # maxlikelihood_clf = MLHClassifier()
    if n_candidates == 3:
        baseline_clf = DecisionTreeBaseline()
        classifiers = [baseline_clf]#, gb_clf, rf_clf5, rf_clf6, gb_clf_300]#[baseline_clf, extra_tree_clf, gb_clf, rfi1_clf, rfi2_clf, rfi3_clf, rfi4_clf, ordinal_clf ,personal_nn_clf,neural_net_cf,nn_cf_2, nn_cf_3, two_layer_nn_cf, three_layer_nn_cf, rf_clf1,rf_clf2, rf_clf3,rf_clf4,rf_clf5, rf_clf6, dt_clf,adaboost_clf,adaboost_clf2, adaboost_clf3,adaboost_clf4, svm_clf, svm_clf2, svm_clf3,logistics_clf]
    else:
        classifiers = [gb_clf, rf_clf5, rf_clf6, gb_clf_300]
        #[extra_tree_clf, gb_clf, rfi1_clf, rfi2_clf, rfi3_clf, rfi4_clf, ordinal_clf,
        #personal_nn_clf, neural_net_cf, nn_cf_2, nn_cf_3, two_layer_nn_cf, three_layer_nn_cf, rf_clf1,
        #rf_clf2, rf_clf3, rf_clf4, rf_clf5, rf_clf6, dt_clf, adaboost_clf, adaboost_clf2, adaboost_clf3,
        #adaboost_clf4, svm_clf, svm_clf2, svm_clf3, logistics_clf]

    return classifiers

def _features_importance(features_ext_df, features_train, targets_train):
    #feature importance
    feature_importance = pd.DataFrame()

    rf_for_fs = RandomForestClassifier(n_estimators=300, random_state=1)
    transformed_features_train = OneShotDataPreparation._prepare_dataset(features_ext_df.loc[features_train.index, :])
    rf_for_fs.fit(X=transformed_features_train, y=targets_train)
    current_feature_importances = pd.DataFrame(rf_for_fs.feature_importances_,
                                               index=features_ext_df.columns,
                                               columns=['importance']).sort_values('importance',
                                                                                   ascending=False)
    if len(feature_importance) == 0:
        feature_importance = current_feature_importances
    else:
        feature_importance['importance'] = feature_importance['importance'] + current_feature_importances['importance']

    feature_importance['importance_percentage'] = feature_importance['importance']/np.max(feature_importance['importance'])

    return feature_importance

def _evaluation_by_files(datasets, target, top_k_fs = 25, select_features = True, clfs = None):
    results_df = pd.DataFrame(columns=['Dataset', 'Instances', 'Classifier', 'FOLD', 'PRECISION', 'RECALL', 'F_MEASURE', 'ACCURACY'])

    #results_scenario_df = pd.DataFrame(columns=['Dataset','Classifier', 'FOLD','INSTANCES','SCENARIO', 'PRECISION', 'RECALL', 'F_MEASURE', 'ACCURACY'])

    prediction = pd.DataFrame(np.matrix([]))

    for dataset in datasets:
        features_importance = pd.DataFrame(np.matrix([]))
        #features_importance_scenario = pd.DataFrame(np.matrix([]))

        fold_files = glob.glob("datasets\\oneshot\\Folds prepared\\"+dataset+"_fold_*.csv")
        for i in range(0,len(fold_files)):
            fold_file = fold_files[i]
            data_df = pd.read_csv(fold_file)

            n_candidates = data_df.iloc[0]["NumberOfCandidates"]
            action_table_df = pd.read_csv("datasets/oneshot/action_table_N" + str(n_candidates) + ".csv")
            scenarios_df = pd.read_csv("datasets/oneshot/scenario_table_N" + str(n_candidates) + ".csv")
            if clfs is None:
                clfs = _get_classifiers(data_df, n_candidates)

            oneshot_dyn_fg = OneShotDynamicFeatureGenerator(action_table_df, scenarios_df, n_candidates)

            train_indices = data_df["SET"] == "TRAIN"
            test_indices = data_df["SET"] == "TEST"
            data_df = data_df.drop(["RoundIndex","SET", "Unnamed: 0"], axis = 1)

            # Split into features and target
            features_df, target_df = data_df.drop([target], axis=1), data_df[target]

            features_train = features_df.loc[train_indices]
            targets_train = target_df[train_indices]
            features_test = features_df.loc[test_indices]

            features_ext_df = oneshot_dyn_fg._dynamic_feature_generation_additional(features_df, features_train, targets_train)

            file_to_save = pd.concat([features_ext_df, target_df], axis=1, join='inner')
            file_to_save.loc[train_indices,"SET"] = "TRAIN"
            file_to_save.loc[test_indices,"SET"] = "TEST"
            file_to_save.index = data_df.index

            baseline_set = features_ext_df.loc[:, ["Scenario", "VoterType"]]
            if select_features:
                # Feature Selection
                selected_features_rfe = oneshot_dyn_fg._select_features(features_ext_df, features_train, targets_train, top_k=top_k_fs)
                current_selected_features = pd.DataFrame(selected_features_rfe)
                current_selected_features.loc[:, "FOLD"] = str(i + 1)
                features_ext_df = features_ext_df.drop(
                    features_ext_df.columns[[not (x in selected_features_rfe) for x in
                                             features_ext_df.columns]].tolist(),
                    axis=1)

            # Feature Importance
            # current_feature_importance = _features_importance(features_ext_df, features_train, targets_train)
            # current_feature_importance.loc[:, "FOLD"] = str(i + 1)
            # if len(features_importance) == 0:
            #     features_importance = current_feature_importance
            # else:
            #     features_importance = pd.concat([features_importance, current_feature_importance])
            #
            # for scenario in scenarios_df["scenario"]:
            #     scenario_index = features_ext_df.loc[features_ext_df["Scenario"] == scenario].index & features_test.index
            #     current_feature_importance = _features_importance(features_ext_df.loc[features_ext_df["Scenario"] == scenario], features_train.loc[features_train["Scenario"] == scenario], targets_train[features_train["Scenario"] == scenario])
            #     current_feature_importance.loc[:, "FOLD"] = str(i + 1)
            #     current_feature_importance.loc[:, "SCENARIO"] = scenario
            #     current_feature_importance.loc[:, "INSTANCES"] = np.count_nonzero(scenario_index)
            #     if len(features_importance_scenario) == 0:
            #         features_importance_scenario = current_feature_importance
            #     else:
            #         features_importance_scenario = pd.concat([features_importance_scenario, current_feature_importance])
            #
            #Feature Importance ----------------------------END

            # encoding the dataframes

            features_encoded_df = OneShotDataPreparation._prepare_dataset(features_ext_df.copy())
            features_encoded_df.index = data_df.index

            # make training and testing datasets
            features_train = features_encoded_df.loc[train_indices]
            features_test = features_encoded_df.loc[test_indices]
            targets_train = target_df[train_indices]
            targets_test = target_df[test_indices]

            for j in range(0, len(clfs)):
                clf = clfs[j]
                clf_name = str(clf).split("(")[0]
                clf.fit(X=features_train.values, y=targets_train)

                if "DecisionTreeBaseline" in clf_name:
                    predicated = clf.predict(baseline_set.loc[[ii for ii in test_indices],])
                else:
                    # Test
                    predicated = clf.predict(features_test.values)

                # aggregate results
                if len(prediction) == 0:
                    prediction = pd.DataFrame(predicated)
                else:
                    prediction = pd.concat([prediction, pd.DataFrame(predicated)])

                file_to_save.loc[test_indices, "Prediction" + "_" + clf_name] = predicated

                # = _convert_prediction(data_df, "Prediction" + "_" + clf_name, n_candidates)

                print(str(clf) + ": F_score = " + str(f1_score(targets_test, predicated, average='weighted')))

                # Measures
                results_df.iloc[len(results_df)] = [dataset, np.count_nonzero(test_indices), str(clf), i + 1,
                                                     precision_score(targets_test, predicated, average='weighted'),
                                                     recall_score(targets_test, predicated, average='weighted'),
                                                     f1_score(targets_test, predicated, average='weighted'),
                                                     accuracy_score(targets_test, predicated)]

                # for scenario in scenarios_df["scenario"]:
                #     scenario_index = file_to_save.loc[file_to_save["Scenario"] == scenario].index & features_test.index
                #     predicted_scenario = clf.predict(baseline_set.loc[scenario_index])
                #     predicted_target = targets_test[scenario_index]
                #     results_scenario_df.loc[len(results_scenario_df)] = [dataset, str(clf), i + 1, np.count_nonzero(scenario_index),scenario,
                #                                          precision_score(predicted_target,predicted_scenario, average='weighted'),
                #                                          recall_score(predicted_target, predicted_scenario, average='weighted'),
                #                                          f1_score(predicted_target, predicted_scenario, average='weighted'),
                #                                          accuracy_score(predicted_target, predicted_scenario)]
                #

            file_to_save.to_csv("datasets/oneshot/extended/" + dataset + "_fold_" + str(i + 1) + "_" + str(top_k_fs) +  ".csv")


        features_importance.to_csv("datasets/oneshot/extended/" + dataset + "_feature_importance_" + str(top_k_fs) + ".csv")
        #features_importance_scenario.to_csv("datasets/oneshot/extended/" + dataset + "_feature_importance_scenario_" + str(top_k_fs) + ".csv")

    results_df.to_csv("datasets/oneshot/extended/results_" + str(top_k_fs) + ".csv")
    #results_scenario_df.to_csv("datasets/oneshot/extended/results_scenario_" + str(top_k_fs) + ".csv")

    pass

def _evaluation(raw_data, clfs, target, folds, action_table_df, scenarios_df, dataset_name, n_candidates = 3, top_k_fs = None, upperbound = False):
    data = raw_data.copy()
    data = data.drop(["Vote"], axis=1)

    # oneshot_static_fg = OneShotStaticFeatureGenerator(action_table_df, scenarios_df, n_candidates)
    #
    # #static features generation
    # data = oneshot_static_fg._static_feature_generation(data)
    n_folds = len(folds)
    results_df = pd.DataFrame(columns=['Dataset', 'Instances', 'Classifier', 'FOLD', 'PRECISION', 'RECALL', 'F_MEASURE', 'ACCURACY'])

    features_importance = pd.DataFrame()
    features_importance_scenario = pd.DataFrame()

    # 10 fold cross validation
    for i in range(0,len(folds)):

        print(str(100*(i/len(folds)))+"%")
        # Split into features and target
        features_df, target_df = data.drop([target], axis=1),data[target]

        if n_folds == 1 and upperbound: #Upperbound case
            test_indices = data.index
            train_indices = data.index
        else:
            test_indices = data.index[[(x[1].RoundIndex in folds[i].tolist()) for x in data.iterrows()]]
            train_indices = data.index[[not (x[1].RoundIndex in folds[i].tolist()) for x in data.iterrows()]]

        # Feature Generation
        features_train = features_df.loc[train_indices]
        targets_train = target_df[train_indices]
        features_test = features_df.loc[test_indices]

        oneshot_dyn_fg = OneShotDynamicFeatureGenerator(action_table_df, scenarios_df, n_candidates, features_train)
        features_ext_df = oneshot_dyn_fg._fit(features_train, targets_train)
#        features_ext_df = features_ext_df.drop(["Vote"], axis=1)
        features_ext_df = features_ext_df.drop(["RoundIndex"], axis=1)
        features_train = features_train.drop(["RoundIndex"], axis=1)

        #Save staff for later use
        # file_to_save = pd.concat([features_ext_df, target_df], axis=1, join='inner')
        # file_to_save.loc[train_indices,"SET"] = "TRAIN"
        # file_to_save.loc[test_indices,"SET"] = "TEST"
        # file_to_save.to_csv("datasets/oneshot/"+dataset_name+"_fold_"+str(i+1)+".csv")
        #
        # baseline_set = features_ext_df.loc[:, ["Scenario", "VoterType"]]
        # scenario_clf_set = features_ext_df.loc[:,["VoterID", "Scenario", "GAP12_pref_poll", "GAP13_pref_poll"]]

        # #Feature Importance
        # current_feature_importance = _features_importance(features_ext_df, features_train, targets_train)
        # current_feature_importance.loc[:, "FOLD"] = str(i + 1)
        # if len(features_importance) == 0:
        #     features_importance = current_feature_importance
        # else:
        #     features_importance = pd.concat([features_importance, current_feature_importance])
        #
        # for scenario in scenarios_df["scenario"]:
        #     scenario_index = features_ext_df.loc[features_ext_df["Scenario"] == scenario].index & features_test.index
        #     current_feature_importance = _features_importance(features_ext_df.loc[features_ext_df["Scenario"] == scenario], features_train.loc[features_train["Scenario"] == scenario], targets_train[features_train["Scenario"] == scenario])
        #     current_feature_importance.loc[:, "FOLD"] = str(i + 1)
        #     current_feature_importance.loc[:, "SCENARIO"] = scenario
        #     current_feature_importance.loc[:, "INSTANCES"] = np.count_nonzero(scenario_index)
        #     if len(features_importance_scenario) == 0:
        #         features_importance_scenario = current_feature_importance
        #     else:
        #         features_importance_scenario = pd.concat([features_importance_scenario, current_feature_importance])
        #
        # #Feature Importance ----------------------------END

        # Feature Selection
        if  top_k_fs is not None:
            selected_features_rfe = oneshot_dyn_fg._select_features(features_ext_df, features_train, targets_train, top_k_fs)
            current_selected_features = pd.DataFrame(selected_features_rfe)
            current_selected_features.loc[:, "FOLD"] = str(i+1)
 #           features_ext_df = features_ext_df.drop(
            selected_columns = [not (x in selected_features_rfe) for x in features_ext_df.columns]
        else: selected_columns = list()


        # encoding the dataframes
        features_encoded_df = pd.DataFrame(OneShotDataPreparation._prepare_dataset(features_ext_df.copy()))
        features_encoded_df.index = data.index

        target_encoded_df = target_df
        # make training and testing datasets
        features_train = features_encoded_df.loc[train_indices]
        features_test = features_encoded_df.loc[test_indices]
        targets_train = target_encoded_df[train_indices.tolist()]
        targets_test = target_encoded_df[test_indices.tolist()]

        for j in range(0,len(clfs)):
            clf = clfs[j]
            clf_name = str(clf).split("(")[0]

            if "ScenarioClassifier" in clf_name:
                features_encoded_df = features_encoded_df.loc[:, scenario_clf_set]
                clf._set_columns(features_encoded_df.columns)
                features_train = features_encoded_df.loc[train_indices]
                features_test = features_encoded_df.loc[test_indices]

            # Train
            clf.fit(X=features_train.values, y=targets_train)

            if "DecisionTreeBaseline" in clf_name:
                cur_predicated = pd.DataFrame(clf.predict(baseline_set.loc[[ii for ii in test_indices],]))
            else:
                # Test
                cur_predicated = pd.DataFrame(clf.predict(features_test.values))
                cur_predicated.index = test_indices


            #aggregate results
            # if len(prediction) == 0:
            #     prediction = cur_predicated
            # else:
            #     prediction = pd.concat([prediction, cur_predicated])

            raw_data.loc[test_indices,"Prediction" + "_" + clf_name] = cur_predicated

            raw_data = _convert_prediction(raw_data, "Prediction" + "_" + clf_name, n_candidates)

            print(str(clf) +": F_score = " + str(f1_score(targets_test, cur_predicated, average='weighted')))
            print(confusion_matrix(targets_test, cur_predicated))
            # Measures
            results_df.loc[len(results_df)] = [dataset_name, np.count_nonzero(test_indices), str(clf), i + 1,
                                                precision_score(targets_test, cur_predicated, average='weighted'),
                                                recall_score(targets_test, cur_predicated, average='weighted'),
                                                f1_score(targets_test, cur_predicated, average='weighted'),
                                                accuracy_score(targets_test, cur_predicated)]

    return results_df, raw_data, features_importance_scenario

def _load_and_run(datasets, load_folds, is_loo = False, fold_set = [10], k_folds = None):
    for dataset in datasets:
        file_path = "data/csv/" + dataset + ".csv"
        data = pd.read_csv(file_path)

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

            raw_data = d_df.copy()
            d_performance_df, d_pred, d_feature_importance = _evaluation(raw_data, classifiers, 'Action', folds, actions_table, scenarios_table, dataset, n_candidates, top_k_fs=k_folds)
            d_performance_df.to_csv("Results\\" + dataset + "_performance_df_" + str(n_folds) + ".csv")
            d_pred.to_csv("Results\\" + dataset + "_pred_" + str(n_folds) + ".csv")
            d_feature_importance.to_csv("Results\\" + dataset + "_feature_importance_" + str(n_folds) + ".csv")

    pass

def _test_eval(datasets, top_k):
    # Selected Classifiers
    rf_clf5 = RandomForestClassifier(n_estimators=300, random_state=1)
    rf_clf6 = RandomForestClassifier(n_estimators=400, random_state=1)
    gb_clf = GradientBoostingClassifier(random_state=1)
    gb_clf_300 = GradientBoostingClassifier(random_state=1, n_estimators=300)
    gb_clf_600 = GradientBoostingClassifier(random_state=1, n_estimators=1000)
    baseline_clf = DecisionTreeBaseline()
    clfs = [rf_clf5, rf_clf6, gb_clf, gb_clf_300,gb_clf_600]

    # Baseline
    #datasets = ["N4_first_90_test_eval"]#["d32_updated_test_eval", "d36_updated_test_eval", "tal_test_eval", "schram_test_eval"]
    for dataset in datasets:
        data = pd.read_csv("datasets\\oneshot\\Folds prepared\\"+dataset+"_fold_1.csv")
        d_df = data.fillna(data.mean())
        n_candidates = d_df.iloc[0]["NumberOfCandidates"]
        actions_table = pd.read_csv("datasets/oneshot/action_table_N" + str(n_candidates) + ".csv")
        scenarios_table = pd.read_csv("datasets/oneshot/scenario_table_N" + str(n_candidates) + ".csv")
        folds = list()
        folds.append(d_df.loc[d_df["SET"] == "TEST"].RoundIndex)
        d_df = d_df.drop(["SET"], axis = 1)

        # raw_data = d_df.copy()
        # d_performance_df, d_pred, d_feature_importance = _evaluation(raw_data, [baseline_clf], 'Action',
        #                                                                                   folds, actions_table,
        #                                                                                   scenarios_table, dataset,
        #                                                                                   n_candidates)
        # d_performance_df.to_csv("Results\\baseline_" + dataset + "_performance.csv")
        # d_pred.to_csv("Results\\baseline_" + dataset + "_pred.csv")
        # d_feature_importance.to_csv("Results\\baseline_" + dataset + "_feature_importance.csv")

        # RF_300 (40)
        # if dataset == "d32_updated_test_eval":
        raw_data = d_df.copy()
        d_performance_df, d_pred, d_feature_importance = _evaluation(raw_data, clfs, 'Action',
                                                                                          folds, actions_table,
                                                                                          scenarios_table, dataset,
                                                                                          n_candidates, top_k_fs=top_k)
        d_performance_df.to_csv("Results\\RAW_BASED_MODELS_" + dataset + "_performance.csv")
        d_pred.to_csv("Results\\RAW_BASED_MODELS_" + dataset + "_pred.csv")
        d_feature_importance.to_csv("Results\\RAW_BASED_MODELS_" + dataset + "_feature_importance.csv")

        # # Gradient Boosting (200)
        #
        # if dataset == "d36_updated_test_eval":
        #     raw_data = d_df.copy()
        #     d_performance_df, d_pred, d_feature_importance = _evaluation(raw_data, [gb_clf], 'Action',
        #                                                                                       folds, actions_table,
        #                                                                                       scenarios_table, dataset,
        #                                                                                       n_candidates, top_k_fs= 200)
        #     d_performance_df.to_csv("Results\\gb_100_" + dataset + "_performance.csv")
        #     d_pred.to_csv("Results\\gb_100_" + dataset + "_pred.csv")
        #
        # # RF_400 (200)
        #
        # if dataset == "tal_test_eval":
        #     raw_data = d_df.copy()
        #     d_performance_df, d_pred, d_feature_importance = _evaluation(raw_data, [rf_clf6], 'Action',
        #                                                                                       folds, actions_table,
        #                                                                                       scenarios_table, dataset,
        #                                                                                       n_candidates, top_k_fs=200)
        #     d_performance_df.to_csv("Results\\rf_400_" + dataset + "_performance.csv")
        #     d_pred.to_csv("Results\\rf_400_" + dataset + "_pred.csv")
        #
        # # NN_50 (100)
        #
        # if dataset == "schram_test_eval":
        #     raw_data = d_df.copy()
        #     d_performance_df, d_pred, d_feature_importance = _evaluation(raw_data, [neural_net_cf], 'Action',
        #                                                                  folds, actions_table,
        #                                                                  scenarios_table, dataset,
        #                                                                  n_candidates, top_k_fs=100)
        #     d_performance_df.to_csv("Results\\NN_50_" + dataset + "_performance.csv")
        #     d_pred.to_csv("Results\\NN_50_" + dataset + "_pred.csv")


#---------------------------------- Classifiers Definition ------------------------------------#

#---------------------------------- Classifiers Definition ------------------------------------#
#----------------------------------- Dataset definition ---------------------------------------#
# datasets: ["schram"]#["d36_2_folds","d36_4_folds","d36_6_folds","d32_2_folds","d32_4_folds","d32_6_folds"]
# datasets = ["schram"]
# n_candidates = 3
#
# _load_and_run(datasets=datasets, load_folds=True, classifiers=classifiers, n_candidates=n_candidates)
#
# datasets = ["N4_first_90_train"] #['voter_sample_for_test']#["tal_train","d32_updated_train","schram_train","N4_first_90"]#['voter_sample_for_test']#["schram_train","tal_train","d36_updated_train","d32_updated_train","N4_first_90_train"] #["N4_first_90", "d32_updated", "d36_updated", "tal", "schram"]#["N4_first_90_sample", "d32_updated_sample", "d36_updated_sample", "tal_sample", "schram_sample"]#["N4_first_90", "d32_updated", "d36_updated", "tal", "schram"]
# fold_set = [10]#, 10]
# _load_and_run(datasets=datasets, load_folds=False,fold_set=fold_set)
# #
#
# datasets = ['voter_sample_for_test']#["schram"]#["N4_first_90", "d32_updated", "d36_updated", "tal", "schram", "N4_first_90_train", "d32_updated_train", "d36_updated_train", "tal_train", "schram_train"]
# for dataset in datasets:
#     file_path = "datasets/oneshot/PartionedDatasets/Original/" + dataset + ".xlsx"
#     xls = pd.ExcelFile(file_path)
#     for sheet in xls.sheet_names:
#         #Get sheet from xlsx
#         data = pd.read_excel(file_path, sheet_name=sheet)
#         data_train, data_test = train_test_split(data, random_state=1, test_size=0.2)
#         data_train.to_excel("datasets\\oneshot\\PartionedDatasets\\" + dataset + "_train.xlsx")
#         data_test.to_excel("datasets\\oneshot\\PartionedDatasets\\" + dataset + "_test.xlsx")
#

 #HERE CUSTOM CODE STARTS!


# datasets = ["schram_train"]#,"N4_first_90_train"]
#
# for k_top in [150,200]:
#     _evaluation_by_files(datasets, "Action", top_k_fs=k_top, select_features=True)
#






# datasets = ["N4_first_90_train"] #"schram_train", "tal_train", "d32_updated_train", "d36_updated_train",
# _load_and_run(datasets, False, k_folds=20)
#datasets = ["N4_first_90_test_eval"]
# _test_eval(datasets, top_k=20)

datasets = ["D32", "D36", "TMG15", "TS16", "D32M4"]
_load_and_run(datasets, False)
# datasets = ["d32_updated_test_eval","schram_test_eval", "tal_test_eval", "d32_updated_test_eval", "d36_updated_test_eval", "N4_first_90_test_eval"]
# _test_eval(datasets, top_k=None)
#

# datasets = ["N4_first_90", "schram","d32_updated","d36_updated","tal"]
#
# for dataset in datasets:
#     file_path = "datasets/oneshot/Original/" + dataset + ".xlsx"
#     xls = pd.ExcelFile(file_path)
#     for sheet in xls.sheet_names:
#         # Get sheet from xlsx
#         data = pd.read_excel(file_path, sheet_name=sheet)
#
#         n_candidates = data.iloc[0]["NumberOfCandidates"]
#         action_table_df = pd.read_csv("datasets/oneshot/action_table_N" + str(n_candidates) + ".csv")
#         scenarios_df = pd.read_csv("datasets/oneshot/scenario_table_N" + str(n_candidates) + ".csv")
#
#         oneshot_static_fg = OneShotStaticFeatureGenerator(action_table_df, scenarios_df, n_candidates)
#
#         data = oneshot_static_fg._static_feature_generation(data)
#
#         data.to_csv(dataset + "StaticFeatures.csv")