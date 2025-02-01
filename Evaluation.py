import numpy as np
import pandas as pd
import sklearn
from datetime import datetime

# Classification models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from OrdinalClassifier import OrdinalClassifier
from OneShotFeatureGenerator import OneShotFeatureGenerator

# Model and feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2

# Classification metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# SVC classifier generator
def SVC_factory(C=1.0,kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=True,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape=None,
                 random_state=None):

    return SVC(C=C,kernel=kernel, degree=degree, gamma=gamma,
                 coef0=coef0, shrinking=shrinking, probability=probability,
                 tol=tol, cache_size=cache_size, class_weight=class_weight,
                 verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                 random_state=random_state)

def RandomForestClassifier_factory(n_estimators=50,criterion="gini",max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.,
                 max_features="auto",max_leaf_nodes=None,min_impurity_split=1e-7,bootstrap=True,oob_score=False, n_jobs=1,random_state=None,
                 verbose=0, warm_start=False, class_weight=None):
    return RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_split=min_impurity_split,bootstrap=bootstrap,oob_score=oob_score, n_jobs=n_jobs,random_state=random_state,
                 verbose=verbose, warm_start=warm_start, class_weight=class_weight)


# DecisionTree classifier generator
def DecisionTreeClassifier_factory(criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_split=1e-7, class_weight=None, presort=False):

    return DecisionTreeClassifier(criterion=criterion,
                 splitter=splitter,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 random_state=random_state,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_split=min_impurity_split,
                 class_weight=class_weight,
                 presort=presort)

# Logistic Regression classifier generator
def LogisticRegressionClassifier_factory(penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

    return LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)


def data_cleaning(data_df,target_feature,le, target_le):
    for c in data_df.columns:
        if data_df[c].dtype in (float, int, np.int64, str, np.object):
            data_df[c] = data_df[c].replace(to_replace=np.nan, value=0, regex=True)
            if(c == target_feature):
                data_df[c] = target_le.fit_transform(data_df[c])
            else:
                data_df[c] = le.fit_transform(data_df[c])
        if data_df[c].dtype in (object, str):
            for row in data_df[c]:
                if isinstance(row, float):
                    data_df[c] = data_df[c].replace(to_replace=np.nan, value="Null", regex=True)
                    data_df[c] = data_df[c].astype("category")
                    if (c == target_feature):
                        data_df[c] = target_le.fit_transform(data_df[c])
                    else:
                        data_df[c] = le.fit_transform(data_df[c])

    return data_df.drop([target_feature], axis=1),data_df[target_feature]

def feature_selection(features_df, target_df, score_func, n): #Return top n features
    kbest = SelectKBest (score_func=score_func, k=n)
    kf = KFold(10, shuffle = True)  # 10 fold cross validation
    for train_indices, test_indices in kf.split(features_df, target_df):
        # make training and testing datasets
        features_train = features_df.loc[[ii for ii in train_indices],]
        targets_train = target_df[[ii for ii in train_indices]]
        # Feature select phase HERE
        kbest.fit(features_train, targets_train)

    scores = pd.DataFrame(kbest.scores_)
    features = pd.DataFrame(np.asarray(features_df.columns), columns=["Feature"])
    features['Score'] = scores
    top_features = features.sort_values(by=["Score"], ascending=False)
    top_features = (top_features[0:n]).Feature

    return top_features

def build_model(features_df, target_df, clf):
    # features_df = train_features
    # target_df = train_target
    # Use a 10-cross validation to build a model
    kf = KFold(10, shuffle = True)  # 10 fold cross validation
    for train_indices, test_indices in kf.split(features_df, target_df):
        # make training and testing datasets
        features_train = features_df.loc[[ii for ii in train_indices],]
        targets_train = target_df[[ii for ii in train_indices]]
        # Train
        clf.fit(features_train.as_matrix(), targets_train.as_matrix())
    return clf

def performance(features_df, target_df, clf):
    n_folds = 10
    results_df = pd.DataFrame(columns=['Measure', 'Result'])
    # Initialize metrics:
    results_df.loc[0] = ['ACCURACY', 0]
    results_df.loc[1] = ['PRECISION', 0]
    results_df.loc[2] = ['RECALL', 0]
    results_df.loc[3] = ['F_MEASURE', 0]
    results_df.loc[4] = ['TRAIN_TIME', 0]
    # 10 fold cross validation
    kf = KFold(n_folds,shuffle=True,random_state=1) # 10 fold cross validation
    for train_indices, test_indices in kf.split(features_df, target_df):
        # make training and testing datasets
        features_train = features_df.loc[[ii for ii in train_indices],]
        features_test = features_df.loc[[ii for ii in test_indices],]
        targets_train = target_df[[ii for ii in train_indices]]
        targets_test = target_df[[ii for ii in test_indices]]
        # Train
        train_time = datetime.now()
        clf.fit(features_train.as_matrix(), targets_train.as_matrix())
        train_time = datetime.now() - train_time
        # Test
        predicated = clf.predict(features_test.as_matrix())

        # Measures
        results_df.iloc[0, 1] = results_df.iloc[0, 1] + accuracy_score(targets_test, predicated)
        results_df.iloc[1, 1] = results_df.iloc[1, 1] + precision_score(targets_test, predicated, average='weighted')
        results_df.iloc[2, 1] = results_df.iloc[2, 1] + recall_score(targets_test, predicated, average='weighted')
        results_df.iloc[3, 1] = results_df.iloc[3, 1] + f1_score(targets_test, predicated, average='weighted')
        results_df.iloc[4, 1] = results_df.iloc[4, 1] + train_time.microseconds

    results_df.Result = results_df.Result.apply(lambda x: x / n_folds)
    return results_df

def evaluation(data_df, target_feature,le, target_le,dataset,n_feature_selection=-1,ordered_class=None):
    # Data cleaning
    train_features, train_target = data_cleaning(data_df,target_feature, le, target_le)

    # feature selection - TOP 15
    if n_feature_selection > 0:
        train_features = train_features[feature_selection(train_features, train_target, chi2, n_feature_selection)]
        print("done running feature selection")


    # feature generation - Aggregated features.
    train_features = OneShotFeatureGenerator.generate_A_ratios(train_features)
    train_features = OneShotFeatureGenerator.generate_voter_type(train_features)
    train_features = OneShotFeatureGenerator.generate_is_random_voter(train_features)
    print("done running feature generation")




    #Performance

    filename = "%s_result.csv" % dataset
    f = open(filename,"w")
    f.write("id,dataset,alogrithm,ACCURACY,PRECISION,RECALL,F_MEASURE,TRAIN_TIME\n")

    id = 0
    atom_classifiers = [RandomForestClassifier_factory, DecisionTreeClassifier_factory, LogisticRegressionClassifier_factory]
    for classifier in atom_classifiers:
        clf = classifier()
        atom_results = performance(train_features,train_target,clf)

        result_list = list(list(atom_results.to_dict().values())[0].values())
        for i in range(len(result_list)):
            result_list[i] = str(result_list[i])
        algorithm = "regular classifier = %s" % (str(classifier))
        f.write("%d,%s,%s,%s\n" % (id, dataset, algorithm, str(','.join(result_list))))

        id += 1

        ordinal_clf = OrdinalClassifier(base_classifier=classifier, ordered_class=ordered_class)
        ordinal_results = performance(train_features,train_target,ordinal_clf)

        result_list = list(list(ordinal_results.to_dict().values())[0].values())
        for i in range(len(result_list)):
            result_list[i] = str(result_list[i])
        algorithm = "ordinal classifier = %s" % (str(classifier))
        f.write("%d,%s,%s,%s\n" % (id, dataset, algorithm, str(','.join(result_list))))

        id += 1

    f.close()

def runExp(path,target_feature,dataset,delimiter_data=';',ordered_class=None):
    # Initialize
    print("starting running on %s" %path)
    data_df = pd.DataFrame()
    le = sklearn.preprocessing.LabelEncoder()
    target_le = sklearn.preprocessing.LabelEncoder()
    ordered_class = target_le.fit_transform(ordered_class)
    # End Initialize
    print("reading the data")
    # Load dataset from file
    data_df = pd.read_csv(path,delimiter=delimiter_data)
    # Evaluate
    print("running experiment")
    if len(data_df.columns) > 15:
        evaluation(data_df, target_feature,le,target_le,dataset,15,ordered_class)
    else:
        evaluation(data_df, target_feature,le,target_le,dataset,ordered_class=ordered_class)
    print("done")

#Data sets
ordered_class = ["Q","Q'","Q''"]
runExp('datasets/oneshot/oneshot.csv',"Action_pref","oneshot",',',ordered_class)


