import sklearn
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from OrdinalClassifier import OrdinalClassifier
import numpy as np

def SVC_factory():
    return SVC(probability=True)

def data_cleaning(data_df, le):
    for c in data_df.columns:
        if data_df[c].dtype in (float, int, np.int64, str, np.object):
            data_df[c] = data_df[c].replace(to_replace=np.nan, value=0, regex=True)
            data_df[c] = le.fit_transform(data_df[c])
        if data_df[c].dtype in (object, str):
            for row in data_df[c]:
                if isinstance(row, float):
                    data_df[c] = data_df[c].replace(to_replace=np.nan, value="Null", regex=True)
                    data_df[c] = data_df[c].astype("category")
                    data_df[c] = le.fit_transform(data_df[c])

    return data_df

def clean_data(X,le=sklearn.preprocessing.LabelEncoder()):
    df = DataFrame(X, columns=None)
    return data_cleaning(df, le).as_matrix()

ordered_classes = ["NONE", "ARAD", "SILVER", "GOLD"]
# class_value = "milt"
# y = ["cold","cold","milt","cold","hot","milt","hot","hot"]
# X = [[1,1,1],[2,2,1],[2,1,2],[2,3,1],[3,2,3],[2,4,2],[3,3,3],[3,3,2]]


# headers = ["COUNTRY","SPORT","IS STUDENT"]
X = [
    [0, 0, 0],
    [0, 1, 1],
    [0, 1, 0],
    [0, 2, 1],
    [0, 2, 0],
    [1, 0, 1],
    [1, 0, 0],
    [1, 1, 1],
    [1, 2, 1],
    [1, 2, 0],
    [2, 0, 1],
    [2, 0, 0],
    [2, 1, 0],
    [2, 2, 1],
    [2, 2, 0]
]

x_test = [
    [0, 0, 1],
    [1, 1, 0],
    [2, 1, 1],
]

# X = [
#     ["ISR", "FOOTBALL", "NO"],
#     ["ISR", "BASKETBALL", "YES"],
#     ["ISR", "BASKETBALL", "NO"],
#     ["ISR", "CHESS", "YES"],
#     ["ISR", "CHESS", "NO"],
#     ["USD", "FOOTBALL", "YES"],
#     ["USD", "FOOTBALL", "NO"],
#     ["USD", "BASKETBALL", "YES"],
#     ["USD", "CHESS", "YES"],
#     ["USD", "CHESS", "NO"],
#     ["FRA", "FOOTBALL", "YES"],
#     ["FRA", "FOOTBALL", "NO"],
#     ["FRA", "BASKETBALL", "NO"],
#     ["FRA", "CHESS", "YES"],
#     ["FRA", "CHESS", "NO"]
# ]
#
y = ["NONE",
     "NONE",
     "ARAD",
     "ARAD",
     "GOLD",
     "NONE",
     "NONE",
     "SILVER",
     "ARAD",
     "SILVER",
     "ARAD",
     "GOLD",
     "ARAD",
     "NONE",
     "NONE"]

# x_test = [
#     ["ISR", "FOOTBALL", "YES"],
#     ["USD", "BASKETBALL", "NO"],
#     ["FRA", "BASKETBALL", "YES"],
# ]
# clean_training = clean_data(X)
clean_training = X
# clean_test = clean_data(x_test)
clean_test = x_test

classifier_tree = OrdinalClassifier(base_classifier=DecisionTreeClassifier, ordered_classes=ordered_classes)
classifier_svc = OrdinalClassifier(base_classifier=SVC_factory, ordered_classes=ordered_classes)

classifier_tree.fit(clean_training, y)
classifier_svc.fit(clean_training, y)

# x_test = [[2,2,1],[2,1,2],[3,3,3],[3,2,2],[1,2,1],[1,3,2]]


tree_prediciton_results = classifier_tree.predict(clean_test)
svc_prediciton_results = classifier_svc.predict(clean_test)

print("TREE: %d", tree_prediciton_results)
print("SVC: %d", svc_prediciton_results)
print("Goal: None,Gold,None")

print("TEST END")

for i in range(0,len(ordered_classes)-1):
    classifier_tree.print_tree(index=i,out_file="tree%d.dot"%i)




