import numpy as np
import pandas as pd
import sklearn

from sklearn import preprocessing




def _data_conversion(data_df, is_target, le):
    if is_target:
        data_df = data_df.astype("category")
        data_df = le.fit_transform(data_df)
    else:
        for c in data_df.columns:
            if data_df[c].dtype in (object, str, np.object, bool):
                if not (data_df[c].dtype in (int, float)):
                    data_df.loc[:,c] = le.fit_transform(data_df.loc[:,c])
    return data_df

class OneShotDataPreparation():
    """ Class for One Shot data preparation

    """
    @staticmethod
    def _prepare_dataset(features_df):
        le = sklearn.preprocessing.LabelEncoder()
        features_encoded_df = pd.DataFrame(
            preprocessing.normalize(preprocessing.scale(_data_conversion(features_df, False, le).values), axis=0,
                                    norm='max'))

        # target_le = sklearn.preprocessing.LabelEncoder()
        # target_df = _data_conversion(target_df, True, target_le)

        return features_encoded_df#, target_df















