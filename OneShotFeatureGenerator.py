import numbers
from OneShotDataPreperation import OneShotDataPreparation
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.feature_selection import RFE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from keras.layers import Input, Dense
from keras.models import Model

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


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
from sklearn.ensemble import RandomForestRegressor


def _autoencode(features):
    # test

    encoding_dim = int(len(features.columns) / 5)
    input_votes = Input(shape=(len(features.columns),))
    encoded = Dense(encoding_dim, activation='relu')(input_votes)
    decoded = Dense(len(features.columns), activation='tanh')(encoded)
    autoencoder = Model(input_votes, decoded)
    encoder = Model(input_votes, encoded)

    #   encoded_input = Input(shape=(encoding_dim,))
    #    decoder_layer = autoencoder.layers[-1]
    #   decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='MSE')

    autoencoder.fit(features, features,
                    epochs=20, #tried 20
                    batch_size=256,
                    shuffle=True, verbose=False)

    encoded_votes = encoder.predict(features)

    return encoded_votes

class OneShotFeatureGenerator():
    """ Class for One Shot feature generation

    """

    def __init__(self,
                 actions_df,
                 scenarios_df,
                 n_candidates):
        self.actions_df = actions_df
        self.scenarios_df = scenarios_df
        self.n_candidates = n_candidates

    def _get_actions(self):
        return ['TRT','WLB','SLB','CMP','DOM']

    def _get_strategic_actions(self):
        return ['WLB','SLB','CMP']

    def _get_preference_features(self):
        preference_features = ['Pref1','Pref2','Pref3']

        if self.n_candidates == 4:
            preference_features.append('Pref4')

        return preference_features

    def _get_gap_pref_features(self):
        feature12 = 'GAP12_pref_poll'
        feature23 = 'GAP23_pref_poll'
        feature13 = 'GAP13_pref_poll'

        features = [feature12, feature23, feature13]

        if self.n_candidates == 4:
            feature14 = 'GAP14_pref_poll'
            feature24 = 'GAP24_pref_poll'
            feature34 = 'GAP34_pref_poll'
            features.extend([feature14,feature24,feature34])

        return features

    def _get_scenarios_by_actions(self,actions):
        scenarios = set([])
        for action in actions:
            action_scenarios = self._get_scenarios_by_action(action)
            scenarios = scenarios.union(action_scenarios)

        return scenarios

    def _get_scenarios_by_action(self, action):
        scenarios = set([x[1].scenario for x in self.actions_df.iloc[[action in str(x) for x in self.actions_df['action_name']],].iterrows()])

        return scenarios

    # def _generate_action_name(self, df):
    #     # Generate action name
    #     df['Action_name'] = [self._get_action_name(df, x[0]) for x in
    #                       df.iterrows()]
    #
    #     return df

    def _get_action_name(self, vote_row):
        action_name = (self.actions_df.loc[(self.actions_df.scenario == vote_row['Scenario']) & (
            self.actions_df.action == int(vote_row['Action'])), 'action_name']).values[0]

        return action_name


    def _get_scenario(self, vote_row):
        scenario_table = self.scenarios_df


        pass



    def _convert_prediction(self, df):
        preference_features = self._get_preference_features()
        for preference_feature in preference_features:
            df.loc[df['Prediction'] == 1, "VotePrediction"] = df.loc[df['Prediction'] == 1, preference_feature]

        return df


class OneShotStaticFeatureGenerator(OneShotFeatureGenerator):
    """ Class for One Shot feature generation

    """

    def __init__(self,
                 actions_df,
                 scenarios_df,
                 n_candidates):
        super().__init__(actions_df, scenarios_df, n_candidates)


    def _generate_scenario(self, df):
        if self.n_candidates == 4:
            get_scenario = lambda vote, scenarios_table, attr : scenarios_table[
                (scenarios_table["Pref1_pos"] == vote["Pref1_pos"]) & (scenarios_table["Pref2_pos"] == vote["Pref2_pos"]) &
                (scenarios_table["Pref3_pos"] == vote["Pref3_pos"]) & (scenarios_table["Pref4_pos"] == vote[
                "Pref4_pos"])][attr].values[0]
        else:
            get_scenario = lambda vote, scenarios_table, attr : scenarios_table[
                (scenarios_table["Pref1_pos"] == vote["Pref1_pos"]) & (scenarios_table["Pref2_pos"] == vote["Pref2_pos"]) &
                (scenarios_table["Pref3_pos"] == vote["Pref3_pos"])][attr].values[0]

        df["Scenario"] = [get_scenario(vote[1], self.scenarios_df, "scenario") for vote in df.iterrows()]
        df["Scenario_type"] = [get_scenario(vote[1], self.scenarios_df, "name") for vote in df.iterrows()]

        return df

    def _generate_pref_positions(self, df):
        for vote in df.iterrows():
            pref_votes = [vote[1]["VotesCand" + str(vote[1]["Pref1"]) + "PreVote"],
                          vote[1]["VotesCand" + str(vote[1]["Pref2"]) + "PreVote"],
                          vote[1]["VotesCand" + str(vote[1]["Pref3"]) + "PreVote"]]
            prefs = [1,2,3]
            if self.n_candidates == 4:
                pref_votes.append(vote[1]["VotesCand" + str(vote[1]["Pref4"]) + "PreVote"])
                prefs.append((4))
            combined = pd.DataFrame({'votes': pref_votes, 'pref': prefs})
            combined = combined.sort_values(by="votes", ascending=0)
            combined = combined.reset_index(drop=True)

            for index in range(0, len(combined)):
                column_name = "Pref" + str(combined["pref"][index]) + "_pos"
                column_value = index + 1
                df.loc[vote[0],column_name] = int(column_value)


        return df


    def _generate_pref_gaps(self, df):
        preference_features = self._get_preference_features()
        for preference_feature in preference_features:
            df["Votes"+preference_feature+"PreVote"] = [x[1]["VotesCand" + str(x[1][preference_feature]) + "PreVote"] for x in df.iterrows()]

        return df

    def _generate_gaps(self, df):
        """Generate Gaps features"""
        X = df

        X['VotesLeader_poll'] = X[['VotesCand1PreVote', 'VotesCand2PreVote', 'VotesCand3PreVote']].max(axis=1)
        X['VotesRunnerup_poll'] = X[['VotesCand1PreVote', 'VotesCand2PreVote', 'VotesCand3PreVote']].apply(
            np.median, axis=1)
        X['VotesThird_poll'] = X[['VotesCand1PreVote', 'VotesCand2PreVote', 'VotesCand3PreVote']].min(axis=1)

        X['GAP12_poll'] = X['VotesLeader_poll'] - X['VotesRunnerup_poll']
        X['GAP23_poll'] = X['VotesRunnerup_poll'] - X['VotesThird_poll']
        X['GAP13_poll'] = X['VotesLeader_poll'] - X['VotesThird_poll']

        # Preference based gaps - I think more suitable for ML for it's more synchronized across the scenarios

        X['GAP12_pref_poll'] = X['VotesPref1PreVote'] - X['VotesPref2PreVote']
        X['GAP23_pref_poll'] = X['VotesPref2PreVote'] - X['VotesPref3PreVote']
        X['GAP13_pref_poll'] = X['VotesPref1PreVote'] - X['VotesPref3PreVote']

        #N=4 case
        if self.n_candidates == 4:
            X['VotesFourth_poll'] = X[['VotesCand1PreVote', 'VotesCand2PreVote', 'VotesCand3PreVote','VotesCand4PreVote']].min(axis=1)

            X['GAP14_poll'] = X['VotesLeader_poll'] - X['VotesFourth_poll']
            X['GAP24_poll'] = X['VotesRunnerup_poll'] - X['VotesFourth_poll']
            X['GAP34_poll'] = X['VotesThird_poll'] - X['VotesFourth_poll']

            X['GAP14_pref_poll'] = X['VotesPref1PreVote'] - X['VotesPref4PreVote']
            X['GAP24_pref_poll'] = X['VotesPref2PreVote'] - X['VotesPref4PreVote']
            X['GAP34_pref_poll'] = X['VotesPref3PreVote'] - X['VotesPref4PreVote']

        return X


    def _static_feature_generation(self, df):
        df = self._generate_pref_gaps(df)
        df = self._generate_gaps(df)
        df = self._generate_pref_positions(df)
        df = self._generate_scenario(df)

        return df


class OneShotDynamicFeatureGenerator(OneShotFeatureGenerator):
    """ Class for One Shot feature generation

    """

    def __init__(self,
                 actions_df,
                 scenarios_df,
                 n_candidates):
        super().__init__(actions_df, scenarios_df, n_candidates)


    def _count_action_for_voter(self, action, voter_df):
        action_counter = np.count_nonzero(([action in self._get_action_name(x[1]) for x in voter_df.iterrows()]))

        return action_counter

    def _generate_A_ratios(self, df, X_train, y_train ,voter_index):
        """Generate A ratios - That is TRT-ratio, CMP-ratio, WLB-ratio, SLB-ratio, DOM-ratio
            Action is in {TRT,DLB,SLB,WLB,CMP,DOM}
            Scenario is in {A,B,C,D,E,F}
        """

        voter_df = pd.concat([X_train.loc[X_train.index & voter_index], y_train], axis=1, join='inner')

        for action in self._get_actions():
            availability_counter = np.count_nonzero([x[1].Scenario in self._get_scenarios_by_action(action) for x in voter_df.iterrows()])
            action_counter = self._count_action_for_voter(action, voter_df)
            df.loc[voter_index, action + '-ratio'] = float(action_counter/availability_counter if availability_counter > 0 else 0)
            df.loc[voter_index, action + '-counter'] = float(action_counter)

        return df

    def _generate_is_random_voter(self, df):
        """Identify random voters using the rule of DOM-counter >= 2 (excluding SLB actions)"""
        df['Is_Random'] = [x >= 2 for x in df['DOM-counter']]

        return df

    def _generate_voter_type(self, df):
        """Generate Voter Type using thresholds over the A-ratio values"""
        df['VoterType'] = 'Other'
        # X.loc[ [int(x[1]['CMP-ratio'])>=0.7 for x in X.iterrows()], 'VoterType'] = 'CMP'
        df.loc[[float(x[1]['WLB-ratio']) > 0.8 for x in df.iterrows()], 'VoterType'] = 'LB'
        df.loc[[float(x[1]['TRT-ratio']) > 0.9 for x in df.iterrows()], 'VoterType'] = 'TRT'

        return df



    def _generate_feature_aggregation_class_dependant(self, df, X_train, y_train, scenarios, voter_index, feature_name, aggregation_func):

        X = df
        #X_train, y_train = X_train.loc[X_train['Scenario'].isin(scenarios)], y_train.loc[X_train['Scenario'].isin(scenarios)]

        #X_train, y_train = X_train, y_train #X.drop([self.target_index], axis=1),X[self.target_index]




        voter_train = X_train.loc[X_train.index & voter_index]
        voter_train = voter_train.loc[voter_train["Scenario"].isin(scenarios)]
        voter_targets = y_train.loc[voter_train.index]
        if len(voter_train) > 0:
            for action in range(1, self.n_candidates + 1):
                actioni_list = [float(x[1][feature_name]) for x in
                                voter_train.loc[voter_targets == action,:].iterrows()]
                if len(actioni_list) > 0:
                    X.loc[voter_index, feature_name + '_action'+ str(action) + '_' + aggregation_func.__name__] = aggregation_func(
                        actioni_list)
        return X

    def _generate_action_aggregation_features(self, df, X_train, y_train, voter_index):
        X = df

        aggregators = [np.average, np.std, np.median]
        feature_name = "Action"

        scenarios = self._get_scenarios_by_actions(self._get_strategic_actions())

        voter_train = X_train.loc[X_train.index & voter_index]
        voter_train = voter_train.loc[voter_train["Scenario"].isin(scenarios)]
        voter_targets = y_train.loc[voter_train.index]

        for aggregation_func in aggregators:
            X.loc[voter_index, feature_name + "_" + aggregation_func.__name__] = aggregation_func(
                [float(voter_targets[x[0]]) for x in voter_train.iterrows()])

        return X

    def _generate_gaps_features(self, df, X_train, y_train, voter_index):
        X = df

        features = self._get_gap_pref_features()
        aggregators = [np.average, np.std, np.median, np.min, np.max, skew, kurtosis]
        scenarios = self._get_scenarios_by_actions(self._get_strategic_actions())

        for aggregator in aggregators:
            for feature in features:
                X = self._generate_feature_aggregation_class_dependant(X, X_train, y_train, scenarios, voter_index, feature, aggregator)

        return X

    def _generate_gap_dif_features(self, df):

        X = df
        features = self._get_gap_pref_features()
        aggregators = [np.average, np.median, np.min, np.max]

        for action in range(1, self.n_candidates + 1):
            for feature in features:
                for aggregator in aggregators:
                    X[feature + '_action'+str(action)+'_' + aggregator.__name__ + '_dif'] = X[feature] - X[
                        feature + '_action'+str(action)+'_' + aggregator.__name__]

        return X



    def _dynamic_feature_generation(self, df, X_train, y_train):
        X = df
        a_ratio_columns, gaps_columns = [], []
        all_voters = pd.DataFrame(X["VoterID"].drop_duplicates())
        for voter in all_voters.iterrows():
            voter_index = X.loc[X['VoterID'] == voter[1].VoterID,].index
            before_columns = len(X.columns)
            X = self._generate_A_ratios(X, X_train, y_train, voter_index)
            if len(a_ratio_columns) == 0:
                a_ratio_columns = list(range(before_columns, len(X.columns)))

            before_columns = len(X.columns)
            X = self._generate_gaps_features(X, X_train, y_train, voter_index)
            if len(gaps_columns) == 0:
                gaps_columns = list(range(before_columns, len(X.columns)))


            X = self._generate_action_aggregation_features(X, X_train, y_train, voter_index)

        # Gaps features encoding
        X = X.fillna(
            X.mean())  # X.fillna(1000) #fill na with some high value (maybe maximum) because the voters with na values didn't choose the action (say q'', 3) in all gaps they incounterd.

        before_columns = len(X.columns)
        X = self._generate_gap_dif_features(X)
        gaps_dif_columns = list(range(before_columns, len(X.columns)))

        total_gaps_columns = a_ratio_columns + gaps_columns + gaps_dif_columns

        gap_pref_features = self._get_gap_pref_features()
        for gap_pref_feature in gap_pref_features:
            total_gaps_columns.append(X.columns.get_loc(gap_pref_feature))

        total_gaps_columns.append(X.columns.get_loc("Scenario"))
        total_gaps_columns.append(X.columns.get_loc("Scenario_type"))
        total_gaps_columns.append(X.columns.get_loc("VoterID"))

        normalized_gap_fs = pd.DataFrame(preprocessing.normalize(OneShotDataPreparation._prepare_dataset(X.iloc[:, total_gaps_columns])))

        encoded_gap_fs = pd.DataFrame(_autoencode(normalized_gap_fs))

        encoded_gap_fs.index = X.index
        X = pd.concat([X, encoded_gap_fs], axis=1, join='inner')


        # #Try auto encode each voter separately
        # # encoded_gap_fs = pd.DataFrame()
        # #
        # # for voter in all_voters.iterrows():
        # #     voter_index = X.loc[X['VoterID'] == voter[1].VoterID].index
        # #     voter_encoded_gap_fs = pd.DataFrame(_autoencode(normalized_gap_fs.iloc[voter_index.tolist(),:]))
        # #     voter_encoded_gap_fs.index = voter_index
        # #
        # #     # aggregate results
        # #     if len(encoded_gap_fs) == 0:
        # #         encoded_gap_fs = pd.DataFrame(voter_encoded_gap_fs)
        # #     else:
        # #         encoded_gap_fs = pd.concat([encoded_gap_fs, pd.DataFrame(voter_encoded_gap_fs)])
        # #
        # # encoded_gap_fs = pd.DataFrame(encoded_gap_fs)
        # #
        # # X = pd.concat([X, encoded_gap_fs], axis=1, join='inner')
        #


        #X = X.drop(X.columns[gaps_columns + gaps_dif_columns], axis=1)

        X = self._generate_is_random_voter(X)
        X = self._generate_voter_type(X)

        # plt.figure(figsize=(12, 10))
        # cor = df.corr()
        # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        # plt.show()

        # Correlation with output variable
        # cor_target = abs(pd.concat([X.loc[X_train.index], y_train], axis=1, join='inner').corr()["Action"])
        # # Selecting highly correlated features
        # relevant_features = cor_target[cor_target > 0.4]
        # print(relevant_features)
        #


        return X







