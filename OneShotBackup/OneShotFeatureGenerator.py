from OneShotDataPreperation import OneShotDataPreparation
import numpy as np
import pandas as pd
import math
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.feature_selection import RFE
from scipy import spatial
from sklearn.ensemble.forest import RandomForestClassifier
from keras.layers import Input, Dense
from keras.models import Model
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from sklearn import preprocessing

def _autoencode(X):
    # test
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    encoding_dim = 6#int(len(X.columns) / 5)
    input_votes = Input(shape=(len(X.columns),))
    encoded = Dense(encoding_dim, activation='relu')(input_votes)
    decoded = Dense(len(X.columns), activation='tanh')(encoded)
    autoencoder = Model(input_votes, decoded)
    encoder = Model(input_votes, encoded)

    #   encoded_input = Input(shape=(encoding_dim,))
    #    decoder_layer = autoencoder.layers[-1]
    #   decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='MSE')

    autoencoder.fit(scaled_X, scaled_X,
                    epochs=300,
                    batch_size=256,
                    shuffle=True, verbose=False)

    encoded_votes = encoder.predict(scaled_X)

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

        #build scenarios by action
        actions = self._get_actions()
        self.scenarios_by_action = {action: self._get_scenarios_by_action(action) for action in actions}

        self.action_names = {action: {scenario: self._get_action_name(scenario, action) for scenario in range(1, math.factorial(self.n_candidates) + 1)} for action in range(1, self.n_candidates + 1)}

    def _get_features_by_scenario_type(self, scenario_type):
        features = list()
        if self.n_candidates == 3:
            if scenario_type == 'TRT':
                features = ["GAP12_pref_poll","GAP13_pref_poll"]
            if scenario_type == 'WLB':
                features = ["GAP12_pref_poll"]
            if scenario_type == 'SLB':
                features = ["GAP13_pref_poll"]
            if scenario_type == 'CMPLB':
                features = ["GAP12_pref_poll"]
            if scenario_type == 'CMP':
                features = ["GAP12_pref_poll","GAP23_pref_poll","GAP13_pref_poll"]
        else:
            if self.n_candidates == 4:
                if scenario_type == 'TRT':
                    features = ["GAP12_pref_poll","GAP13_pref_poll","GAP14_pref_poll"]
                if scenario_type == 'WLB':
                    features = ["GAP12_pref_poll"]
                if scenario_type == 'LB':
                    features = ["GAP13_pref_poll"]
                if scenario_type == 'SLB':
                    features = ["GAP14_pref_poll"]
                if scenario_type == 'CMP':
                    features = ["GAP12_pref_poll","GAP24_pref_poll","GAP14_pref_poll"]
                if scenario_type == 'SCMP':
                    features = ["GAP13_pref_poll","GAP34_pref_poll","GAP14_pref_poll"]
        return features

    def _get_actions_by_gap_feature(self, gap_feature):
        first_action = gap_feature[3]
        second_action = gap_feature[4]
        return [int(first_action), int(second_action)]

    def _get_aggregators_by_action(self, gap_actions, action):
        aggregators = [np.average, np.median, np.std, len]
        if action == gap_actions[0]:
            aggregators.extend([np.min])
        elif action == gap_actions[1]:
            aggregators.extend([np.max])
        else:
            aggregators = []

        return aggregators

    def _get_gap_feature_actions(self, vote, gap_feature):
        first_action = gap_feature[3]
        if first_action == "L":
            first_action = vote["Pos1_pref"]
        second_action = gap_feature[4]
        return [int(first_action), int(second_action)]

    def _get_relevant_scenarios(self, scenario_type):
        scenarios = list()
        if self.n_candidates == 3:
            if scenario_type == 'TRT':
                scenarios = [1, 2]
            if scenario_type == 'WLB':
                scenarios = [3, 5, 6]
            if scenario_type == 'SLB':
                scenarios = [4, 6]
            if scenario_type == 'CMPLB':
                scenarios = [3, 5, 6]
            if scenario_type == 'CMP':
                scenarios = [3, 4, 5, 6]
        else:
            if self.n_candidates == 4:
                if scenario_type == 'TRT':
                    scenarios = [1, 2, 3, 4, 5, 6]
                if scenario_type == 'WLB':
                    scenarios = [7, 8, 9, 10, 11, 12, 21, 22]
                if scenario_type == 'LB':
                    scenarios = [13, 14, 15, 16, 17, 18, 23, 24]
                if scenario_type == 'SLB':
                    scenarios = [19, 20, 21, 22, 23, 24]
                if scenario_type == 'CMP':
                    scenarios = [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24]
                if scenario_type == 'SCMP':
                    scenarios = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        return scenarios

    def _get_featuresfamilies_by_scenario_type(self, scenario_types):
        features_families = list()
        features_families.extend(["Action"])

        if "TRT" in scenario_types:
            features_families.extend(["DOM"])
        if "WLB" in scenario_types:
            features_families.extend(["GAP12_pref_poll", "GAP23_pref_poll"])
        if "SLB" in scenario_types:
            features_families.extend(["GAP13_pref_poll", "GAP23_pref_poll"])
        if "CMPLB" in scenario_types:
            features_families.extend(["GAP12_pref_poll", "GAP23_pref_poll"])
        if "CMP" in scenario_types:
            features_families.extend(["GAP13_pref_poll", "GAP23_pref_poll"])

        return features_families

    def _get_actions(self):
        if self.n_candidates == 3:
            actions = ['TRT','WLB','SLB','CMP','DOM']
        else:
            if self.n_candidates == 4:
                actions = ['TRT','WLB','LB', 'SLB', 'CMP', 'SCMP', 'DOM']
        return actions

    def _get_strategic_actions(self):
        return ['WLB','SLB','CMP']

    def _get_preference_features(self):
        preference_features = ['Pref1','Pref2','Pref3']

        if self.n_candidates == 4:
            preference_features.append('Pref4')

        return preference_features

    def _get_gap_feature_actions(self, vote, gap_feature):
        first_action = gap_feature[3]
        if first_action == "L":
            first_action = vote["Pos1_pref"]
        second_action = gap_feature[4]
        return [int(first_action), int(second_action)]

    def _get_gap_pref_features(self):
        feature12 = 'GAP12_pref_poll'
        feature23 = 'GAP23_pref_poll'
        feature13 = 'GAP13_pref_poll'
        # featureLeader1 = 'GAPL1_Pref_poll'
        # featureLeader2 = 'GAPL2_Pref_poll'
        # featureLeader3 = 'GAPL3_Pref_poll'

        features = [feature12, feature23, feature13]#, featureLeader1, featureLeader2, featureLeader3]


        if self.n_candidates == 4:
            feature14 = 'GAP14_pref_poll'
            feature24 = 'GAP24_pref_poll'
            feature34 = 'GAP34_pref_poll'
            #featureLeader4 = 'GAPL4_Pref_poll'
            features.extend([feature14, feature24, feature34])#, featureLeader4])

        # if returnNorm:
        #     features_with_norm = list()
        #     for feature in features:
        #         #features_with_norm.append(feature)
        #         features_with_norm.append((feature + "_normalized_dif"))
        #     features = features_with_norm

        return features

    def _get_scenarios_by_actions(self,actions):
        scenarios = set([])
        for action in actions:
            action_scenarios = self._get_scenarios_by_action(action)
            scenarios = scenarios.union(action_scenarios)

        return scenarios

    def _get_scenario_types_by_actions(self,actions):
        scenario_types = set([])
        for action in actions:
            action_scenario_types = self._get_scenario_types_by_action(action)
            scenario_types = scenario_types.union(action_scenario_types)

        return scenario_types

    def _get_scenarios_by_action(self, action):
        scenarios = set([x[1].scenario for x in self.actions_df.iloc[[action in str(x) for x in self.actions_df['action_name']],].iterrows()])

        return scenarios

    def _get_rational_actions_by_scenario_type(self, scenario_type):
        scenarios = [str(x) for x in self.scenarios_df.loc[self.scenarios_df['name'] == scenario_type, "scenario"]]
        rational_actions = set([x[1].action for x in self.actions_df.loc[self.actions_df["scenario"].isin(scenarios),:].loc[self.actions_df["action_name"] != "DOM",:].iterrows()])

        return rational_actions


    def _get_scenario_types_by_action(self, action):
        scenario_types = set([x[1].name for x in self.actions_df.iloc[[action in str(x) for x in self.actions_df['action_name']],].iterrows()])

        return scenario_types

    def _get_action_name(self, scenario, action):
        action_name = (self.actions_df.loc[(self.actions_df.scenario == scenario) & (
            self.actions_df.action == int(action)), 'action_name']).values[0]

        return action_name

    def _convert_prediction(self, df):
        preference_features = self._get_preference_features()
        for preference_feature in preference_features:
            df.loc[df['Prediction'] == 1, "VotePrediction"] = df.loc[df['Prediction'] == 1, preference_feature]

        return df

    def _candidate_distance(self, vote, prefi, prefj):
        i_pos = int(vote["Pref" + str(int(prefi)) + "_pos"])
        j_pos = int(vote["Pref" + str(int(prefj)) + "_pos"])
        return np.abs(i_pos - j_pos)/(self.n_candidates)

    def _get_actions_normalized(self, inverse=False):
        action_scores = [1, 2, 3]
        if self.n_candidates == 4:
            action_scores.append(4)
        if inverse:
            action_scores.reverse()

        action_scores = action_scores / np.max(action_scores)
        return action_scores

    def _get_norm_of_gaps(self, vote):
        max_gap = -np.inf
        for i in range(1, self.n_candidates):
            for j in range(i + 1, self.n_candidates + 1):
                posi_votes = vote["VotesPref" + str(int(vote["Pos"+str(int(i))+"_pref"])) + "PreVote"]
                posj_votes = vote["VotesPref" + str(int(vote["Pos"+str(int(j))+"_pref"])) + "PreVote"]
                cur_gap = posi_votes - posj_votes
                if cur_gap > max_gap:
                    max_gap = cur_gap
        return max_gap

    def _get_compromise_score(self, vote, pref):
        pref1_votes = vote["VotesPref1PreVote"]

        posi_pref = int(pref)
        posi_votes = vote["VotesPref" + str(posi_pref) + "PreVote"]
        i_pos = vote["Pref" + str(posi_pref) + "_pos"]
        if i_pos == 1: #LEADER BIAS NOT COMPROMISE
            pos_score = 0
        else:
            pos_score = self._candidate_distance(vote, pref, 1)

        action_score = self._get_actions_normalized()[posi_pref - 1]
        gap_score = np.max([0, (posi_votes - pref1_votes)/self._get_norm_of_gaps(vote)])


        return action_score, gap_score, pos_score

    def _get_perfect_compromise_score(self, vote):
        perfect_pref = vote["Pos1_pref"]
        max_compromise_scores = self._get_compromise_score(vote=vote, pref=perfect_pref)
        for i in range(1, int(vote["Pref1_pos"])):
            cur_pref = vote["Pos"+str(i)+"_pref"]
            cur_scores = self._get_compromise_score(vote=vote, pref=cur_pref)
            if np.prod(cur_scores) > np.prod(max_compromise_scores):
                max_compromise_scores = cur_scores
                perfect_pref = cur_pref

        return max_compromise_scores

    def _get_principle_score(self, vote, action):
        #Measure for how truthful\principled voter is. The most principled will vote q in extreme situations where q is the last in poll and the gap between q to leader is high.
        chosen_votes = vote["VotesPref" + str(action) + "PreVote"]
        leader_votes = vote["VotesLeader_poll"]
        gap_score = (leader_votes - chosen_votes) / self._get_norm_of_gaps(vote)
        action_score = self._get_actions_normalized(inverse=True)[action - 1]
        pos_score = self._candidate_distance(vote, action, int(vote["Pos1_pref"]))

        return action_score, gap_score, pos_score

    def _get_perfect_principle_score(self, vote):
        #Measure for how truthful\principled voter is. The most principled will vote q in extreme situations where q is the last in poll and the gap between q to leader is high.
        return self._get_principle_score(vote=vote, action=1)

    def _get_leaderbias_score(self, vote, action):
        actions = self._get_actions_normalized(inverse=False) #Inceasing
        pref1_votes = vote["VotesPref1PreVote"]
        pos_chosen_pref = int(vote["Pref" + str(action) + "_pos"])
        chosen_action_votes = vote["VotesPref" + str(action) + "PreVote"]
        if pos_chosen_pref == 1: #consider only leader choices.
                chosen_action_score = actions[int(action)-1]
        else:
                chosen_action_score = 0

        gap_score = (chosen_action_votes - pref1_votes)/self._get_norm_of_gaps(vote)
        pos_score = self._candidate_distance(vote, action, 1)

        return chosen_action_score, gap_score, pos_score

    def _get_perfect_leaderbias_score(self, vote):
        posleader_pref = int(vote["Pos1_pref"])
        return self._get_leaderbias_score(vote=vote, action=posleader_pref)

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
                prefs.append(4)
            combined = pd.DataFrame({'votes': pref_votes, 'pref': prefs})
            combined = combined.sort_values(by="votes", ascending=0)
            combined = combined.reset_index(drop=True)

            for index in range(0, len(combined)):
                column_name = "Pref" + str(combined["pref"][index]) + "_pos"
                column_value = index + 1
                df.loc[vote[0],column_name] = int(column_value)
                #New added
                column_name = "Pos" + str(index + 1) + "_pref"
                column_value = combined["pref"][index]
                df.loc[vote[0], column_name] = int(column_value)


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

    def _generate_gaps_to_leader(self, df):
        """Generate Gaps features"""
        X = df
        X['GAPL1_Pref_poll'] = X['VotesLeader_poll'] - X['VotesPref1PreVote']
        X['GAPL2_Pref_poll'] = X['VotesLeader_poll'] - X['VotesPref2PreVote']
        X['GAPL3_Pref_poll'] = X['VotesLeader_poll'] - X['VotesPref3PreVote']

        # N=4 case
        if self.n_candidates == 4:
            X['GAPL4_Pref_poll'] = X['VotesLeader_poll'] - X['VotesPref4PreVote']

        return X

    def _generate_pos(self, df):
        """Generate Pos features"""
        X = df

        X['Pos12'] = X['Pref1_pos'] - X['Pref2_pos']
        X['Pos23'] = X['Pref2_pos'] - X['Pref3_pos']
        X['Pos13'] = X['Pref1_pos'] - X['Pref3_pos']

        #N=4 case
        if self.n_candidates == 4:
            X['Pos14'] = X['Pref1_pos'] - X['Pref4_pos']
            X['Pos24'] = X['Pref2_pos'] - X['Pref4_pos']
            X['Pos34'] = X['Pref3_pos'] - X['Pref4_pos']

        return X

    def _generate_pos_to_leader(self, df):
        """Generate Pos features"""
        X = df
        X['PosL1'] = 1 - X['Pref1_pos']
        X['PosL2'] = 1 - X['Pref2_pos']
        X['PosL3'] = 1 - X['Pref3_pos']

        # N=4 case
        if self.n_candidates == 4:
            X['PosL4'] = 1 - X['Pref4_pos']

        return X

    def _static_feature_generation(self, df):
        df = self._generate_pref_gaps(df)
        df = self._generate_gaps(df)
        df = self._generate_pref_positions(df)
        df = self._generate_scenario(df)
        df = self._generate_gaps_to_leader(df)
        df = self._generate_pos(df)
        df = self._generate_pos_to_leader(df)

        return df



class OneShotDynamicFeatureGenerator(OneShotFeatureGenerator):
    """ Class for One Shot feature generation

    """

    def __init__(self,
                 actions_df,
                 scenarios_df,
                 n_candidates, X_train):
        super().__init__(actions_df, scenarios_df, n_candidates)

        all_voters = list(X_train["VoterID"].drop_duplicates())
        self.voters_index = {key: X_train.loc[X_train["VoterID"] == key,].index for key in all_voters}

        self.columns = {c: i for i, c in enumerate(X_train.columns)}

        scenario_types = self.scenarios_df["name"].drop_duplicates()
        self.aggregators = {scenario_type: {gap_feature: {action: self._get_aggregators_by_action(self._get_actions_by_gap_feature(gap_feature), action) for action in self._get_rational_actions_by_scenario_type(scenario_type)} for gap_feature in self._get_features_by_scenario_type(scenario_type)} for scenario_type in scenario_types}






    def _select_features(self, X, X_train, y_train, top_k=6):
        # selected_features = ["VoterID", "GameIndexInSession", "GAP12_pref_poll", "GAP23_pref_poll", "GAP13_pref_poll",
        #                      "CMP-ratio", "SLB-ratio", "GAP13_pref_poll_action2_amin_dif",
        #                      "GAP13_pref_poll_action1_amin_dif",
        #                      "PointsPreVote", "Action_average", "TRT-ratio", "GAP12_pref_poll_action1_amin_diff",
        #                      "WLB-ratio",
        #                      "GAP12_pref_poll_action1_amin", "GAP23_pref_poll_action1_amax_dif",
        #                      "GAP13_pref_poll_action1_amax_dif",
        #                      "GAP13_pref_poll_action2_amax_dif"]

        scenarios_types = self.scenarios_df["name"].drop_duplicates()
        total_selected = list()

        for scenario_type in scenarios_types:
            X_train_scenario = X_train.loc[X_train["Scenario_type"] == scenario_type]
            y_scenario_targets = y_train.loc[X_train_scenario.index]

            cols = list(X.columns)
            model = RandomForestClassifier(n_estimators=300, random_state=1)
            # Initializing RFE model
            rfe = RFE(model, top_k)
            X_rfe = rfe.fit_transform(OneShotDataPreparation._prepare_dataset(X.loc[X_train_scenario.index, :]), y_scenario_targets)
            # Fitting the data to model
            model.fit(X_rfe, y_scenario_targets)
            temp = pd.Series(rfe.support_, index=cols)
            selected_features_rfe = temp[temp == True].index.tolist()
            print(scenario_type + " : ")
            print(selected_features_rfe)
            total_selected += selected_features_rfe

        total_selected = list(set(total_selected))

        return total_selected

    def _compute_error(self, cur_range, voter_train, voter_targets, feature):
        error = 0
        actions = self._ordered_action(voter_train, feature)
        for i in range(0, len(voter_train)):
            vote = voter_train.iloc[i,:]
            cur_gap = vote[feature]
            cur_action = int(voter_targets.iloc[i])
            if cur_action in actions:

                if cur_gap < np.average(cur_range):
                    correct_action = actions[0]
                else:
                    correct_action = actions[1]

                if cur_action != correct_action:
                    error += 1

        return error

    def _ordered_action(self, voter_train, feature):
        vote = voter_train.iloc[0,:]
        actions = self._get_gap_feature_actions(vote, feature)
        below_action = actions[1]
        above_action = actions[0]
        return list([below_action, above_action])

    def _get_norm_value(self, voter_train_scenario, voter_targets, feature):

        gaps = list(set(voter_train_scenario[feature].tolist()))
        ordered_action = self._ordered_action(voter_train_scenario,feature)
        min_value = list(set(voter_train_scenario.loc[voter_targets == ordered_action[1], feature].tolist()))
        if len(min_value) > 0:
            norm_val = np.min(min_value)
        else:
            norm_val = 0

        min_error = self._compute_error([min_value, min_value], voter_train_scenario, voter_targets, feature)

        min_gap = np.min([np.min(gaps) - 0.5, 0])
        max_gap = np.max([np.max(gaps) + 0.5, 0])
        gaps.append(min_gap)
        gaps.append(max_gap)
        gaps.sort()

        for i in range(0, len(gaps) - 1):
            cur_range = [gaps[i], gaps[i + 1]]
            range_error = self._compute_error(cur_range, voter_train_scenario, voter_targets, feature)
            if range_error <= min_error:
                min_error = range_error
                norm_val = np.average(cur_range)

        return norm_val, min_error / (len(gaps)-2)

    def _normalize_gaps(self, df, X_train, y_train, voter_index):
        X = df
        features = self._get_gap_pref_features()
        voter_train = X_train.loc[X_train.index & voter_index]
        voter_targets = y_train.loc[voter_train.index]
        scenarios = self.scenarios_df["scenario"].drop_duplicates()
        for scenario in scenarios:
            voter_train_scenario = voter_train.loc[voter_train["Scenario"] == scenario]
            if len(voter_train_scenario) > 0:
                voter_targets_scenario = y_train.loc[voter_train_scenario.index]
                cur_index = X.loc[voter_index & X.loc[X["Scenario"] == scenario].index, :].index
                for feature in features:
                    norm_value, error = self._get_norm_value(voter_train_scenario, voter_targets_scenario, feature)
                    X.loc[cur_index, feature + "_normalized_dif"] = X.loc[cur_index, feature] - norm_value
                    X.loc[cur_index, feature + "_normalized_ratio"] = X.loc[cur_index, feature] / norm_value
                    X.loc[cur_index, feature + "_error_val"] = error
                    X.loc[cur_index, feature + "_norm_val"] = norm_value
        return X

    def _generate_behavioral_profile(self, X_train, y_train):
        voters_df = pd.DataFrame(list(self.voters_index.keys()), columns=['VoterID'])
        for i in voters_df.index:
            voter_index = self.voters_index[voters_df.values[i,0]]
            voter_train = X_train.loc[voter_index,:]
            voter_targets = y_train[voter_index]

            actual_principle, actual_leaderbias, actual_compromise = list(), list(), list()
            pref_principle, pref_leaderbias, pref_compromise = list(), list(), list()

            for i in range(0, len(voter_train)):
                vote = voter_train.iloc[i,:]
                action = int(voter_targets.values[i])
                #Compute perfect_behavior_scores
                pref_principle.append(np.prod(self._get_perfect_principle_score(vote)))

                pref_leaderbias.append(np.prod(self._get_perfect_leaderbias_score(vote)))

                pref_compromise.append(np.prod(self._get_perfect_compromise_score(vote)))

                #Compute actual scores
                actual_principle.append(np.prod(self._get_principle_score(vote, action)))

                actual_leaderbias.append(np.prod(self._get_leaderbias_score(vote, action)))

                actual_compromise.append(np.prod(self._get_compromise_score(vote, action)))

            voters_df.loc[i, "Principle_ratio_Score"] = np.sum(actual_principle) / np.sum(pref_principle)
            voters_df.loc[i, "Leaderbias_ratio_score"] = np.sum(actual_leaderbias) / np.sum(pref_leaderbias)
            voters_df.loc[i, "Compromise_ratio_score"] = np.sum(actual_compromise) / np.sum(pref_compromise)

            voters_df.loc[i, "Principle_sim"] = 1 - spatial.distance.cosine(actual_principle, pref_principle)
            voters_df.loc[i, "Leaderbias_sim"] = 1 - spatial.distance.cosine(actual_leaderbias, pref_leaderbias)
            voters_df.loc[i, "Compromise_sim"] = 1 - spatial.distance.cosine(actual_compromise, pref_compromise)

        return voters_df

    def _count_action_for_voter(self, action, voter_actions):
        action_counter = np.count_nonzero([action in voter_action for voter_action in voter_actions])

        return action_counter

    def _generate_A_ratios(self, X_train, y_train):
        """Generate A ratios - That is TRT-ratio, CMP-ratio, WLB-ratio, SLB-ratio, DOM-ratio
            Action is in {TRT,DLB,SLB,WLB,CMP,DOM}
            Scenario is in {A,B,C,D,E,F}
        """
        voters_df = pd.DataFrame(list(self.voters_index.keys()), columns=['VoterID'])
        for i in voters_df.index:
            voter_index = self.voters_index[voters_df.values[i,0]]
            voter_df = pd.concat([X_train.loc[voter_index,:], y_train[voter_index]], axis=1, join='inner')
            voter_actions = [self.action_names[x[1].Action][x[1].Scenario] for x in voter_df.iterrows()]

            for action in self._get_actions():
                availability_counter = np.count_nonzero([x[1].Scenario in self.scenarios_by_action[action] for x in voter_df.iterrows()])
                action_counter = self._count_action_for_voter(action, voter_actions)
                #df.loc[voter_index, action + '-ratio'] = float(action_counter/availability_counter if availability_counter > 0 else 0)
                #df.loc[voter_index, action + '-counter'] = float(action_counter)
                voters_df.loc[i, action + '-ratio'] = float(action_counter/availability_counter if availability_counter > 0 else 0)



        return voters_df

    def _generate_is_random_voter(self, df):
        """Identify random voters using the rule of DOM-counter >= 2 (excluding SLB actions)"""
        df['Is_Random'] = [x >= 2 for x in df['DOM-counter']]

        return df

    def _generate_voter_type(self, df):
        """Generate Voter Type using thresholds over the A-ratio values"""
        if self.n_candidates == 3:
            df['VoterType'] = 'Other'
            # X.loc[ [int(x[1]['CMP-ratio'])>=0.7 for x in X.iterrows()], 'VoterType'] = 'CMP'
            df.loc[[float(x[1]['WLB-ratio']) > 0.8 for x in df.iterrows()], 'VoterType'] = 'LB'
            df.loc[[float(x[1]['TRT-ratio']) > 0.9 for x in df.iterrows()], 'VoterType'] = 'TRT'

        return df

    def _generate_action_aggregation_features(self, X_train, y_train):
        aggregators = [np.average, np.std, np.median]
        feature_name = "Action"
        scenarios = self._get_scenarios_by_actions(self._get_strategic_actions())

        voters_df = pd.DataFrame(list(self.voters_index.keys()), columns=['VoterID'])
        for i in voters_df.index:
            voter_index = self.voters_index[voters_df.values[i, 0]]
            voter_train = X_train.loc[voter_index, :]
            train_index = np.isin(voter_train.values[:, self.columns["Scenario"]], list(scenarios))
            voter_targets = y_train[voter_index].values[train_index]

            for aggregation_func in aggregators:
                voters_df.loc[i, feature_name + "_" + aggregation_func.__name__] = aggregation_func(voter_targets)

        return voters_df

    def _generate_action_aggregation_scenario_dependant(self, df, X_train, y_train, voter_index):
        X = df

        aggregators = [np.average, np.std, np.median, skew, kurtosis]
        feature_name = "Action"

        scenarios_types = self.scenarios_df["name"].drop_duplicates()
        voter_train = X_train.loc[X_train.index & voter_index]

        for scenario_type in scenarios_types:
            voter_train_scenario = voter_train.loc[voter_train["Scenario"].isin(self._get_relevant_scenarios(scenario_type))]
            voter_targets = y_train.loc[voter_train_scenario.index]

            for aggregation_func in aggregators:
                X.loc[voter_index, feature_name + "_" + aggregation_func.__name__ + "_" + scenario_type] = aggregation_func(
                    [float(voter_targets[x[0]]) for x in voter_train_scenario.iterrows()])

        return X

    def _generate_relevant_action_aggregation_scenario_dependant(self, df, X_train, voter_index):
        X = df
        aggregators = [np.average, np.std, np.median, skew, kurtosis]
        feature_name = "Action"
        scenarios_types = self.scenarios_df["name"].drop_duplicates()
        voter_train = X_train.loc[X_train.index & voter_index]

        for scenario_type in scenarios_types:
            voter_train = voter_train.loc[voter_train["Scenario_type"] == scenario_type]
            for aggregation_func in aggregators:
                relevant_indices = voter_index & X.loc[X["Scenario_type"] == scenario_type].index
                X.loc[relevant_indices,  feature_name + "_" + aggregation_func.__name__ + "_currentScenarioType"] = X.loc[relevant_indices, feature_name + "_" + aggregation_func.__name__ + "_" + scenario_type]

        return X

    def _generate_gaps_features(self, X_train, y_train):
        voters_df = pd.DataFrame(list(self.voters_index.keys()), columns=['VoterID'])
        for i in voters_df.index:
            voter_index = self.voters_index[voters_df.values[i,0]]
            voter_df = X_train.loc[voter_index,:]
            voter_targets_df = y_train[voter_index]

            for scenario_type in self.aggregators.keys():
                scenarios = self._get_relevant_scenarios(scenario_type)
                train_index = np.isin(voter_df.values[:,self.columns["Scenario"]],scenarios)
                voter_train = voter_df.values[train_index,:]
                voter_targets = voter_targets_df.values[train_index]
                for feature_name in self.aggregators[scenario_type].keys():
                    for action in self.aggregators[scenario_type][feature_name].keys():
                        actioni_list = [float(x[self.columns[feature_name]]) for x in voter_train[voter_targets == action, :]]
                        for aggregation_func in self.aggregators[scenario_type][feature_name][action]:
                            if len(actioni_list) > 0:
                                aggregation_result = aggregation_func(actioni_list)
                            else:  # AUTO complete
                                current_actions = self._get_actions_by_gap_feature(feature_name)
                                if action == current_actions[0]:
                                    aggregation_result = 0
                                else:
                                    aggregation_result = None

                            voters_df.loc[i, scenario_type + "_" + feature_name + '_action' + str(
                                action) + '_' + aggregation_func.__name__] = aggregation_result

        voters_df = voters_df.fillna(voters_df.min())

        return voters_df

    def _generate_gap_dif_features(self, df):

        X = df
        features = self._get_gap_pref_features()
        aggregators = [np.average, np.median, np.min, np.max]
        scenario_types = self.scenarios_df["name"].drop_duplicates()

        for scenario_type in scenario_types:
            relevant_index = X.loc[X["Scenario_type"] == scenario_type,:].index
            for action in range(1, self.n_candidates + 1):
                for feature in features:
                    for aggregator in aggregators:
                        X.loc[relevant_index, "currentScenarioType" + "_" + feature + '_action'+str(action)+'_' + aggregator.__name__ + '_dif'] = X.loc[relevant_index, feature] - X.loc[relevant_index,
                            scenario_type + "_" + feature + '_action'+str(action)+'_' + aggregator.__name__]

        return X

    def _convert_to_multiple_instance(self, X, y, X_train, y_train):
        MI = pd.DataFrame(columns=['C_VoterID','C_GAP12_pref_poll','C_GAP13_pref_poll','T_GAP12_pref_poll','T_GAP13_pref_poll','T_Action','C_Action'])

        for row in range(0,len(X)):
            current = X.iloc[row]
            voter_index = X.loc[X['VoterID'] == current["VoterID"],].index
            voter_train = X_train.loc[X_train.index & voter_index]
            for train_row in range(0, len(voter_train)):
                train = voter_train.iloc[train_row]
                if current[0] != train[0]:
                    MI.loc[len(MI)] = [current["VoterID"] + "_" + str(row), current["GAP12_pref_poll"], current["GAP13_pref_poll"], train["GAP12_pref_poll"], train["GAP13_pref_poll"], y_train[train_row], y[row]]

        return MI

    def _fit(self, X_train, y_train):

        self.voters_A_ratios = self._generate_A_ratios(X_train, y_train)

        self.voters_GAP_features = self._generate_gaps_features(X_train, y_train)

        self.voters_actions = self._generate_action_aggregation_features(X_train, y_train)

        self.voters_behavioral_profile = self._generate_behavioral_profile(X_train, y_train)

        #
        # before_columns = len(X.columns)
        # X = self._generate_action_aggregation_scenario_dependant(X, X_train, y_train)
        # if len(X.columns) > before_columns:
        #     action_scenario_columns = action_scenario_columns +  list(range(before_columns, len(X.columns)))
        #
        # before_columns = len(X.columns)
        # X = self._generate_relevant_action_aggregation_scenario_dependant(X, X_train)
        # if len(X.columns) > before_columns:
        #     action_scenario_rel_columns = action_scenario_rel_columns + list(range(before_columns, len(X.columns)))
        #
        # before_columns = len(X.columns)
        #
        # if len(X.columns) > before_columns:
        #     behavioral_columns = behavioral_columns +  list(range(before_columns, len(X.columns)))

        # X = df
        # before_columns = len(X.columns)
        # X = self._generate_gap_dif_features(X)
        # gaps_dif_columns = list(range(before_columns, len(X.columns)))
        #
        # X = X.replace([np.inf], np.nan)
        # X = X.fillna(
        #     X.max())
        # X = X.replace([-np.inf], np.nan)
        # X = X.fillna(
        #     X.min())
        #
        # encoded_gap_fs = pd.DataFrame(_autoencode(pd.DataFrame(OneShotDataPreparation._prepare_dataset(X.iloc[:, to_encode_columns]))))
        # encoded_gap_fs.index = X.index
        # X = pd.concat([X, encoded_gap_fs], axis=1, join='inner')
        #
        # X = self._generate_is_random_voter(X)
        # X = self._generate_voter_type(X)

        return self

    def _transform(self, X):
        #TODO: complete the transform method
        pass




