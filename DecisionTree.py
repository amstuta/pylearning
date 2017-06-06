#!/usr/bin/env python3

import random
from math import log, sqrt


class DecisionTreeRegressor:
    """
    :param  max_depth:          Maximum number of splits during training
    :param  random_features:    If False, all the features will be used to
                                train and predict. Otherwise, a random set of
                                size sqrt(nb features) will be chosen in the
                                features.
                                Usually, this option is used in a random
                                forest.
    :param min_leaf_examples:   Minimum number of examples in a meaf node.
    """

    class DecisionNode:
        def __init__(self, col=-1, value=None, result=None, tb=None, fb=None):
            self.col = col
            self.value = value
            self.result = result
            self.tb = tb
            self.fb = fb


    def __init__(self, max_depth=-1, random_features=False, min_leaf_examples=6):
        self.root_node = None
        self.max_depth = max_depth
        self.features_indexes = []
        self.random_features = random_features
        self.min_leaf_examples = min_leaf_examples

    """
    :param  rows:       The data used to rain the decision tree. It must be a
                        list of lists. The last vaue of each inner list is the
                        value to predict.
    """
    def fit(self, features, targets):
        if len(features) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if self.random_features:
            self.features_indexes = self.choose_random_features(features[0])
            features = [self.get_features_subset(row) for row in features]
        self.root_node = self.build_tree(features, targets, self.max_depth)

    """
    Returns a prediction for the given features.
    :param  features:   A list of values
    """
    def predict(self, features):
        if self.random_features:
            if not all(i in range(len(features))
                       for i in self.features_indexes):
                raise ValueError("The given features don't match\
                                 the training set")
            features = self.get_features_subset(features)

        return self.classify(features, self.root_node)

    """
    Randomly selects indexes in the given list.
    """
    def choose_random_features(self, row):
        nb_features = len(row) - 1
        return random.sample(range(nb_features), int(sqrt(nb_features)))

    """
    Returns the randomly selected values in the given features
    """
    def get_features_subset(self, row):
        return [row[i] for i in self.features_indexes]

    """
    Divides the given dataset depending on the value at the given column index.
    :param  rows:   The dataset
    :param  column: The index of the column used to split data
    :param  value:  The value used for the split
    """
    def divide_set(self, features, targets, column, feature_value):
        split_function = None
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            split_function = lambda row: row[column] >= feature_value
        else:
            split_function = lambda row: row[column] == feature_value

        set1 = [row for row in zip(features, targets) if split_function(row[0])]
        set2 = [row for row in zip(features, targets) if not split_function(row[0])]

        feat1, targs1 = [x[0] for x in set1], [x[1] for x in set1]
        feat2, targs2 = [x[0] for x in set2], [x[1] for x in set2]

        return feat1, targs1, feat2, targs2


    def mean_output(self, targets):
        return sum(targets) / len(targets)


    def variance(self, targets):
        if len(targets) == 0:
            return None
        mean = self.mean_output(targets)
        variance = sum([(x - mean)**2 for x in targets])
        return variance


    """
    Recursively creates the decision tree by splitting the dataset until no
    gain of information is added, or until the max depth is reached.
    :param  rows:   The dataset
    :param  func:   The function used to calculate the best split and stop
                    condition
    :param  depth:  The current depth in the tree
    """
    def build_tree(self, features, targets, depth):
        if len(features) == 0:
            return self.DecisionNode()
        if depth == 0:
            return self.DecisionNode(result=self.mean_output(targets))
            # Add check: if all features have same value, stop

        lowest_variance = None
        best_criteria = None
        best_sets = None

        for column in range(len(features[0])):
            column_values = sorted([feature[column] for feature in features]) # Maybe remove sorted
            for feature_value in column_values:
                feats1, targs1, feats2, targs2 = \
                    self.divide_set(features, targets, column, feature_value)
                var1 = self.variance(targs1) # Replace variance by deviation
                var2 = self.variance(targs2)
                if var1 is None or var2 is None:
                    continue
                variance = var1 + var2

                if lowest_variance is None or variance < lowest_variance:
                    lowest_variance = variance
                    best_criteria   = (column, feature_value)
                    best_sets       = ((feats1, targs1),(feats2, targs2))

        # Check variance value also
        if lowest_variance is not None and \
            len(best_sets[0][0]) >= self.min_leaf_examples and \
            len(best_sets[1][0]) >= self.min_leaf_examples:
            left_branch = self.build_tree(best_sets[0][0], best_sets[0][1], depth - 1)
            right_branch = self.build_tree(best_sets[1][0], best_sets[1][1], depth - 1)
            return self.DecisionNode(col=best_criteria[0],
                                     value=best_criteria[1],
                                     tb=left_branch,
                                     fb=right_branch)
        else:
            return self.DecisionNode(result=self.mean_output(targets))


    """
    Makes a prediction using the given features.
    :param  observation:    The features to use to predict
    :param  tree:           The current node
    """
    def classify(self, observation, tree):
        if tree.result is not None:
            return tree.result
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)




def read_data(filepath):
    features, targets = [], []
    with open(filepath) as fd:
        for row in fd:
            row = row.split()
            features.append([float(row[x]) for x in range(21)])
            targets.append(float(row[21]))
    return features, targets


def test_tree():
    from sklearn.model_selection import train_test_split

    features, targets = read_data("user_cpu.data")

    train, test = train_test_split(list(zip(features, targets)), test_size=0.3)
    train_features = [f[0] for f in train]
    train_targets = [t[1] for t in train]

    tree = DecisionTreeRegressor(random_features=True)
    tree.fit(train_features, train_targets)

    absolute_errors = []
    squared_errors = []
    for feat, targ in test:
        pred = tree.predict(feat)
        squared_errors.append((pred - targ)**2)
        absolute_errors.append(abs(pred - targ))

    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = sqrt(sum(squared_errors) / len(squared_errors))
    print("MAE = {}; RMSE = {}".format(mae, rmse))


if __name__ == '__main__':
    test_tree()
