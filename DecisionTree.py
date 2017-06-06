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


    def fit(self, features, targets):
        """
        :param features:    Features used to train the tree.
                            Array-like object of dimensions (nb_samples, nb_features)
        :param targets:     Target values corresponding to the features.
                            Array-like object of dimensions (n_samples)
        """
        if len(features) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if self.random_features:
            self.features_indexes = self.choose_random_features(features[0])
            features = [self.get_features_subset(row) for row in features]
        self.root_node = self.build_tree(features, targets, self.max_depth)


    def predict(self, features):
        """
        Predict a value for the given features.
        :param  features:   Array of features of dimension (nb_features)
        :return:            Float value.
        """
        if self.random_features:
            if not all(i in range(len(features))
                       for i in self.features_indexes):
                raise ValueError("The given features don't match\
                                 the training set")
            features = self.get_features_subset(features)
        return self.classify(features, self.root_node)


    def choose_random_features(self, row):
        """
        Randomly selects indexes in the given list. The number of indexes
        chosen is the square root of the number of elements in the initial
        list.
        :param row: One-dimensional array
        :return:    Array containing the chosen indexes
        """
        nb_features = len(row) - 1
        return random.sample(range(nb_features), int(sqrt(nb_features)))


    def get_features_subset(self, row):
        """
        Returns the randomly selected values in the given features.
        :param row: One-dimensional array of features
        """
        return [row[i] for i in self.features_indexes]


    def divide_set(self, features, targets, column, feature_value):
        """
        Divide the given dataset depending on the value at the given column index.
        :param features:    Features of the dataset
        :param targets:     Targets of the dataset
        :param column:      The index of the column used to split data
        :param value:       The value used for the split
        """
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
        """
        Calculate the mean value of the given list.
        :param targets: One-dimensional array of floats or ints
        :return:        Float value
        """
        return sum(targets) / len(targets)


    def variance(self, targets):
        """
        Calculate the variance in the given list.
        :param targets: One-dimensional array of float or ints
        :return:        Float value
        """
        if len(targets) == 0:
            return None
        mean = self.mean_output(targets)
        variance = sum([(x - mean)**2 for x in targets])
        return variance


    def build_tree(self, features, targets, depth):
        """
        Recursively create the decision tree by splitting the dataset until there
        is no real reduce in variance, or there is less examples in a node than
        the minimum number of examples, or until the max depth is reached.
        :param features:    Array-like object of features of shape (nb_samples, nb_features)
        :param targets:     Array-like object of target values of shape (nb_samples)
        :param depth:       The current depth in the tree
        :return:            The root node of the constructed tree
        """
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


    def classify(self, observation, tree): # Change name
        """
        Makes a prediction using the given features.
        :param  observation:    The features to use to predict
        :param  tree:           The current node
        :return:                Predicted value (float)
        """
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
