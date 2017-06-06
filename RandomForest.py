#!/usr/bin/env python3

import logging
import random
from concurrent.futures import ProcessPoolExecutor
from DecisionTree import DecisionTreeRegressor


class RandomForestRegressor:

    """
    :param  nb_trees:       Number of decision trees to use
    :param  nb_samples:     Number of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """
    def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers

    """
    Trains self.nb_trees number of decision trees.
    :param  data:   A list of lists with the last element of each list being
                    the value to predict
    """
    def fit(self, features, targets):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            zipped = list(zip(features, targets))
            random_features = \
                [(x, random.sample(zipped, self.nb_samples)) for x in range(self.nb_trees)]
            self.trees = list(executor.map(self.train_tree, random_features))

    """
    Trains a single tree and returns it.
    :param  data:   A List containing the index of the tree being trained
                    and the data to train it
    """
    def train_tree(self, data):
        logging.info('Training tree {}'.format(data[0] + 1))
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        features, targets = [x[0] for x in data[1]], [x[1] for x in data[1]]
        tree.fit(features, targets)
        return tree

    """
    Returns a prediction for the given feature. The result is the value that
    gets the most votes.
    :param  feature:    The features used to predict
    """
    def predict(self, feature):
        predictions = [tree.predict(feature) for tree in self.trees]
        ensemble_prediction = sum(predictions) / len(predictions)
        return ensemble_prediction




def test_rf():
    from math import sqrt
    from sklearn.model_selection import train_test_split

    features, targets = [], []
    with open("user_cpu.data") as fd:
        for row in fd:
            row = row.split()
            features.append([float(row[x]) for x in range(21)])
            targets.append(float(row[21]))
    train, test = train_test_split(list(zip(features, targets)), test_size=0.3)
    train_features = [f[0] for f in train]
    train_targets = [t[1] for t in train]

    rf = RandomForestRegressor(nb_trees=10, nb_samples=1000, max_workers=4)
    rf.fit(train_features, train_targets)

    absolute_errors = []
    squared_errors = []
    for feat, targ in test:
        pred = rf.predict(feat)
        squared_errors.append((pred - targ)**2)
        absolute_errors.append(abs(pred - targ))
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = sqrt(sum(squared_errors) / len(squared_errors))
    print("MAE = {}; RMSE = {}".format(mae, rmse))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_rf()
