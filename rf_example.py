#!/usr/bin/env python3

import logging
from math import sqrt
from sklearn.model_selection import train_test_split
from random_forest.random_forest import RandomForestRegressor


def test_rf():
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
