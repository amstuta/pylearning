import logging
import random
from concurrent.futures import ProcessPoolExecutor

from .trees import DecisionTreeRegressor
from .trees import DecisionTreeClassifier


class RandomForest:
    """
    Base class for random forest algorithms.
    This class is not meant to be instanciated, its subclasses should be used
    instead.
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


    def fit(self, features, targets):
        """
        Trains self.nb_trees number of decision trees.
        :param features:    Array-like object of shape (nb_samples, nb_features)
                            containing the training examples
        :param targets:     Array-like object of shape (nb_samples) containing the
                            training targets.
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            zipped = list(zip(features, targets))
            random_features = \
                [(x, random.sample(zipped, self.nb_samples)) for x in range(self.nb_trees)]
            self.trees = list(executor.map(self.train_tree, random_features))



class RandomForestRegressor(RandomForest):
    """
    Implementation of a random forest used for regression problems.
    :param  nb_trees:       Number of decision trees to use
    :param  nb_samples:     Number of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """


    def train_tree(self, data):
        """
        Trains a single tree.
        :param data:    A tuple containing the index of the tree being trained and
                        the data to train it
        :return:        The trained tree
        """
        logging.info('Training tree {}'.format(data[0] + 1))
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        features, targets = [x[0] for x in data[1]], [x[1] for x in data[1]]
        tree.fit(features, targets)
        return tree


    def predict(self, feature):
        """
        Returns a prediction for the given feature. The result is the mean of
        predictions made by all trees.
        :param  feature:    The features used to predict: shape (nb_features)
        :return:            Prediction as a float value
        """
        predictions = [tree.predict(feature) for tree in self.trees]
        ensemble_prediction = sum(predictions) / len(predictions)
        return ensemble_prediction



class RandomForestClassifier(RandomForest):
    """
    Implementation of a random forest used for classification problems.
    :param nb_trees:       Number of decision trees to use
    :param nb_samples:     Number of samples to give to each tree
    :param max_depth:      Maximum depth of the trees
    :param max_workers:    Maximum number of processes to use for training
    """


    def train_tree(self, data):
        """
        Trains a single tree.
        :param data:    A tuple containing the index of the tree being trained and
                        the data to train it
        :return:        The trained tree
        """
        logging.info('Training tree {}'.format(data[0] + 1))
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        features, targets = [x[0] for x in data[1]], [x[1] for x in data[1]]
        tree.fit(features, targets)
        return tree


    def predict(self, feature):
        """
        Returns a prediction for the given feature. The result is the maximum
        vote in the predicions made by all the trees.
        :param  feature:    The features used to predict: shape (nb_features)
        :return:            The predicted class
        """
        predictions = [tree.predict(feature) for tree in self.trees]
        ensemble_vote = max(set(predictions), key=predictions.count)
        return ensemble_vote
