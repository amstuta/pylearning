import logging
import random
from concurrent.futures import ProcessPoolExecutor

from .trees import DecisionTreeRegressor
from .trees import DecisionTreeClassifier


class RandomForest:
    """
    Base class for random forest algorithms. This class is not meant to be
    instanciated, ont its subclasses should be used.
    """

    def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1,\
                min_leaf_examples=6, max_split_features="auto",
                split_criterion=None):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers
        self.min_leaf_examples = min_leaf_examples

        if max_split_features in ["auto","sqrt","log2"] or \
            isinstance(max_split_features, int) or max_split_features is None:
            self.max_split_features = max_split_features
            self.considered_features = None
        else:
            raise ValueError("Argument max_split_features must be 'auto', \
                            'sqrt', 'log2', an int or None")

        self.split_criterion = split_criterion


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

    :param  nb_trees:           Number of decision trees to use
    :param  nb_samples:         Number of samples to give to each tree
    :param  max_depth:          Maximum depth of the trees
    :param  max_workers:        Maximum number of processes to use for training
    :param  min_leaf_examples:  Minimum number of examples in a leaf node.
    :param  max_split_features: Maximum number of features considered at each
                                split (default='auto') :
                                   - If int, the given number of will be used
                                   - If 'auto' or 'sqrt', number of features
                                     considered = sqrt(nb_features)
                                   - If 'log2', considered = log2(nb_features)
                                   - If None, all features will be considered
    :param  criterion:          Not used here
    """


    def train_tree(self, data):
        """
        Trains a single tree.
        :param data:    A tuple containing the index of the tree being trained and
                        the data to train it
        :return:        The trained tree
        """
        logging.info('Training tree {}'.format(data[0] + 1))
        tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                    min_leaf_examples=self.min_leaf_examples,
                                    max_split_features=self.max_split_features)
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

    :param  nb_trees:           Number of decision trees to use
    :param  nb_samples:         Number of samples to give to each tree
    :param  max_depth:          Maximum depth of the trees
    :param  max_workers:        Maximum number of processes to use for training
    :param  min_leaf_examples:  Minimum number of examples in a leaf node.
    :param  max_split_features: Maximum number of features considered at each
                                split (default='auto') :
                                   - If int, the given number of will be used
                                   - If 'auto' or 'sqrt', number of features
                                     considered = sqrt(nb_features)
                                   - If 'log2', considered = log2(nb_features)
                                   - If None, all features will be considered
    :param  criterion:          The function used to split data at each node of the
                                tree. If None, the criterion used is entropy.
    """


    def train_tree(self, data):
        """
        Trains a single tree.
        :param data:    A tuple containing the index of the tree being trained and
                        the data to train it
        :return:        The trained tree
        """
        logging.info('Training tree {}'.format(data[0] + 1))
        tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                    min_leaf_examples=self.min_leaf_examples,
                                    max_split_features=self.max_split_features,
                                    criterion=self.split_criterion)
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
