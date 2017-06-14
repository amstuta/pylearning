import random
import abc
from math import log2, sqrt
from numbers import Number

from .node import DecisionNode


class DecisionTree(metaclass=abc.ABCMeta):
    """
    Abstract base class for decision trees. This class is not meant to be
    instanciated,only its subclasses can be used.
    """


    def __init__(self, max_depth=-1, min_leaf_examples=6, max_split_features="auto"):
        self.root_node = None
        self.max_depth = max_depth
        self.min_leaf_examples = min_leaf_examples

        if max_split_features in ["auto","sqrt","log2"] or \
            isinstance(max_split_features, int) or max_split_features is None:
            self.max_split_features = max_split_features
            self.considered_features = None
        else:
            raise ValueError("Argument max_split_features must be 'auto', \
                            'sqrt', 'log2', an int or None")


    def fit(self, features, targets):
        """
        Trains the algorithm using the given dataset.
        :param features:    Features used to train the tree.
                            Array-like object of dimensions (nb_samples, nb_features)
        :param targets:     Target values corresponding to the features.
                            Array-like object of dimensions (n_samples)
        """
        if len(features) < 1:
            raise ValueError("Not enough samples in the given dataset")
        self.set_number_features_evaluated_split(features[0])
        self.root_node = self.build_tree(features, targets, self.max_depth)


    def predict(self, features):
        """
        Predict a value for the given features.
        :param features:    Array of features of dimension (nb_features)
        :return:            Float value or predicted class
        """
        return self.propagate(features, self.root_node)


    @abc.abstractmethod
    def build_tree(self, features, targets, depth):
        """ Abstract method used to build the tree """
        pass


    def set_number_features_evaluated_split(self, row):
        """
        Sets the number of considered features at each split depending on the
        max_split_features parameter.
        :param row: A single row of the features of shape (nb_features)
        """
        if isinstance(self.max_split_features, int):
            self.considered_features = self.max_split_features if \
                self.max_split_features <= len(row) else len(row)
        elif isinstance(self.max_split_features, str):
            if self.max_split_features in ['auto','sqrt']:
                self.considered_features = int(sqrt(len(row)))
            elif self.max_split_features == 'log2':
                self.considered_features = int(log2(len(row)))
        else:
            self.considered_features = len(row)


    def choose_random_features(self, row):
        """
        Randomly selects indexes in the given list. The number of indexes
        chosen is the square root of the number of elements in the initial
        list.
        :param row: One-dimensional array
        :return:    Array containing the chosen indexes
        """
        return random.sample(range(len(row)), self.considered_features)


    def get_features_subset(self, row):
        """
        Returns the randomly selected values in the given features.
        :param row: One-dimensional array of features
        """
        return [row[i] for i in self.features_indexes]


    def entropy(self, targets):
        """
        Returns the entropy in the given rows.
        :param targets:     1D array-like targets
        :return:            Float value of entropy
        """
        results = self.unique_counts(targets)
        ent = 0.0
        for val in results.values():
            p = float(val) / len(targets)
            ent -= p * log2(p)
        return ent


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


    def unique_counts(self, targets):
        """
        Returns the occurence of each result in the given dataset.
        :param targets:     1D array-like targets
        :return:            Dictionary of target => number of occurences
        """
        counts = {k: targets.count(k) for k in set(targets)}
        return counts


    def divide_set(self, features, targets, column, feature_value):
        """
        Divide the given dataset depending on the value at the given column index.
        :param features:    Features of the dataset
        :param targets:     Targets of the dataset
        :param column:      The index of the column used to split data
        :param value:       The value used for the split
        """
        split_function = None
        if isinstance(feature_value, Number):
            split_function = lambda row: row[column] >= feature_value
        else:
            split_function = lambda row: row[column] == feature_value

        set1 = [row for row in zip(features, targets) if split_function(row[0])]
        set2 = [row for row in zip(features, targets) if not split_function(row[0])]

        feat1, targs1 = [x[0] for x in set1], [x[1] for x in set1]
        feat2, targs2 = [x[0] for x in set2], [x[1] for x in set2]
        return feat1, targs1, feat2, targs2


    def propagate(self, observation, tree):
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
            if isinstance(v, Number):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.propagate(observation, branch)



class DecisionTreeRegressor(DecisionTree):
    """
    :param  max_depth:          Maximum number of splits during training
    :param min_leaf_examples:   Minimum number of examples in a leaf node.
    :param max_split_features:  Maximum number of features considered at each
                                split (default='auto') :
                                   - If int, the given number of will be used
                                   - If 'auto' or 'sqrt', number of features
                                     considered = sqrt(nb_features)
                                   - If 'log2', considered = log2(nb_features)
                                   - If None, all features will be considered
    """


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
            return DecisionNode()
        if depth == 0:
            return DecisionNode(result=self.mean_output(targets))

        lowest_variance = None
        best_criteria = None
        best_sets = None

        considered_features = self.choose_random_features(features[0])
        for column in considered_features:
            column_values = [feature[column] for feature in features]
            for feature_value in column_values:
                feats1, targs1, feats2, targs2 = \
                    self.divide_set(features, targets, column, feature_value)
                var1 = self.variance(targs1)
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
            return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                                tb=left_branch, fb=right_branch)
        else:
            return DecisionNode(result=self.mean_output(targets))



class DecisionTreeClassifier(DecisionTree):
    """
    :param  max_depth:          Maximum number of splits during training
    :param min_leaf_examples:   Minimum number of examples in a leaf node.
    :param max_split_features:  Maximum number of features considered at each
                                split (default='auto') :
                                   - If int, the given number of will be used
                                   - If 'auto' or 'sqrt', number of features
                                     considered = sqrt(nb_features)
                                   - If 'log2', considered = log2(nb_features)
                                   - If None, all features will be considered
    """


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
            return DecisionNode()
        if depth == 0:
            return DecisionNode(result=max(self.unique_counts(targets)))

        current_score = self.entropy(targets)
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        considered_features = self.choose_random_features(features[0])
        for col in considered_features:
            column_values = set([row[col] for row in features])
            for value in column_values:
                feats1, targs1, feats2, targs2 = \
                    self.divide_set(features, targets, col, value)
                p = float(len(feats1)) / len(features)
                gain = current_score - p * self.entropy(targs1) - \
                        (1 - p) * self.entropy(targs2)
                if gain > best_gain and len(feats1) > 0 and len(feats2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = ((feats1, targs1), (feats2, targs2))

        if best_gain > 0:
            left_branch = self.build_tree(best_sets[0][0], best_sets[0][1], depth - 1)
            right_branch = self.build_tree(best_sets[1][0], best_sets[1][1], depth - 1)
            return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                                tb=left_branch, fb=right_branch)
        else:
            return DecisionNode(result=max(self.unique_counts(targets)))
