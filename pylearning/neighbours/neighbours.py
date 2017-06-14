import abc
import numpy as np


class KNN(metaclass=abc.ABCMeta):
    """
    Abstract base class for nearest neighbours algorithms. This class is not
    meant to be instanciated, ont its subclasses can be used.
    """

    def __init__(self, samples_per_class=None, k=1):
        self.samples_per_class = samples_per_class
        self.k = k


    def get_distances(self, feat):
        """
        Calculates the distances between the given feature and all the instances
        saved during training.
        :param  feat:   An array-like object of shape (nb_features)
        :return:        A numpy array of shape (nb_training_examples)
        """
        p_squared = np.square(feat).sum() #.(axis=1)
        q_squared = np.square(self.features).sum(axis=1)
        product   = -2 * feat.dot(self.features.T)
        distances = np.sqrt(product + q_squared + np.matrix(p_squared).T)
        return distances



class KNNClassification(KNN):
    """
    K-nearest neighbours algorithm used for classification.

    :param  samples_per_class:  Number of training examples that will be saved
                                by the model. If None, all examples will be kept
    :param  k:                  Number of neighbours to use for prediction
    """


    def fit(self, features, targets):
        groups = list(set(features))
        zipped = list(zip(features, targets))
        data = {a : np.array([b[1] for b in zipped if b[0] == a]) for a in groups}
        features, targets = [], []
        self.group_means = {}
        for group in data:
            mean = np.mean(data[group], axis=0)
            features.append(mean)
            targets.append(group)
            self.group_means[group] = mean
            nb_samples = len(data[group]) if not self.samples_per_class else self.samples_per_class
            idxs = np.random.choice(range(len(data[group])), nb_samples, replace=False)
            for i in idxs:
                features.append(data[group][i])
                targets.append(group)
        self.features = np.array(features)
        self.targets = np.array(targets)


    def predict(self, X):
        distances = self.get_distances(X)
        pred = np.zeros(len(distances))
        for i in range(len(distances)):
            labels = self.targets[np.argsort(distances[i])].flatten()
            k_closest = list(labels[:self.k])
            pred[i] = max(k_closest, key=k_closest.count)
        return pred



class KNNRegression(KNN):
    """
    K-nearest neighbours algorithm used for regression.

    :param  samples_per_class:  Not used in this class
    :param  k:                  Number of neighbours to use for prediction
    """


    def fit(self, features, targets):
        self.features = features
        self.targets = targets


    def predict(self, X):
        distances = self.get_distances(X)
        predictions = np.zeros(len(distances))
        for i in range(len(distances)):
            values = self.targets[np.argsort(distances[i])].flatten()
            k_closest = list(values[:self.k])
            predictions[i] = np.mean(k_closest)
        return predictions
