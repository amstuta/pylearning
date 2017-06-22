import logging
import numpy as np


class DBSCAN:
    """
    Density-based spatial clustering of application with noise.

    :param  epsilon:        Maximum distance between two examples to be
                            considered as in the same cluster
    :param  min_examples:   Minimum number of examples per cluster
    """

    def __init__(self, epsilon=0.5, min_examples=5):
        self.eps = epsilon
        self.min_examples = min_examples
        self.labels = None


    def fit(self, features):
        """
        Finds the labels for each example in the given dataset.

        :param  features:   Dataset, array-like object of shape
                            (nb_samples, nb_features)
        """
        cluster = 0
        self.labels = np.full(shape=(features.shape[0]), fill_value=-1)
        visited_nodes = np.zeros(shape=(features.shape[0]))

        for idx, feat in enumerate(features):
            if visited_nodes[idx]:
                continue
            visited_nodes[idx] = 1
            neighbours = self.find_neighbours(idx, features)
            if len(neighbours) < self.min_examples:
                continue
            self.expand_cluster(features, idx, neighbours, cluster, visited_nodes)
            cluster += 1


    def expand_cluster(self, features, idx, neighbours, cluster, visited_nodes):
        """
        Expands the cluster formed by the feature at index idx and it's
        neighbours.

        :param  features:       Dataset, array-like object of shape
                                (nb_samples, nb_features)
        :param  idx:            Index of the first feature of the current cluster
        :param  neighbours:     The first neighbours found in the same cluster as idx
        :param  cluster:        Index of the current cluster
        :param  visited_nodes:  Set of already visited nodes
        """
        self.labels[idx] = cluster
        for n in neighbours:
            if not visited_nodes[n]:
                visited_nodes[n] = 1
                new_neighbours = self.find_neighbours(n, features)
                if len(new_neighbours) >= self.min_examples:
                    neighbours += new_neighbours
            if self.labels[n] == -1:
                self.labels[n] = cluster


    def find_neighbours(self, idx, features):
        """
        Finds the neighbours of the given point which are at a maximum distance
        of self.eps from it.

        :param  idx:        Index of the current point
        :param  features:   Dataset, array-like object of shape
                            (nb_samples, nb_features)
        :returns:           List containing the indexes of the neighbours
        """
        data = features[np.setdiff1d(np.arange(features.shape[0]), idx)]
        distances = self.get_distances(features[idx], data)
        same_cluster = [idx]
        for i, dist in enumerate(distances.tolist()[0]):
            real_index = i if i < idx else i + 1
            if dist <= self.eps:
                same_cluster.append(real_index)
        return same_cluster


    def get_distances(self, point, data):
        """
        Returns the distances between a particular example and all the other
        ones in the dataset.

        :param  point:  Index of the current example
        :param  data:   Dataset, array-like object of shape
                        (nb_samples - 1, nb_features)
        :returns:       Matrix of distances of shape (1, nb_samples - 1)
        """
        p_squared = np.square(point).sum()
        q_squared = np.square(data).sum(axis=1)
        product   = -2 * point.dot(data.T)
        distances = np.sqrt(product + q_squared + np.matrix(p_squared).T)
        return distances
