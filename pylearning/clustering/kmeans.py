import logging
import numpy as np


class KMeans:
    """
    K-means algorithm.
    """

    def __init__(self, k, max_iterations=1000):
        self.k = k
        self.centroids = None
        self.labels = None
        self.max_iterations = max_iterations


    def fit(self, features):
        mins = features.min()
        maxs = features.max()

        self.centroids = np.zeros((self.k, features.shape[1]))
        for idx, col in enumerate(features.T):
            col_min, col_max = np.min(col), np.max(col)
            for k in range(self.k):
                self.centroids[k, idx] = np.random.uniform(col_min, col_max, size=(1))

        self.labels = np.random.random_integers(low=0, high=self.k - 1, size=(features.shape[0]))

        for i in range(self.max_iterations):
            logging.info("Iteration {}".format(i))
            changed = 0
            distances = self.get_distances(features)
            groups = {key: [] for key in range(self.k)}

            for idx, feat in enumerate(features):
                closest = distances[idx].argmin()
                if self.labels[idx] != closest:
                    self.labels[idx] = closest
                    changed += 1
                groups[closest].append(feat)
            for group in sorted(groups):
                group_feats = groups[group]
                if len(group_feats):
                    self.centroids[group] = np.mean(groups[group], axis=0)
                else:
                    self.centroids[group] = np.random.uniform(mins, maxs, size=(features.shape[1]))

            logging.info("Examples that changed of cluster: {}".format(changed))
            if not changed:
                break


    def predict(self, feature):
        distance = self.get_distances(feature)
        closest = distance.argmin()
        return closest


    def get_distances(self, features):
        try:
            p_squared = np.square(features).sum(axis=1)
        except:
            p_squared = np.square(features)
        q_squared = np.square(self.centroids).sum(axis=1)
        product   = -2 * features.dot(self.centroids.T)
        distances = np.sqrt(product + q_squared + np.matrix(p_squared).T)
        return distances
