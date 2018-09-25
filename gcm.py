import numpy as np

class GCM:
    def __init__(self, n_categories, n_dimensions, n_exemplars, exemplars, strengths, c=1, gamma=1, g=0, r=1, q=1,
                 biases=None, weights=None):
        self.n_categories = n_categories
        self.n_dimensions = n_dimensions
        self.n_exemplars = n_exemplars
        self.exemplars = exemplars
        self.weights = weights
        self.c = c
        self.gamma = gamma
        self.r = r
        self.q = q
        self.g = g

        strengths = np.array(strengths)
        if len(strengths.shape) == 1:
            self.strengths = np.array((n_exemplars, n_categories))
            for i, s in enumerate(strengths):
                self.strengths[i, s] = 1
        else:
            self.strengths = strengths

        if biases is None:
            self.biases = np.ones(n_categories)
        else:
            self.biases = biases

        if weights is None:
            self.weights = np.ones(n_dimensions)
        else:
            self.weights = weights

    def predict(self, X):
        probabilities = np.zeros((len(X), self.n_categories))
        for i, x in enumerate(X):
            distances = ((np.abs(self.exemplars - x) ** self.r) @ self.weights) ** (1 / self.r)
            similarities = np.exp(-self.c * distances ** self.q)
            probs = self.biases * (similarities.T @ self.strengths) ** self.gamma
            if np.sum(probs) != 0:
                probs /= np.sum(probs)
            else:
                probs += (1 / self.n_categories)
            probs = (1 - self.g) * probs + self.g * (1 / self.n_categories)
            probabilities[i, :] = probs
        return probabilities

    def score(self, X, y):
        pred = self.predict(X)
        return np.array([x[y[i]] for i, x in enumerate(pred)])
