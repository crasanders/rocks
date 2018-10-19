import numpy as np

class GCM:
    def __init__(self, n_categories, n_dimensions, n_exemplars, exemplars, strengths, c=1, gamma=1, g=0, r=2, q=1,
                 biases=None, weights=None):
        self.n_categories = n_categories
        self.n_dimensions = n_dimensions

        self.exemplars = exemplars.astype(float)
        self.weights = weights
        self.c = c
        self.gamma = gamma
        self.r = r
        self.q = q
        self.g = g

        strengths = np.array(strengths)
        if len(strengths.shape) == 1:
            self.strengths = np.zeros((n_exemplars, n_categories))
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

    def log_likelihood(self, X, conf_mat, include_factorial=True):
        loglik = 0
        pred = self.predict(X)
        loglik += np.sum(np.log(pred) * conf_mat)

        if include_factorial:
            for s in conf_mat.sum(1):
                for i in range(1, int(s) + 1):
                    loglik += np.log(i)
            for row in conf_mat:
                for col in row:
                    for i in range(1, int(col) + 1):
                        loglik -= np.log(i)

        return loglik

class GCM_Sup(GCM):
    def __init__(self, n_categories, n_dimensions, n_exemplars, exemplars, strengths, c=1, gamma=1, g=0, r=2, q=1,
                 biases=None, weights=None, supp=0, u=1, v=1, w=1, refs=None):
        super().__init__(n_categories, n_dimensions, n_exemplars, exemplars, strengths, c, gamma, g, r, q,
                 biases, weights)
        self.supp = supp
        self.u = u
        self.v = v
        self.w = w

        if refs is None:
            self.refs = np.zeros(self.n_dimensions - self.supp)
        else:
            self.refs = refs

        for j, i in enumerate(range(self.supp, self.n_dimensions)):
            self.exemplars[:,i] = self.transform(self.exemplars[:,i], self.refs[j])

    def predict(self, X):
        X_t = X.copy().astype(float)
        for j, i in enumerate(range(self.supp, self.n_dimensions)):
            X_t[:,i] = self.transform(X[:,i], self.refs[j])
        return super().predict(X_t)

    def transform(self, X, ref):
        X_t = X.copy().astype(float)
        X_t[X >= ref] = ref + self.u * (X_t[X >= ref] - ref) ** self.v
        X_t[X < ref] = ref - (ref - X_t[X < ref]) ** self.w
        return X_t


