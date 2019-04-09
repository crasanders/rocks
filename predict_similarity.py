import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.spatial.distance import pdist, squareform


#simulated data
np.random.seed(0)

features_360 = np.random.rand(50, 100)
features_120 = np.random.rand(50, 100)

w_ = np.linspace(0, 99, 100)
sim_360 = 10 - squareform(pdist(features_360 * w_))
sim_120 = 10 - squareform(pdist(features_120 * w_))

# sim_360 = np.array(pd.read_csv('data/similarity_360.csv', header=None))
# sim_120 = np.array(pd.read_csv('data/similarity_120.csv', header=None))

#
# features_360 = np.loadtxt('data/resnet50_features_360.txt')
# features_120 = np.loadtxt('data/resnet50_features_120.txt')


n, dim = features_360.shape
x = np.array([(features_360[i,:] - features_360[j,:])**2 for i in range(n-1) for j in range(i+1, n)])
y = 10 - sim_360[np.triu(sim_360, k=1).astype(bool)]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

y = y ** 2

kf = KFold(5)
best_loss = np.inf
best_lam = None
for i, lam in enumerate(np.logspace(-1, 3, 10)):
    print(i)
    this_loss = 0
    for train, val in kf.split(x):
        x_train, x_val = x[train], x[val]
        y_train, y_val = y[train], y[val]

        X = np.vstack((x_train, lam*np.eye(dim)))
        Y = np.concatenate((y_train, [0] * dim))

        w, _ = nnls(X, Y)
        pred = x_val @ w
        loss = np.sum((pred - y_val)**2)
        this_loss += loss

    if this_loss < best_loss:
        best_loss = this_loss
        best_lam = lam

print(best_lam)
X = np.vstack((x, best_lam*np.eye(dim)))
Y = np.concatenate((y, [0] * dim))
w, loss = nnls(X, Y)

raw_360 = x @ np.ones((x.shape[1],))
transf_360 = x @ w
print(r2_score(y, raw_360))
print(r2_score(y, transf_360))

transf_features = features_120 * np.sqrt(w)
# np.savetxt('resnet50_transformed_120.txt', transf_features, fmt='%.18f')
# np.savetxt('regression_weights.txt', w, fmt='%.18f')

n, dim = features_120.shape
dist = np.array([(features_120[i,:] - features_120[j,:])**2 for i in range(n-1) for j in range(i+1, n)])
transf_dist = np.array([(transf_features[i,:] - transf_features[j,:])**2 for i in range(n-1) for j in range(i+1, n)])
y2 = 10 - sim_120[np.triu(sim_120, k=1).astype(bool)]

dist = dist[~np.isnan(y2)]
y2 = y2[~np.isnan(y2)]
y2 = y2 ** 2

raw = dist @ np.ones((dim,))
transf = transf_dist @ np.ones((dim,))
print(r2_score(y2, raw))
print(r2_score(y2, transf))