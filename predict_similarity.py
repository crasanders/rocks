import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

features = np.loadtxt('data/vgg16_features_120.txt')

dotproduct = features @ features.transpose()
dotproduct = dotproduct[np.triu(dotproduct, k=1).astype(bool)]

obs_sim = np.array(pd.read_csv('data/similarity_120.csv', header=None))
obs_sim = obs_sim[np.triu(obs_sim, k=1).astype(bool)]

features_360 = np.loadtxt('data/vgg16_features_360.txt')
n, dim = features_360.shape
x = np.array([features_360[i,:] * features_360[j,:] for i in range(n-1) for j in range(i+1, n)])
y = np.array(pd.read_csv('data/similarity_360.csv', header=None))
y = y[np.triu(y, k=1).astype(bool)]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

scaler = StandardScaler()
kf = KFold(5)
best_loss = np.inf
best_lam = None
for lam in np.logspace(-8, 2, 10):
    this_loss = 0
    for train, val in kf.split(x):
        x_train, x_val = x[train], x[val]
        y_train, y_val = y[train], y[val]

        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        X = np.vstack((x_train, lam*np.eye(dim)))
        Y = np.concatenate((y_train, [0] * dim))

        w, _ = nnls(X, Y)
        pred = x_val @ w
        loss = np.sum((pred - y_val)**2)
        this_loss += loss

    if this_loss < best_loss:
        best_loss = this_loss
        best_lam = lam

x = scaler.fit_transform(x)
X = np.vstack((x, best_lam*np.eye(dim)))
Y = np.concatenate((y, [0] * dim))
w, loss = nnls(X, Y)

transf_features = features * np.sqrt(w)
transf_dot = transf_features @ transf_features.transpose()
transf_dot = transf_dot[np.triu(transf_dot, k=1).astype(bool)]

sims = pd.DataFrame({'Raw': dotproduct, 'Transformed': transf_dot, 'Observed': obs_sim})
print(sims.corr())

np.savetxt('vgg16_transformed_120.txt', transf_features, fmt='%.18f')

