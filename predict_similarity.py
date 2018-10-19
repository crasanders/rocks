import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

features = np.loadtxt('resnet50_features_120.txt')

dotproduct = features @ features.transpose()
dotproduct = dotproduct[np.triu(dotproduct, k=1).astype(bool)]

obs_sim = np.array(pd.read_csv('similarity_120.csv', header=None))
obs_sim = obs_sim[np.triu(obs_sim, k=1).astype(bool)]

features_360 = np.loadtxt('resnet50_features.txt')
n = len(features_360)
x = np.array([features_360[i,:] * features_360[j,:] for i in range(n-1) for j in range(i+1, n)])
y = np.array(pd.read_csv('similarity_360.csv', header=None))
y = y[np.triu(y, k=1).astype(bool)]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]
ridge = RidgeCV(alphas=[i for i in np.logspace(-1, 4, 50)], cv=10, fit_intercept=False).fit(x, y)

transf_features = features * ridge.coef_
transf_dot = transf_features @ transf_features.transpose()
transf_dot = transf_dot[np.triu(transf_dot, k=1).astype(bool)]

sims = pd.DataFrame({'Raw': dotproduct, 'Transformed': transf_dot, 'Observed': obs_sim})
print(sims.corr())

np.savetxt('resnet50_transformed_120.txt', transf_features, fmt='%.18f')