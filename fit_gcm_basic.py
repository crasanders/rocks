import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from os.path import join
from crasanders.gcm import GCM
from scipy.optimize import minimize, basinhopping
import pickle

from sklearn.preprocessing import scale

class MyBounds(object):
    def __init__(self, xmax=[20,5], xmin=[0,0] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

data_dir = 'data'
nbiases = 10
nweights = 9
startsup = 8

conditions = ["Igneous", "Metamorphic", "Mixed"]
nconditions = len(conditions)

np.random.seed(0)
mds_rand = (np.random.random(120)*10 - 5).reshape(120,1)
cnn_rand = (np.random.random(120)*10 - 5).reshape(120,1)

representations = {
    'mds': np.loadtxt(join(data_dir, 'mds_120.txt')),
    'cnn': np.loadtxt(join(data_dir, 'cnn_120.txt')),
    # 'mds_rand': np.hstack((np.loadtxt(join(data_dir, 'mds_120.txt')), mds_rand))
    # 'cnn_rand': np.hstack((np.loadtxt(join(data_dir, 'cnn_120.txt')), cnn_rand))
    # 'resnet50': np.loadtxt(join(data_dir, 'resnet50_features_120.txt')),
    # 'resnet50_transformed': np.loadtxt(join(data_dir, 'resnet50_transformed_120.txt'))
    # 'vgg16': np.loadtxt(join(data_dir, 'vgg16_features_120.txt')),
    # 'vgg16_transformed': np.loadtxt(join(data_dir, 'vgg16_transformed_120.txt'))
}

cm = {
    'Igneous': np.loadtxt(join(data_dir, "confusion_matrix_igneous.txt")),
    'Metamorphic': np.loadtxt(join(data_dir, "confusion_matrix_metamorphic.txt")),
    'Mixed': np.loadtxt(join(data_dir, "confusion_matrix_mixed.txt"))
}

logn = np.log(sum([cm[cond].sum() for cond in conditions]))
strengths = np.array([i // 2 for i in range(20)])
training = np.array([1,1,0,0] * 30, dtype=bool)

categories = {
    'Igneous': [i for i in range(0,10)],
    'Metamorphic': [i for i in range(10, 20)],
    'Mixed': [1, 2, 5, 8, 11, 14, 24, 25, 26, 28],
}
cats = np.array([i // 4 for i in range(120)])

stim = {}
exemplars = {}
for rep in representations:
    stim[rep] = {}
    exemplars[rep] = {}
    for cond in conditions:
        stim[rep][cond] = representations[rep][np.isin(cats, categories[cond]),:]
        exemplars[rep][cond] = representations[rep][np.logical_and(training, np.isin(cats, categories[cond])),:]

def fit_gcm(parms, args):
    rep, fitted = args
    fit = 0
    predictions = []
    for cond in conditions:
        gcm = GCM(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=parms[0], gamma=parms[1])
        fit += gcm.log_likelihood(stim[rep][cond], cm[cond], include_factorial=fitted)
        predictions.append(gcm.predict(stim[rep][cond]))

    if np.isnan(fit):
        return np.inf

    if not fitted:
        return -fit
    else:
        return [-fit, predictions]


fits = {}
for rep in representations:
    print('fitting:', rep)
    if rep == 'resnet50':
        parm = [.26, 1.]
    if rep == 'resnet50_transformed':
        parm = [14.6, 1.]
    else:
        parm = [.1, .1]
    nweights = representations[rep].shape[1]
    fit = basinhopping(fit_gcm, parm, minimizer_kwargs={'args':[rep, False]}, accept_test=MyBounds())
    fit.n_log_lik, fit.predictions = fit_gcm(fit.x, args=[rep, True])
    fit.free_parm = len(parm)
    fit.bic = 2*fit.n_log_lik + fit.free_parm * logn
    fits[rep] = fit

with open(join(data_dir, 'best_fits.pkl'), 'rb') as f:
    best = pickle.load(f)

with open(join(data_dir, 'best_fits.pkl'), 'wb') as f:
    pickle.dump(best, f)

for rep in representations:
    print(rep)
    fit = fits[rep]
    print('free parms:', fit.free_parm, '-ln(L):', fit.n_log_lik, 'BIC:', fit.bic)
    print()