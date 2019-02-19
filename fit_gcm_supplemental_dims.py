import numpy as np
# from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from scipy.optimize import minimize, basinhopping
from crasanders.gcm import GCM_Sup
import pickle
from os.path import join

data_dir = 'data'
nbiases = 10
nMDS = 8
nSup = 5
startsup = 8

conditions = ["Igneous", "Metamorphic", "Mixed"]
nconditions = len(conditions)

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

fits = {}
representations = {
    'mds_sup': np.loadtxt(join(data_dir, 'mds_120_supplemental_dims.txt')),
    'cnn_sup': np.loadtxt(join(data_dir, '120_predictions_supplemental_dims.txt'))
}

stim = {}
exemplars = {}
for rep in representations:
    stim[rep] = {}
    exemplars[rep] = {}
    for cond in conditions:
        stim[rep][cond] = representations[rep][np.isin(cats, categories[cond]), :]
        exemplars[rep][cond] = representations[rep][np.logical_and(training, np.isin(cats, categories[cond])), :]

def fit_gcm(space, args):
    rep, fitted = args
    fit = 0
    predictions = []
    for cond in conditions:
        nweights = exemplars[rep][cond].shape[1]

        weights = np.array([1]*nMDS + list(space[10:]))

        gcm = GCM_Sup(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=space[0], gamma=space[1], weights=weights,
                      supp=startsup, u=space[2], v=space[3], w=space[4], refs=space[5:10])
        fit -= gcm.log_likelihood(stim[rep][cond], cm[cond], include_factorial=fitted)
        predictions.append(gcm.predict(stim[rep][cond]))

    if np.isnan(fit):
        return np.inf

    if not fitted:
        return fit
    else:
        return [fit, predictions]

    return fit


class MyBounds(object):
    def __init__(self,
                 xmax=[np.inf, np.inf, np.inf, np.inf, np.inf,
                       5,5,5,5,5,
                       np.inf, np.inf, np.inf, np.inf, np.inf],
                 xmin=[0,0,0,0,0,
                       -5,-5,-5,-5,-5,
                       0,0,0,0,0]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

fits = {}
for rep in representations:
    print('fitting:', rep)
    if rep == 'mds_sup':
        parm = [.88, 1.1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    if rep == 'cnn_sup':
        parm = [.88, 1., 1.94, -.14, .52, -2.6, -1.62, 2.23, -2.39, 1.86, .53, 1.74, 4.3, .48, 5.03]
    fit = basinhopping(fit_gcm, parm, minimizer_kwargs={'args':[rep, False]}, accept_test=MyBounds())
    fit.n_log_lik, fit.predictions = fit_gcm(fit.x, args=[rep, True])
    fit.free_parm = len(parm)
    fit.bic = 2*fit.n_log_lik + fit.free_parm * logn
    fits[rep] = fit

for rep in representations:
    print(rep)
    fit = fits[rep]
    print('free parms:', fit.free_parm, '-ln(L):', fit.n_log_lik, 'BIC:', fit.bic)
    print()

with open(join(data_dir, 'best_fits_supplemental.pkl'), 'wb') as f:
    pickle.dump(fits, f)

