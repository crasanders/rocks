import numpy as np
from scipy.optimize import basinhopping, minimize
from os.path import join
from crasanders.gcm import GCM, GCM_Sup
import pickle

data_dir = 'data'

nbiases = 10
start_sup = 8
n_sup = 5

conditions = ["Igneous", "Metamorphic", "Mixed"]
nconditions = len(conditions)

representations = {
    'mds_sup': np.loadtxt(join(data_dir, 'mds_120_supplemental_dims.txt')),
    'cnn_sup': np.hstack((np.loadtxt(join(data_dir, 'cnn_120.txt')),
                          np.loadtxt(join(data_dir, '120_predictions_supplemental_dims.txt'))[:, -5:]))
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

def fit_full_sup(parms, args):
    rep, fitted = args
    fit = 0
    offset1 = 4 + n_sup
    offset2 = offset1 + nbiases * nconditions
    predictions = []
    for cond in conditions:
        gcm = GCM_Sup(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=parms[0],
                          biases=parms[offset1:nbiases+offset1], weights=parms[offset2:nweights+offset2],
                          supp=start_sup, u=parms[1], v=parms[2], w=parms[3], refs=parms[4:offset1])
        fit += gcm.log_likelihood(stim[rep][cond], cm[cond], include_factorial=fitted)
        predictions.append(gcm.predict(stim[rep][cond]))
        offset1 += nbiases
        offset2 += nweights
    if not fitted:
        return -fit
    else:
        return [-fit, predictions]

niter = 10

fits = {}
for rep in representations:
    print('fitting:', rep)

    offset = 4 + n_sup
    nweights = representations[rep].shape[1]


    parm = [1.] + [1., 1., 1.] + [0]*n_sup + [1/nbiases]*nbiases*nconditions + [1/nweights]*nweights*nconditions

    fit = minimize(fit_full_sup, parm, args=[rep, False],
                   bounds=[(0, None), (0, None)] + [(None, None)]*2 + [(-5, 5)]*n_sup + [(0,1)]*nbiases*nconditions + [(0,1)]*nweights*nconditions,
                   constraints=[{'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[offset:nbiases+offset])},
                     {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[nbiases+offset:2*nbiases+offset])},
                     {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[2*nbiases+offset:3*nbiases+offset])},
                     {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[3*nbiases+offset:3*nbiases+nweights+offset])},
                     {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[3*nbiases+nweights+offset:3*nbiases+2*nweights+offset])},
                     {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[3*nbiases+2*nweights+offset:3*nbiases+3*nweights+offset])}])

    fit.n_log_lik, fit.predictions = fit_full_sup(fit.x, args=[rep, True])

    fit.free_parm = 1 + (nbiases-1)*3 + (nweights-1)*3 + n_sup + 3
    fit.bic = 2*fit.n_log_lik + fit.free_parm * logn

    fits[rep] = fit

with open(join(data_dir, 'fits_full_supp.pkl'), 'wb') as f:
    pickle.dump(fits, f)

for rep in representations:
    print(rep)
    fit = fits[rep]
    print('free parms:', fit.free_parm, '-ln(L):', fit.n_log_lik, 'BIC:', fit.bic)
    print()


def fit_partial_sup(parms, args):
    rep, fitted, dim = args
    fit = 0
    offset1 = 5
    offset2 = offset1 + nbiases * nconditions
    predictions = []
    for cond in conditions:
        ex = np.column_stack((exemplars[rep][cond][:, :8], exemplars[rep][cond][:, dim]))
        st = np.column_stack((stim[rep][cond][:,:8], stim[rep][cond][:,dim]))
        gcm = GCM_Sup(nbiases, nweights, 20, ex, strengths, c=parms[0],
                          biases=parms[offset1:nbiases+offset1], weights=parms[offset2:nweights+offset2],
                          supp=8, u=parms[1], v=parms[2], w=parms[3], refs=[parms[4]])
        fit += gcm.log_likelihood(st, cm[cond], include_factorial=fitted)
        predictions.append(gcm.predict(st))
        offset1 += nbiases
        offset2 += nweights
    if not fitted:
        return -fit
    else:
        return [-fit, predictions]

partial_fits = {}
for rep in representations:
    partial_fits[rep] = {}
    for dim in range(8,13):
        print('fitting:', rep, 'dimension:', dim)

        offset = 5
        nweights = 9
        parm = [1.] + [1., 1., 1., 0] + [1/nbiases]*nbiases*nconditions + [1/nweights]*nweights*nconditions

        fit = basinhopping(fit_partial_sup, parm, minimizer_kwargs={'args':[rep, False, dim],
                       'bounds':[(0, None), (0, None)] + [(None, None)]*2 + [(-5, 5)] + [(0,1)]*nbiases*nconditions + [(0,1)]*nweights*nconditions,
                       'constraints':[{'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[offset:nbiases+offset])},
                         {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[nbiases+offset:2*nbiases+offset])},
                         {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[2*nbiases+offset:3*nbiases+offset])},
                         {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[3*nbiases+offset:3*nbiases+nweights+offset])},
                         {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[3*nbiases+nweights+offset:3*nbiases+2*nweights+offset])},
                         {'type': 'eq', 'fun': lambda parms: 1- np.sum(parms[3*nbiases+2*nweights+offset:3*nbiases+3*nweights+offset])}]}
                       )

        fit.n_log_lik, fit.predictions = fit_partial_sup(fit.x, args=[rep, True, dim])

        fit.free_parm = 1 + (nbiases-1)*3 + (nweights-1)*3 + 4
        fit.bic = 2*fit.n_log_lik + fit.free_parm * logn

        partial_fits[rep][dim] = fit

with open(join(data_dir, 'fits_partial_supp_bh.pkl'), 'wb') as f:
    pickle.dump(partial_fits, f)

for rep in representations:
    print(rep)
    for dim in range(8, 13):
        fit = partial_fits[rep][dim]
        print('free parms:', fit.free_parm, '-ln(L):', fit.n_log_lik, 'BIC:', fit.bic)
        print()

with open(join(data_dir, 'fits_full_supp.pkl'), 'rb') as f:
    old_fits = pickle.load(f)