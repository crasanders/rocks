import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from os.path import join
from crasanders.gcm import GCM_Sup
from scipy.optimize import minimize
import pickle

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

def fit_gcm(space):
    fit = 0
    predictions = []
    for cond in conditions:
        nweights = exemplars[rep][cond].shape[1]

        weights = np.array([1]*nMDS + [space['weight_{}'.format(i)] for i in range(nSup)])

        gcm = GCM_Sup(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=space['c'], weights=weights,
                      supp=startsup, u=space['u'], v=space['v'], w=space['w'], refs=[space['ref_{}'.format(i)] for i in range(nSup)])
        fit -= gcm.log_likelihood(stim[rep][cond], cm[cond], include_factorial=True)
        predictions.append(gcm.predict(stim[rep][cond]))
        free_parm = len(space)
        bic = 2 * fit + free_parm * logn

        if np.isnan(fit) or np.any(np.isnan(predictions)):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

    return {'loss': fit, 'status': status, 'predictions': predictions, 'free parameters': free_parm, 'bic': bic}

space = {'c': hp.loguniform('c', -1, 2),
         'u': hp.uniform('u', 0, 10), 'v': hp.uniform('v', -3, 3), 'w': hp.uniform('w', -3, 3)}
for i in range(nSup):
    space['ref_{}'.format(i)] = hp.uniform('ref_{}'.format(i), -4, 4)
    space['weight_{}'.format(i)] = hp.loguniform('weight_{}'.format(i), -2, 2)
    #for cond in conditions:
     #   space['weight_{}_{}'.format(cond, i)] = hp.loguniform('weight_{}_{}'.format(cond, i), -5, 5)


for rep in representations:
    trials = Trials()
    best = fmin(fn=fit_gcm, space=space, algo=tpe.suggest, max_evals=100000, trials=trials)
    results = trials.best_trial['result']
    results['parms'] = best
    fits[rep] = results
    print(rep, results['loss'], results['bic'])

with open(join(data_dir, 'gcm_fits_supplemental_dims_1.pkl'), 'wb') as f:
    pickle.dump(fits, f)