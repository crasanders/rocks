import numpy as np
from scipy.optimize import basinhopping
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
    'mds_base': np.loadtxt(join(data_dir, 'mds_120.txt')),
    'mds_sup': np.loadtxt(join(data_dir, 'mds_120_supplemental_dims.txt')),
    'cnn_base': np.loadtxt(join(data_dir, 'cnn_120.txt')),
    'cnn_sup': np.hstack((np.loadtxt(join(data_dir, 'cnn_120.txt')), np.loadtxt(join(data_dir, '120_predictions_supplemental_dims.txt'))[:, -5:])),
    'resnet': np.loadtxt(join(data_dir, 'resnet50_features_120.txt')),
    'resnet_transf': np.loadtxt(join(data_dir, 'resnet50_transformed_120.txt')),
    # 'pixel': np.loadtxt(join(data_dir, 'pixel_120.txt')) / 255
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


def fit_c(parms, args):
    rep, factorial = args
    nweights = representations[rep].shape[1]
    fit = 0
    for cond in conditions:
        gcm = GCM(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=parms[0])
        fit += gcm.log_likelihood(stim[rep][cond], cm[cond], include_factorial=factorial)
    return -fit

def fit_c_sup(parms, args):
    rep, factorial = args
    nweights = representations[rep].shape[1]
    fit = 0
    for cond in conditions:
        gcm = GCM_Sup(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=parms[0],
                      supp=start_sup, u=parms[1], v=parms[2], w=parms[3], refs=parms[4:])
        fit += gcm.log_likelihood(stim[rep][cond], cm[cond], include_factorial=factorial)

    return -fit

# def fit_biases(parms, rep):
#     nweights = representations[rep].shape[1]
#     fit = 0
#     offset = 1
#     for cond in conditions:
#         gcm = GCM(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=parms[0],
#                   biases=parms[offset:nbiases+offset], r=2)
#         fit += gcm.log_likelihood(stim[rep][cond], cm[cond])
#         offset += nbiases
#     return -fit
#
# def fit_weights(parms, rep):
#     nweights = representations[rep].shape[1]
#     fit = 0
#     offset = 1
#     for cond in conditions:
#         gcm = GCM(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=parms[0],
#                   weights=parms[offset:nweights+offset], r=2)
#         fit += gcm.log_likelihood(stim[rep][cond], cm[cond])
#         offset += nweights
#     return -fit
#
# def fit_full(parms, rep):
#     nweights = representations[rep].shape[1]
#     fit = 0
#     offset1 = 1
#     offset2 = nbiases * nconditions + 1
#     for cond in conditions:
#         gcm = GCM(nbiases, nweights, 20, exemplars[rep][cond], strengths, c=parms[0],
#                   biases=parms[offset1:nbiases+offset1], weights=parms[offset2:nweights+offset2])
#         fit += gcm.log_likelihood(stim[rep][cond], cm[cond], include_factorial=False)
#         offset1 += nbiases
#         offset2 += nweights
#     return -fit

fits = {}
for rep in representations:
    print(rep)

    # if '_sup' in rep:
    #     parm = [1,1,1,1,0,0,0,0,0]
    #     fit = basinhopping(fit_c_sup, parm, minimizer_kwargs={'args':[rep, False]})
    #     fit.n_log_lik = fit_c_sup(fit.x, args=[rep, True])

    parm = [1]
    fit = basinhopping(fit_c, parm, minimizer_kwargs={'args':[rep, False]})
    fit.n_log_lik = fit_c(fit.x, args=[rep, True])

    fit.free_parm = len(parm)
    fit.bic = 2*fit.n_log_lik + fit.free_parm * logn

    fits[rep] = fit

# with open(join(data_dir, 'fits_c.pkl'), 'wb') as f:
#     pickle.dump(fits, f)

for rep in representations:
    print(rep)
    fit = fits[rep]
    print('-ln(L):', fit.n_log_lik, 'BIC:', fit.bic)
    print('parameters:', fit.x)
    print()
