import numpy as np
import pandas as pd
from scipy.optimize import minimize
from crasanders.gcm import GCM


data = pd.read_csv('categorization 120.csv')
mds = np.loadtxt('mds_120.txt')
cnn = np.load('ensemble_120_pred.npy')

training = np.array([1, 1, 0, 0] * 30, dtype=bool)
cats = np.array([i // 4 for i in range(120)])

igneous = [i for i in range(0, 10)]
metamorphic = [i for i in range(10, 20)]
mixed = [1, 2, 5, 8, 11, 14, 24, 25, 26, 28]

obs = np.array(data.query('Session == 2 and outlier == False').groupby(['Condition', 'Subject', 'Subtype', 'Item Type'],
                                                                       as_index=False).mean().groupby(['Condition', 'Subtype', 'Item Type']).mean()['Correct'])
categories = np.array([i // 2 for i in range(20)])

strengths = [i for i in range(10) for j in range(2)]

def min_gcm(cs, obs, rep):
    preds = []

    for i, cat in enumerate(igneous):
        exemplars = rep[np.logical_and(training, np.isin(cats, igneous)), :]
        preds.append(np.mean(
            GCM(10, 8, 20, exemplars, strengths, c=cs, r=2).score(rep[np.logical_and(~training, cats == cat), :],
                                                                  [i, i])))
        preds.append(np.mean(
            GCM(10, 8, 20, exemplars, strengths, c=cs, r=2).score(rep[np.logical_and(training, cats == cat), :],
                                                                  [i, i])))

    for i, cat in enumerate(metamorphic):
        exemplars = rep[np.logical_and(training, np.isin(cats, metamorphic)), :]
        preds.append(np.mean(
            GCM(10, 8, 20, exemplars, strengths, c=cs, r=2).score(rep[np.logical_and(~training, cats == cat), :],
                                                                  [i, i])))
        preds.append(np.mean(
            GCM(10, 8, 20, exemplars, strengths, c=cs, r=2).score(rep[np.logical_and(training, cats == cat), :],
                                                                  [i, i])))

    for i, cat in enumerate(mixed):
        exemplars = rep[np.logical_and(training, np.isin(cats, mixed)), :]
        preds.append(np.mean(
            GCM(10, 8, 20, exemplars, strengths, c=cs, r=2).score(rep[np.logical_and(~training, cats == cat), :],
                                                                  [i, i])))
        preds.append(np.mean(
            GCM(10, 8, 20, exemplars, strengths, c=cs, r=2).score(rep[np.logical_and(training, cats == cat), :],
                                                                  [i, i])))

    preds = np.array(preds)
    return [preds, np.mean((preds - obs) ** 2)]


ssd_mds = lambda c: min_gcm(c, obs, mds)[1]
mds_min = minimize(ssd_mds, [1, 1])

ssd_cnn = lambda c: min_gcm(c, obs, cnn)[1]
cnn_min = minimize(ssd_cnn, [1, 1])

preds = min_gcm(cnn_min.x, obs, mds)[0]

ctgrs = np.array([i // 4 for i in range(40)])
train = training[:40]


def get_preds(rep, params):
    ig_exemplars = rep[np.logical_and(training, np.isin(cats, igneous)), :]
    met_exemplars = rep[np.logical_and(training, np.isin(cats, metamorphic)), :]
    mix_exemplars = rep[np.logical_and(training, np.isin(cats, mixed)), :]

    ig = GCM(10, 8, 20, ig_exemplars, strengths, c=params, r=2).score(rep[np.isin(cats, igneous), :], ctgrs)
    met = GCM(10, 8, 20, met_exemplars, strengths, c=params, r=2).score(rep[np.isin(cats, metamorphic), :], ctgrs)
    mix = GCM(10, 8, 20, mix_exemplars, strengths, c=params, r=2).score(rep[np.isin(cats, mixed), :], ctgrs)

    preds = {}

    preds['overall'] = [np.mean(ig[training[:40]]),
                        np.mean(met[training[:40]]),
                        np.mean(mix[training[:40]]),
                        np.mean(ig[~training[:40]]),
                        np.mean(met[~training[:40]]),
                        np.mean(mix[~training[:40]])]

    obs = {}
    obs['overall'] = [0.957012, 0.962209, 0.979878, 0.653049, 0.639826, 0.789329]

    preds['subtypes'] = []

    preds['Igneous'] = []
    for i in range(10):
        p = np.mean(ig[np.logical_and(~train, ctgrs == i)])
        preds['Igneous'].append(p)
        preds['subtypes'].append(p)

    preds['Metamorphic'] = []
    for i in range(10):
        p = np.mean(met[np.logical_and(~train, ctgrs == i)])
        preds['Metamorphic'].append(p)
        preds['subtypes'].append(p)

    preds['Mixed'] = []
    for i in range(10):
        p = np.mean(mix[np.logical_and(~train, ctgrs == i)])
        preds['Mixed'].append(p)
        preds['subtypes'].append(p)

    return preds


cnn_preds = get_preds(cnn, cnn_min.x)
mds_preds = get_preds(mds, mds_min.x)

strengths = np.array(strengths)
if len(strengths.shape) == 1:
    st = np.zeros((20, 10))
    for i, s in enumerate(strengths):
        st[i, s] = 1