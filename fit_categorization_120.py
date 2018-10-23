import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import basinhopping
from crasanders.gcm import GCM_cw

data_dir = 'data'

data = pd.read_csv(join(data_dir, 'categorization_120_data.txt'), sep='\t')
mds = np.loadtxt(join(data_dir, 'mds_120.txt'))
cnn = np.loadtxt(join(data_dir, 'cnn_120.txt'))
resnet = np.loadtxt(join(data_dir, 'resnet50_features_120.txt'))

training = np.array([1, 1, 0, 0] * 30, dtype=bool)
cats = np.array([i // 4 for i in range(120)])

igneous = [i for i in range(0, 10)]
metamorphic = [i for i in range(10, 20)]
mixed = [1, 2, 5, 8, 11, 14, 24, 25, 26, 28]

obs = data.query('Session == 2 and outlier == False').groupby(
    ['Condition', 'Subject', 'Subtype', 'Token', 'Item_Type'], as_index=False).mean().groupby(
    ['Condition', 'Subtype', 'Token', 'Item_Type'], as_index=False).mean()
categories = np.array([i // 2 for i in range(20)])

strengths = [i for i in range(10) for j in range(2)]

def min_gcm(c, obs, rep):
    preds = []

    for cond in [igneous, metamorphic, mixed]:
        exemplars = rep[np.logical_and(training, np.isin(cats, cond)), :]
        preds.extend(GCM_cw(10, rep.shape[1], 20, exemplars, strengths, c=c, r=2) \
                     .score(rep[np.isin(cats, cond),:], cats[:40]))

    obs['Predicted'] = preds
    df = obs.groupby(['Condition', 'Subtype', 'Item_Type'], as_index=False).mean()
    mse = np.mean((df['Predicted'] - df['Correct']) ** 2)
    return [df['Predicted'], mse]


mds_min = basinhopping(lambda c: min_gcm(c, obs.copy(), mds)[1], [1, 1])
cnn_min = basinhopping(lambda c: min_gcm(c, obs.copy(), cnn)[1], [1, 1])
resnet_min = basinhopping(lambda c: min_gcm(c, obs.copy(), resnet)[1], [1, 1])


df = obs.groupby(['Condition', 'Subtype', 'Item_Type'], as_index=False).mean()
df['MDS_Pred'] = min_gcm(mds_min.x, obs.copy(), mds)[0]
df['CNN_Pred'] = min_gcm(cnn_min.x, obs.copy(), cnn)[0]
df['Resnet_Pred'] = min_gcm(resnet_min.x, obs.copy(), resnet)[0]

data = data.merge(df[['Condition', 'Subtype', 'Item_Type', 'CNN_Pred', 'MDS_Pred', 'Resnet_Pred']])

import seaborn as sns
import matplotlib.pyplot as plt

condition = 'Igneous'

d = data.query('Session == 2 and outlier == False and Condition == "{}" and Training == False'.format(condition)) \
    .groupby(['Subject', 'Category'], as_index=False).mean()
predictions = d.groupby(['Category']).mean()

plot = sns.barplot(x='Category', y='Correct', data=d, ci=95, color='gray')
for j, p in enumerate(plot.axes.patches):
    p_x = p.get_x() + p.get_width() / 2
    p_y = predictions['MDS_Pred'][j]
    plot.axes.plot(p_x, p_y, 'x', markerfacecolor='none', markeredgecolor='black',
                   markeredgewidth=1., markersize=12)
    p_y = predictions['CNN_Pred'][j]
    plot.axes.plot(p_x, p_y, 'o', markerfacecolor='none', markeredgecolor='black',
                   markeredgewidth=1., markersize=12)
    # p_y = predictions['Resnet_Pred'][j]
    # plot.axes.plot(p_x, p_y, '*', markerfacecolor='none', markeredgecolor='black',
    #                markeredgewidth=1., markersize=12)
plt.show()
plt.close()

