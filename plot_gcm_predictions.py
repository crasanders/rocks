import pickle
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data_dir = 'data'
save_dir = 'plots/gcm predictions'

with open(join(data_dir, 'best_gcm_fits.pkl'), 'rb') as f:
    fit = pickle.load(f)

names = [['', 'Andesite', 'Basalt', 'Diorite', 'Gabbro', 'Granite', 'Obsidian', 'Pegmatite', 'Peridotite', 'Pumice', 'Rhyolite'],
         ['', 'Amphibolite', 'Anthracite', 'Gneiss', 'Hornfels', 'Marble', 'Migmatite', 'Phyllite', 'Quartzite', 'Schist', 'Slate'],
         ['', 'Basalt', 'Diorite', 'Obsidian', 'Pumice', 'Anthracite', 'Marble', 'Dolomite', 'Micrite', 'Rock Gypsum', 'Sandstone']]

cm = [
    np.loadtxt(join(data_dir, "confusion_matrix_igneous.txt")),
    np.loadtxt(join(data_dir, "confusion_matrix_metamorphic.txt")),
    np.loadtxt(join(data_dir, "confusion_matrix_mixed.txt"))
]

categories = []
for cat in range(10):
    for token in range(4):
        row = [0 for i in range(10)]
        row[cat] = cat + 1
        categories.append(row)
categories = np.array(categories).reshape(-1)

markers = ['x', 'o', '^', '<', '>', 'v', 's', 'X', '*', 'P', 'd']

for rep in ['mds', 'cnn']:
    for model in fit.keys():
        try:
            predictions = fit[model][rep].predictions
        except:
            predictions = fit[model][rep]['predictions']

        all_obs = []
        all_pred = []
        for i, cond in enumerate(['Igneous', 'Metamorphic', 'Mixed']):
            obs = (cm[i] / cm[i].sum(1, keepdims=True)).reshape(-1)
            for o in obs:
                all_obs.append(o)
            pred = predictions[i].reshape(-1)
            for p in pred:
                all_pred.append(p)

            category = 0
            mask = categories == category
            plt.scatter(pred[mask], obs[mask], marker=markers[category], facecolors='black', edgecolors='black', s=25)
            for category in range(1, 11):
                mask = categories == category
                plt.scatter(pred[mask], obs[mask], marker=markers[category], facecolors='none', edgecolors='black', label=names[i][category], s=64)
            plt.plot(plt.ylim(), plt.ylim(), color='black', linestyle='--')
            plt.legend(loc='lower right')
            plt.ylabel('Observed Classification Probability', fontsize=18)
            plt.xlabel('Predicted Classification Probability', fontsize=18)
            plt.title('{} Condition'.format(cond), fontsize=18)
            plt.tick_params(labelsize=9)
            plt.savefig(join(save_dir, '{}_{}_{}.pdf'.format(model, rep, cond)))
            # plt.show()
            plt.close()
        print(model, rep, r2_score(all_obs, all_pred))
