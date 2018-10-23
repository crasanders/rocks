import numpy as np
from crasanders.plot_images import plot_images
from os.path import join

nDim = 13

data_dir = 'data'
im_dir = 'images'
save_dir = join('plots', 'cnn predictions', 'supplemental')

#needed to sort observations and predictions into correct order
test_indices = np.load(join(data_dir, 'indices_test.npy'))

obs = np.loadtxt(join(data_dir, "mds_360_supplemental_dims.txt"))
obs = obs[sorted(test_indices), :]

pred = np.loadtxt(join(data_dir, 'test_predictions_supplemental_dims.txt'))
p = [(test_indices[i], r) for i, r in enumerate(pred)]
sorted_pred = np.array([r for i, r in sorted(p)])

obs_120 = np.loadtxt(join(data_dir, "mds_120_supplemental_dims.txt"))
pred_120 = np.loadtxt(join(data_dir, '120_predictions_supplemental_dims.txt'))


labels = ['Lightness', 'Grain Size', 'Roughness', 'Shininess', 'Organization', 'Chromaticity', 'Shape', 'Hue',
          'Porphyritic Texture', 'Pegmatitic Texture', 'Conchoidal Fractures', 'Holes', 'Layers']

for i in range(nDim):
    if i >= 8 and i < 11:
        sorted_pred[:, i] += 4.5
        obs[:, i] += 4.5

        pred_120[:, i] += 4.5
        obs_120[:, i] += 4.5

    elif i >= 11:
        sorted_pred[:, i] = (sorted_pred[:, i] + 5) / 10
        obs[:, i] = (obs[:, i] + 5) / 10

        pred_120[:, i] = (pred_120[:, i] + 5) / 10
        obs_120[:, i] = (obs_120[:, i] + 5) / 10

    plot_images(join(im_dir, 'Test Rocks'), sorted_pred[:, i], obs[:, i],
               'Predicted {}'.format(labels[i]), 'Observed {}'.format(labels[i]),
               correlate=True, save_file=join(save_dir, 'test set', '{}_predictions.pdf'.format(labels[i])))

    plot_images(join(im_dir, '120 Rocks Raw'), pred_120[:, i], obs_120[:, i],
               'Predicted {}'.format(labels[i]), 'Observed {}'.format(labels[i]),
               correlate=True, save_file=join(save_dir, '120 rocks', '{}_predictions_120.pdf'.format(labels[i])))
