import numpy as np
from crasanders.plot_images import plot_images
from os.path import join

nDim = 8

data_dir = 'data'
im_dir = 'images'
save_dir = join('plots', 'cnn predictions', 'base')

#needed to sort observations and predictions into correct order
test_indices = np.load(join(data_dir, 'indices_test.npy'))

obs = np.loadtxt(join(data_dir, "mds_360.txt"))
obs = obs[sorted(test_indices), :]

pred = np.loadtxt(join(data_dir, 'cnn_test.txt'))
p = [(test_indices[i], r) for i, r in enumerate(pred)]
sorted_pred = np.array([r for i, r in sorted(p)])

obs_120 = np.loadtxt(join(data_dir, "mds_120.txt"))
pred_120 = np.loadtxt(join(data_dir, 'cnn_120.txt'))


labels = ['Lightness', 'Grain Size', 'Roughness', 'Shininess', 'Organization', 'Chromaticity', 'Shape', 'Hue',
          'Porphyritic Texture', 'Pegmatitic Texture', 'Conchoidal Fractures', 'Holes', 'Layers']

for i in range(nDim):
    plot_images(join(im_dir, 'Test Rocks'), sorted_pred[:, i], obs[:, i],
               'Predicted {}'.format(labels[i]), 'Observed {}'.format(labels[i]),
               correlate=True, save_file=join(save_dir, 'test set', '{}_predictions.pdf'.format(labels[i])))

    plot_images(join(im_dir, '120 Rocks Raw'), pred_120[:, i], obs_120[:, i],
               'Predicted {}'.format(labels[i]), 'Observed {}'.format(labels[i]),
               correlate=True, save_file=join(save_dir, '120 rocks', '{}_predictions_120.pdf'.format(labels[i])))
