import numpy as np
from crasanders.plot_images import plot_images
from os.path import join

nDim = 13

data_dir = 'data'
im_dir = 'images'
save_dir = join('plots', 'cnn predictions')

#needed to sort observations and predictions into correct order
test_indices = np.load(join(data_dir, 'indices_test.npy'))

obs = np.loadtxt(join(data_dir, "mds_360_supplemental_dims.txt"))
obs = obs[sorted(test_indices), :]

pred = np.hstack((np.loadtxt(join(data_dir, 'cnn_test.txt')),
                  np.loadtxt(join(data_dir, 'test_predictions_supplemental_dims.txt'))[:,-5:]))

p = [(test_indices[i], r) for i, r in enumerate(pred)]
sorted_pred = np.array([r for i, r in sorted(p)])

obs_120 = np.loadtxt(join(data_dir, "mds_120_supplemental_dims.txt"))
pred_120 = np.hstack((np.loadtxt(join(data_dir, 'cnn_120.txt')),
                      np.loadtxt(join(data_dir, '120_predictions_supplemental_dims.txt'))[:,-5:]))


labels = ['Lightness', 'Grain Size', 'Roughness', 'Shininess', 'Organization', 'Chromaticity', 'Shape', 'Hue',
          'Porphyritic Texture', 'Pegmatitic Texture', 'Conchoidal Fractures', 'Holes', 'Layers']

for i in range(nDim):
    plot_images(join(im_dir, 'Test Rocks'), sorted_pred[:, i], obs[:, i],
                'CNN-Predicted {}'.format(labels[i]), 'MDS-Derived {}'.format(labels[i]), fig_size=8, img_scale=.05,
                correlate=True, font_size=24, save_file=join(save_dir, 'test set', '{}_predictions.pdf'.format(labels[i])))

    plot_images(join(im_dir, '120 Rocks Raw'), pred_120[:, i], obs_120[:, i],
                'CNN-Predicted {}'.format(labels[i]), 'MDS-Derived {}'.format(labels[i]), fig_size=8, img_scale=.05,
                correlate=True, font_size=24, save_file=join(save_dir, '120 rocks', '{}_predictions_120.pdf'.format(labels[i])))