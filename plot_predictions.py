import numpy as np
from crasanders.plot_images import plot_images
from os.path import join

nDim = 13

data_dir = 'data'
im_dir = 'images'
save_dir = join('plots', 'cnn predictions', 'augmented')

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
   # plot_images(join(im_dir, 'Test Rocks'), sorted_pred[:, i], obs[:, i],
   #             'Predicted {}'.format(labels[i]), 'Observed {}'.format(labels[i]),
   #             correlate=True, save_file=join(save_dir, 'test set', 'augmented_{}_predictions.pdf'.format(labels[i])))

   plot_images(join(im_dir, '120 Rocks Raw'), pred_120[:, i], obs_120[:, i],
               'Predicted {}'.format(labels[i]), 'Observed {}'.format(labels[i]),
               correlate=True, save_file=join(save_dir, '120 rocks', 'augmented_{}_predictions_120.pdf'.format(labels[i])))

# for i in range(0, 12, 2):
#     plot_images('120 Rocks Raw', rocks120[:,i], rocks120[:,i+1], '', '', correlate=False, save_file='rocks120_{}_{}_predictions.png'.format(i,i+1))
#
# plot_images('120 Rocks Raw', rocks120[:,11], rocks120[:,12], '', '', correlate=False, save_file='rocks120_11_12_predictions.png')


# x = np.arange(0, 10, .1)
#
# def transform(x, ref, u, p, q):
#     x_t = x.copy()
#     x_t[x >= ref] = ref + u * (x_t[x >= ref] - ref)**p
#     x_t[x < ref] = ref - (ref - x_t[x < ref])**q
#     return x_t
#
#
# def logistic(x, m, L, k):
#     return L / (1 + np.exp(-k * (x - m)))
#
# for i in range(10):
#     plt.plot(x, logistic(x, 5, 9, i))
# plt.show()
# plt.close()