import numpy as np
from crasanders.plot_images import plot_images
from os.path import join

nDim = 8

data_dir = 'data'
im_dir = 'images'
save_dir = join('plots', 'mds space')

mds_120 = np.loadtxt(join(data_dir, "mds_120_supplemental_dims.txt"))


labels = ['MDS Dimension 1', 'MDS Dimension 2', 'MDS Dimension 3', 'MDS Dimension 4', 'MDS Dimension 5', 'MDS Dimension 6',
          'MDS Dimension 7', 'MDS Dimension 8']

for i in range(0, nDim, 2):
    plot_images(join(im_dir, '120 Rocks Raw'), mds_120[:, i], mds_120[:, i+1],
                labels[i], labels[i+1], fig_size=8, img_scale=.05,
                correlate=False, font_size=16, save_file=join(save_dir, '{} x {} 120.pdf'.format(labels[i], labels[i+1])))

plot_images(join(im_dir, '120 Rocks Raw'), mds_120[:, 7], mds_120[:, 5],
                labels[7], labels[5], fig_size=8, img_scale=.05,
                correlate=False, font_size=16, save_file=join(save_dir, '{} x {} 120.pdf'.format(labels[7], labels[5])))