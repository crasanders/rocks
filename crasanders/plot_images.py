import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import numpy as np

def plot_images(img_dir, x, y, xlabel, ylabel,
                img_scale=.1, fig_size=36, file_format='.png', correlate=False, save_file=None):
    fig = plt.figure(figsize=(fig_size,fig_size))
    ax = fig.add_subplot(1,1,1)
    for subdir, dirs, files in os.walk(img_dir):
        dirs.sort()
        sorted_files = sorted(filter(lambda f: f.endswith(file_format), files))
        for i, file in enumerate(sorted_files):
            if file.endswith(file_format):
                image = plt.imread(os.path.join(subdir, file))
                im = OffsetImage(image, zoom=img_scale)
                x0 = x[i]
                y0 = y[i]
                artists = []
                ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
                artists.append(ax.add_artist(ab))
                ax.update_datalim(np.column_stack([x0, y0]))
                ax.autoscale()
    for spine in ax.spines.values():
        spine.set_visible(False)
    if correlate:
        plt.plot(plt.ylim(), plt.ylim(), color='black', linestyle='--', linewidth=4)
        corr = np.corrcoef(x, y)[0,1]
        ax.annotate('r = {}'.format(round(corr, 2)), xy=(max(x), min(y)), size=50)
    plt.xlabel(xlabel, fontsize=fig_size)
    plt.ylabel(ylabel, fontsize=fig_size)
    plt.tick_params(labelsize=fig_size/2)
    if save_file is not None:
        plt.savefig(save_file)
    # plt.show()
    plt.close()


ratings = np.zeros((120, 1))
for subdir, dirs, filez in os.walk('/Users/craigsanders/Downloads/pegmatite_data'):
    files = filter(lambda f: f.endswith('.txt'), filez)
    for file in files:
        if file.endswith('.txt'):
            sub = np.loadtxt(os.path.join(subdir, file))
            for row in sub:
                ratings[int(row[2]-1)] += row[-1]
ratings /= 11