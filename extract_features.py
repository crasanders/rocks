from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from scipy.spatial.distance import pdist
import numpy as np
import os

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
directory = '120 Rocks'
nPixels = 224
nRocks = 120
nDim = 8

X = []
for subdir, dirs, files in os.walk(directory):
    for file in sorted(files):
        if file.endswith(".jpg"):
            img = image.load_img(os.path.join(subdir, file), target_size=(nPixels, nPixels))
            x = image.img_to_array(img)
            X.append(x)
X = np.stack(X)
X = preprocess_input(X)

image_vectors = X.reshape((nRocks, -1))
pixel_dist = pdist(image_vectors)
np.savetxt('pixel_120.txt', image_vectors, fmt='%.18f')
np.savetxt('pixel_dist_120.txt', pixel_dist, fmt='%.18f')

features = model.predict(X)
feature_dist =  pdist(features)
np.savetxt('resnet50_features_120.txt', features, fmt='%.18f')
np.savetxt('resnet50_features_dist_120.txt', feature_dist, fmt='%.18f')
