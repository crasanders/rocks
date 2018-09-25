# This program trains the ensemble of CNNs model reported in the manuscript

# Import modules

from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import backend as K

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load data
nTest = 90
nPixels = 224

mds_360 = np.loadtxt("mds_360_supplemental_dims.txt")
categories = [i for i in range(30) for j in range(12)]


def load_images(directory, nPixels, preprocesser):
    X = []
    for subdir, dirs, files in os.walk(directory):
        for file in sorted(files):
            if file.endswith(".jpg"):
                img = load_img(os.path.join(subdir, file), target_size=(nPixels, nPixels))
                x = img_to_array(img)
                X.append(x)
    X = np.stack(X)
    X = preprocesser(X)
    return X


X = load_images("360 Rocks", nPixels, lambda x: resnet50.preprocess_input(np.expand_dims(x, axis=0)).squeeze())
(X_train_, X_test,
 Y_train_, Y_test,
 categories_train_, categories_test) = train_test_split(X,
                                                        mds_360,
                                                        categories,
                                                        test_size=nTest,
                                                        stratify=categories,
                                                        random_state=0)

(X_train, X_validate,
 Y_train, Y_validate) = train_test_split(X_train_,
                                         Y_train_,
                                         test_size=nTest,
                                         stratify=categories_train_,
                                         random_state=0)

X_120 = load_images("120 Rocks", nPixels, lambda x: resnet50.preprocess_input(np.expand_dims(x, axis=0)).squeeze())

# Set hyperparameters
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=True)

nDim = 13
nEpochs = 500
dropout = 0.5
nEnsemble = 10

nDense = 512
nLayers = 3
loglr = -2.05142224482199

lr = 10 ** loglr

batch_size = 30

# Train models
for e in range(nEnsemble):
    print('Training Ensemble {}'.format(e))

    # Build model
    arch = resnet50.ResNet50(include_top=False, pooling='avg')
    for layer in arch.layers:
        layer.trainable = False

    x = arch.output
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    for lyr in range(nLayers):
        x = Dense(nDense, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    x = Dense(nDim)(x)

    model = Model(inputs=arch.input, outputs=x)

    # Initial training
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))

    checkpoint1 = ModelCheckpoint('intermediate_model.hdf5', save_best_only=True)

    hist1 = model.fit_generator(datagen.flow(X_train, Y_train, batch_size),
                                steps_per_epoch=len(X_train) / batch_size,
                                epochs=nEpochs,
                                validation_data=(X_validate, Y_validate),
                                callbacks=[checkpoint1],
                                verbose=False)

    # Fine tuning
    model = load_model("intermediate_model.hdf5")

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mean_squared_error')

    checkpoint2 = ModelCheckpoint('ensemble_{}.hdf5'.format(e), save_best_only=True)

    hist2 = model.fit_generator(datagen.flow(X_train, Y_train, batch_size),
                                steps_per_epoch=len(X_train) / batch_size,
                                epochs=nEpochs,
                                validation_data=(X_validate, Y_validate),
                                callbacks=[checkpoint2],
                                verbose=False)

    K.clear_session()  # Clear tensorflow session to prevent memory issues

# Get predictions for validation and training sets
validate_pred = np.zeros((nEnsemble, nTest, nDim))
test_pred = np.zeros((nEnsemble, nTest, nDim))
rocks_120_pred = np.zeros((nEnsemble, 120, nDim))
for e in range(nEnsemble):
    model = load_model("ensemble_{}.hdf5".format(e))
    validate_pred[e, :] = model.predict(X_validate)
    test_pred[e, :] = model.predict(X_test)
    rocks_120_pred[e, :] = model.predict(X_120)

    K.clear_session()

validate_prediction = np.mean(validate_pred, 0)
test_prediction = np.mean(test_pred, 0)
rocks_120_prediction = np.mean(rocks_120_pred, 0)

np.savetxt('test_predictions_supplemental_dims.txt', test_prediction, fmt='%.18f')
np.savetxt('120_predictions_supplemental_dims.txt', rocks_120_prediction, fmt='%.18f')


# Get MSE
print(mean_squared_error(Y_validate, validate_prediction))
print(mean_squared_error(Y_test, test_prediction))

# Get R2
print(r2_score(Y_validate, validate_prediction))
print(r2_score(Y_test, test_prediction))
