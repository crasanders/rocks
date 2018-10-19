from keras.applications import *
from keras.preprocessing import image
import keras.backend as K
import os
import numpy as np

def extract_features(directory, network, nPixels=None, pooling=None, file_format='.jpg'):
    """"Takes as input a directory of images and returns their penultimate-layer representations
    from a keras CNN."""

    if nPixels is None:
        if network == 'inception_v3' or network == 'inception_resnet_v2':
            nPixels = 299

        elif network == 'nasnetlarge':
            nPixels = 331

        elif network == 'pixels':
            raise ValueError('nPixels must be specified when extracting raw pixel values.')

        else:
            nPixels = 224

    X = []
    for subdir, dirs, files in os.walk(directory):
        dirs.sort()
        sorted_files = sorted(filter(lambda f: f.endswith(file_format), files))
        for file in sorted_files:
            if file.endswith(file_format):
                img = image.load_img(os.path.join(subdir, file), target_size=(nPixels, nPixels))
                x = image.img_to_array(img)
                X.append(x)
    X = np.stack(X)

    network = network.lower()

    if network == 'pixels':
        return  X.reshape((X.shape[0], -1))

    elif network == 'xception':
        model = xception.Xception(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = xception.preprocess_input

    elif network == 'vgg16':
        model = vgg16.VGG16(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = vgg16.preprocess_input

    elif network == 'vgg19':
        model = vgg19.VGG19(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = vgg19.preprocess_input

    elif network == 'resnet50':
        model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = resnet50.preprocess_input

    elif network == 'inception_v3':
        model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = inception_v3.preprocess_input

    elif network == 'inception_resnet_v2':
        model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = inception_resnet_v2.preprocess_input

    elif network == 'mobilenet':
        model = mobilenet.MobileNet(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = mobilenet.preprocess_input

    elif network == 'densenet121':
        model = densenet.DenseNet121(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = densenet.preprocess_input

    elif network == 'densenet169':
        model = densenet.DenseNet169(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = densenet.preprocess_input

    elif network == 'densenet201':
        model = densenet.DenseNet201(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = densenet.preprocess_input

    elif network == 'nasnetlarge':
        model = nasnet.NASNetLarge(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = nasnet.preprocess_input

    elif network == 'nasnetmobile':
        model = nasnet.NASNetMobile(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = nasnet.preprocess_input

    elif network == 'mobilenetv2':
        model = mobilenetv2.MobileNetV2(weights='imagenet', include_top=False, pooling=pooling)
        preprocessor = mobilenetv2.preprocess_input

    else:
        raise ValueError("{} is not a recognized network.".format(network))

    features = model.predict(preprocessor(X))
    K.clear_session()

    return features