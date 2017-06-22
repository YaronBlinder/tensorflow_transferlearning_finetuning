import argparse
import os

import keras.layers
import numpy as np
from keras.applications import ResNet50, VGG16, VGG19
from keras.initializers import TruncatedNormal
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from scipy.misc import imread


N_classes = 2


def get_base_model(model):
    implemented_models = ['resnet50', 'vgg16', 'vgg19']

    assert model in implemented_models, '{} is not an implemented model!'.format(model)

    if model == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=keras.layers.Input(shape=(224, 224, 3)))

    elif model == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=keras.layers.Input(shape=(224, 224, 3)))

    elif model == 'vgg19':
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_tensor=keras.layers.Input(shape=(224, 224, 3)))

    # elif model == 'inception_v3':
    #     base_model = applications.inception_v3.InceptionV3(
    #         weights='imagenet',
    #         include_top=False,
    #         input_tensor=input_tensor)
    # elif model == 'xception':
    #     base_model = applications.xception.Xception(
    #         weights='imagenet',
    #         include_top=False,
    #         input_tensor=input_tensor)


    return base_model


def get_model(model, freeze_base=False):
    base_model = get_base_model(model)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(1024, activation="relu", kernel_initializer=glorot_normal(), trainable=True)(x)
    # x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.Dense(1024, activation="relu", kernel_initializer=glorot_normal(), trainable=True)(x)

    x = keras.layers.Dense(1024)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Dropout(0.25)(x)

    predictions = keras.layers.Dense(N_classes, activation='softmax', name='class_id', trainable=True)(x)
    full_model = Model(inputs=base_model.input, outputs=predictions)
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    return full_model


def predict(model, group, position, file):
    weights_path = 'models/{group}/{position}/{model}/finetuned_model.h5'.format(group=group, position=position,
                                                                                 model=model)
    assert os.path.exists(weights_path), 'Model not trained!'

    size = 224
    clf = get_model(model)
    clf.load_weights(weights_path)
    img = load_img(
        file,
        False,
        target_size=(size, size))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # image = imread(file)
    # image = np.reshape(image, [1, size, size, 1])
    preds = clf.predict(x)

    return preds


def ensemble(group, position, file):
    resnet50_pred = predict('resnet50', group, position, file)
    vgg16_pred = predict('vgg16', group, position, file)
    vgg19_pred = predict('vgg19', group, position, file)

    print('resnet50_pred: {}'.format(resnet50_pred))
    print('vgg16_pred: {}'.format(vgg16_pred))
    print('vgg19_pred: {}'.format(vgg19_pred))
    ens_pred = np.mean([resnet50_pred, vgg16_pred, vgg19_pred])

    return (ens_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', help='The network eg. resnet50')
    parser.add_argument('--group', default='F_Adult', help='Demographic group')
    parser.add_argument('--position', default='PA', help='patient position')
    parser.add_argument('--file', required=True, help='path to image')
    parser.add_argument('--ensemble', action='store_true', help='Flag for ensemble classification')

    args = parser.parse_args()
    model = args.model
    group = args.group
    position = args.position
    file = args.file

    if args.ensemble:
        ens_preds = ensemble(group, position, file)
        print(ens_preds)
    else:
        preds = predict(model, group, position, file)
        print(preds)


if __name__ == '__main__':
    main()
