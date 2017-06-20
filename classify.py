import os
import argparse
import numpy as np
from keras.applications import ResNet50, VGG16, VGG19
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.initializers import TruncatedNormal

from scipy.misc import imread



def get_base_model(model):

    implemented_models = ['resnet50', 'vgg16', 'vgg19']

    assert model in implemented_models, '{} is not an implemented model!'.format(model)

    if model == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

    elif model == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

    elif model == 'vgg19':
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

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
    x = Flatten()(x)
    x = Dense(
        1024,
        activation='relu',
        kernel_initializer=TruncatedNormal(),
        name='fcc_0')(x)
    x = Dropout(0.5)(x)
    x = Dense(
        1024,
        activation='relu',
        kernel_initializer=TruncatedNormal(),
        name='fcc_1')(x)
    predictions = Dense(N_classes, activation='softmax', name='class_id')(x)
    full_model = Model(input=base_model.input, output=predictions)
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    return full_model


def predict(model, group, file):

    weights_path = 'models/' + group + '/' + model + '/finetuned_model.h5'

    Assert os.path.exists(weights_path), 'Model not trained!'

    clf = get_model(model)
    clf.load_weights(weights_path)
    image = imread(file)
    preds = clf.predict(image)

    return preds


def ensemble(group, file):

    resnet50_pred = predict('resnet50', group, file)
    vgg16_pred = predict('vgg16', group, file)
    vgg19_pred = predict('vgg19', group, file)

    ens_pred = np.mean([resnet50_pred, vgg16_pred, vgg19_pred])

    return(ens_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', help='The network eg. resnet50')
    parser.add_argument('--group', default='F_Adult', help='Demographic group')
    # parser.add_argument('--position', default='PA', help='Positional argument')
    parser.add_argument('--file', required=True, help='path to image')
    parser.add_argument('--ensemble', action='store_true', help='Flag for ensemble classification')

    args = parser.parse_args()
    model = args.model
    group = args.group
    file = args.file

    if args.ensemble:
        ens_preds = ensemble(group, file)
        print(ens_preds)
    else:
        preds = predict(model, group, file)
        print(preds)



if __name__ == '__main__':
    main()
