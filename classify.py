import argparse
import os

import keras.layers
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.applications import ResNet50, VGG16, VGG19
from keras.initializers import TruncatedNormal
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from scipy.misc import imread


N_classes = 2


def get_base_model(model):
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

    elif model == 'inception_v3':
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_tensor=keras.layers.Input(shape=(299, 299, 3)))

    elif model == 'xception':
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_tensor=keras.layers.Input(shape=(299, 299, 3)))

    else:
        assert False, '{} is not an implemented model!'.format(model)

    return base_model


def get_model(model, top, freeze_base=False):
    assert top in ['chollet', 'waya', 'linear'], 'top selection invalid'

    base_model = get_base_model(model)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    if top == 'chollet':
        x = keras.layers.Dense(1024, activation="relu", kernel_initializer=glorot_normal(), trainable=True)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(1024, activation="relu", kernel_initializer=glorot_normal(), trainable=True)(x)
    elif top == 'waya':
        x = keras.layers.Dense(1024)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.advanced_activations.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)
    elif top == 'linear':
        x = keras.layers.Dense(256)(x)
        x = keras.layers.Dropout(0.5)(x)
    else:
        assert False, 'you should not be here'

    predictions = keras.layers.Dense(N_classes, activation='softmax', name='class_id')(x)

    # predictions = keras.layers.Dense(N_classes, activation='softmax', name='class_id', trainable=True)(x)
    full_model = Model(inputs=base_model.input, outputs=predictions)
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    return full_model


def predict(model, group, position, file):
    # weights_path = 'models/{group}/{position}/{model}/bottleneck_fc_model.h5'.format(group=group, position=position,
    #                                                                              model=model)
    weights_path = 'saved_models/{model}/{top}/top_trained.h5'.format(model=model, top=top)
    assert os.path.exists(weights_path), 'Model not trained! ({})'.format(weights_path)

    size = 224
    clf = get_model(model)
    clf.load_weights(weights_path)
    if file == None:
        pass
    else:
        img = load_img(
            file,
            False,
            target_size=(size, size))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        preds = clf.predict(x)

    return preds


def ensemble(group, position, file):
    resnet50_pred = predict('resnet50', group, position, file)
    vgg16_pred = predict('vgg16', group, position, file)
    vgg19_pred = predict('vgg19', group, position, file)

    # print('resnet50_pred: {}'.format(resnet50_pred))
    # print('vgg16_pred: {}'.format(vgg16_pred))
    # print('vgg19_pred: {}'.format(vgg19_pred))

    ens_pred = np.mean([resnet50_pred, vgg16_pred, vgg19_pred], axis=0)

    return (ens_pred)


def ensemble_all(group, position):


    data_path = 'data/{position}/train_256_3ch_flip/{group}/test/'.format(position=position, group=group)

    rows_list = []
    for filename in tqdm(os.listdir(data_path+'1')[:2]):
        tmp_dict = {}
        file_path = data_path + '1/' + filename
        id = filename.split('.')[0]
        label = 0
        ensemble_score = ensemble(group, position, file_path)
        resnet50_score = predict('resnet50', group, position, file_path)
        vgg16_score = predict('vgg16', group, position, file_path)
        vgg19_score = predict('vgg19', group, position, file_path)
        tmp_dict.update({
            'id':id,
            'label':label,
            'ensemble_score':ensemble_score,
            'resnet50_score':resnet50_score,
            'vgg16_score':vgg16_score,
            'vgg19_score':vgg19_score})
        rows_list.append(tmp_dict)


    for filename in tqdm(os.listdir(data_path+'other')[:2]):
        tmp_dict = {}
        file_path = data_path + 'other/' + filename
        id = filename.split('.')[0]
        label = 1
        ensemble_score = ensemble(group, position, file_path)
        resnet50_score = predict('resnet50', group, position, file_path)
        vgg16_score = predict('vgg16', group, position, file_path)
        vgg19_score = predict('vgg19', group, position, file_path)
        tmp_dict.update({
            'id': id,
            'label': label,
            'ensemble_score': ensemble_score,
            'resnet50_score': resnet50_score,
            'vgg16_score': vgg16_score,
            'vgg19_score': vgg19_score})
        rows_list.append(tmp_dict)

    df = pd.DataFrame(rows_list, columns=['id', 'label', 'ensemble_score', 'resnet50_score', 'vgg16_score', 'vgg19_score'])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', help='The network eg. resnet50')
    parser.add_argument('--top', default='chollet', help='Top classifier architecture')
    parser.add_argument('--group', default='F_Adult', help='Demographic group')
    parser.add_argument('--position', default='PA', help='patient position')
    parser.add_argument('--file', default=None, help='path to image')
    parser.add_argument('--ensemble', action='store_true', help='Flag for ensemble classification')
    parser.add_argument('--ensemble_all', action='store_true', help='Flag to go over test set and produce dataframe')


    args = parser.parse_args()
    model = args.model
    group = args.group
    position = args.position
    file = args.file

    if args.ensemble_all:
        df = ensemble_all(group, position)
        df.to_csv('ensemble_{group}_{position}.csv'.format(group=group, position=position))
    elif args.ensemble:
        ens_preds = ensemble(group, top, position, file)
        print('ensemble pred: {}'.format(ens_preds))
    else:
        preds = predict(model, top, group, position, file)
        print(preds)


if __name__ == '__main__':
    main()
