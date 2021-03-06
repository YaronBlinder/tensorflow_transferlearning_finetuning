import os
from keras.applications import ResNet50, VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, callbacks
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense
from keras.models import Model
import argparse
import tensorflow as tf
import numpy as np
import glob

N_classes = 3

def assert_validity(args):
    valid_models = ['resnet50', 'vgg16', 'vgg19', 'inception_v3', 'xception']
    valid_groups = [
        'F_Ped', 'M_ped',
        'F_YA', 'M_YA',
        'F_Adult', 'M_Adult',
        'F_Ger', 'M_Ger']

    assert args.model in valid_models, '{} not a valid model name'.format(args.model)
    assert args.group in valid_groups, '{} not a valid group'.format(args.group)


def prep_dir(args):
    group, model = args.group, args.model
    model_path = 'models/' + group + '/' + model + '/'
    if not os.path.exists(model_path):
        os.mkdir('models')
        os.mkdir('models/'+group)
        os.mkdir(model_path)
    return model_path


def get_base_model(model):
    if model == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
         input_tensor=Input(shape=(224, 224, 3)))

    elif model == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

    # elif model == 'vgg19':
    #     base_model = applications.vgg19.VGG19(
    #         weights='imagenet',
    #         include_top=False,
    #         input_tensor=input_tensor)
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
    else:
        print('You should not be here')

    return base_model


def one_hot_labels(labels):
    one_hot = np.zeros((labels.size, N_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def count_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
        return cnt


def get_callbacks(model, group):
    """
    :return: A list of `keras.callbacks.Callback` instances to apply during training.

    """
    path = 'models/' + group + '/' + model + '/'
    return [
        callbacks.ModelCheckpoint(
            filepath=path+'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_acc',
            verbose=1,
            save_best_only=True),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=2,
            verbose=1),
        # callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
        callbacks.TensorBoard(
            log_dir='TBlog/',
            histogram_freq=4,
            write_graph=True,
            write_images=True)
    ]


def generate_bn_features(model, group):
    model = get_base_model(model)
    train_path = 'data/train_224x224/' + group + '/train/'
    test_path = 'data/train_224x224/' + group + '/test/'
    batch_size = Batch_size
    n_steps_train = np.ceil(count_files(train_path) / batch_size)
    n_steps_test = np.ceil(count_files(test_path) / batch_size)
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        directory=train_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator=train_generator,
        steps=n_steps_train,
        workers=4,
        verbose=1)
    np.save('models/' + group + '/bottleneck_features_train',
            bottleneck_features_train)
    np.save('models/' + group + '/train_classes', train_generator.classes)

    test_generator = datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_test = model.predict_generator(
        generator=test_generator,
        steps=n_steps_test,
        workers=4,
        verbose=1)
    np.save('models/' + group + '/bottleneck_features_test',
            bottleneck_features_test)
    np.save('models/' + group + '/test_classes', test_generator.classes)


def train_top_only(model, group):
    base_model = get_base_model(model)
    weights_path = 'models/' + group + '/' + model + '/bottleneck_fc_model.h5'
    train_data = np.load('models/' + group + '/bottleneck_features_train.npy')
    train_labels = one_hot_labels(
        np.load('models/' + group + '/train_classes.npy'))
    test_data = np.load('models/' + group + '/bottleneck_features_test.npy')
    test_labels = one_hot_labels(
        np.load('models/' + group + '/test_classes.npy'))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', name='fcc_0'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(N_classes, activation='softmax', name='class_id'))
    print('Model bottom loaded.')

    top_model.compile(
        optimizer='SGD',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    print('Please input top training parameters: \n')
    Batch_size = input('Batch size: ')
    N_Epochs = input('Epochs:')

    top_model.fit(
        x=train_data,
        y=train_labels,
        epochs=N_Epochs,
        batch_size=Batch_size,
        validation_data=(test_data, test_labels),
        callbacks=get_callbacks(),
        verbose=1)

    top_model.save_weights(weights_path)
    print('Model top trained.')


def fine_tune(model, group):

    weights_path = 'models/' + group + '/' + model + '/bottleneck_fc_model.h5'
    train_path = 'data/train_224x224/' + group + '/train/'
    test_path = 'data/train_224x224/' + group + '/test/'

    N_train_samples = count_files(train_path)
    N_test_samples = count_files(test_path)

    base_model = get_base_model(model)
    print('Model bottom loaded.')
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', name='fcc_0'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(N_classes, activation='softmax', name='class_id'))
    top_model.load_weights(weights_path)
    full_model = Model(inputs=base_model.input,
                  outputs=top_model(base_model.output))

    print('Please input fine-tuning parameters: \n')
    Batch_size = input('Batch size: ')
    N_Epochs = input('Epochs:')
    N_layers_to_finetune = input('# of last layers to finetune:')

    for layer in full_model.layers[-N_layers_to_finetune:]:
        layer.trainable = False
    for layer in full_model.layers[:-N_layers_to_finetune]:
        layer.trainable = True

    full_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        class_mode='categorical',
        shuffle=True)

    test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        class_mode='categorical',
        shuffle=True)

    # fine-tune the model
    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(N_train_samples / Batch_size),
        epochs=N_epochs,
        verbose=1,
        callbacks=get_callbacks(model, group),
        validation_data=test_generator,
        validation_steps=np.ceil(N_test_samples / Batch_size),
        class_weight=None,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50',
                        help='The network eg. resnet50')
    parser.add_argument('--group', default='F_Adult', help='Demographic group')
    parser.add_argument('--generate_bn_features', action='store_true',
                        help='Flag to generate bottleneck features')
    parser.add_argument('--train_top_only',
                        action='store_true',  help='Flag to retrain')
    parser.add_argument('--finetune', action='store_true',
                        help='Flag to fine tune')

    args = parser.parse_args()

    assert_validity(args)
    model_path = prep_dir(args)
    bn_features_path = model_path + 'bottleneck_features_train.npy'
    weights_path = model_path + 'bottleneck_fc_model.h5'

    if args.generate_bn_features:
        generate_bn_features(model, group)

    if args.train_top_only:
        if not os.path.exists(bn_features_path):
            print('Bottleneck features file not found! Generate first.')
        else:
            train_top_only(model, group)

    if args.finetune:
        if not os.path.exists(weights_path):
            print('Weights file not found! Train top first.')
        else:
            print('Fine tuning:')
            fine_tune(model, group)

    if not args.train_top_only and not args.fine_tune:
        print('No retraining selected.')


if __name__ == '__main__':
    main()
