import argparse
import glob
import os

import keras.layers
import numpy as np
from keras import optimizers, callbacks
from keras.applications import ResNet50, VGG16, VGG19
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_normal


# This version uses the finetuning example from Keras documentation
# instead of the bottleneck feature generation

N_classes = 2


def assert_validity(args):
    valid_models = ['resnet50', 'vgg16', 'vgg19', 'scratch']  # , 'inception_v3', 'xception']
    valid_groups = [
        'F_Ped', 'M_ped',
        'F_YA', 'M_YA',
        'F_Adult', 'M_Adult',
        'F_Ger', 'M_Ger']
    valid_positions = ['PA', 'Lateral']

    assert args.model in valid_models, '{} not a valid model name'.format(args.model)
    assert args.group in valid_groups, '{} not a valid group'.format(args.group)
    assert args.position in valid_positions, '{} not a valid position'.format(args.position)


def prep_dir(args):
    group, model, position = args.group, args.model, args.position
    model_path = 'models/{group}/{position}/{model}/'.format(group=group, position=position, model=model)
    TBlog_path = 'TBlog/{group}/{position}/{model}/'.format(group=group, position=position, model=model)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(TBlog_path, exist_ok=True)
    os.makedirs(model_path + 'top', exist_ok=True)
    os.makedirs(model_path + 'ft', exist_ok=True)
    os.makedirs(model_path + 'ft_notop', exist_ok=True)
    return model_path


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
        assert False, '{} is not an implemented model!'.format(model)

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


def get_callbacks(model, group, position, train_type):
    """
    :return: A list of `keras.callbacks.Callback` instances to apply during training.

    """
    path = 'models/{group}/{position}/{model}/{train_type}/'.format(
        group=group,
        position=position,
        model=model,
        train_type=train_type)
    return [
        callbacks.ModelCheckpoint(
            filepath=path + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
            monitor='val_acc',
            verbose=1,
            save_best_only=True),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1),
        # callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
        callbacks.TensorBoard(
            log_dir='TBlog/' + path,
            histogram_freq=1,
            write_graph=True,
            write_images=True)
    ]


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


def get_train_datagen():
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        zoom_range=0.1,
        fill_mode="nearest",
        vertical_flip=True
    )
    return datagen

def get_test_datagen():
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    return datagen


def train_top(model, group, position):
    print('Loading model...')
    full_model = get_model(model, freeze_base=True)
    full_model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    train_path = 'data/{position}/train_224x224_w_flip/{group}/train/'.format(position=position, group=group)
    test_path = 'data/{position}/train_224x224_w_flip/{group}/test/'.format(position=position, group=group)
    print('Please input top training parameters: \n')
    # Batch_size = int(input('Batch size: '))
    # N_Epochs = int(input('Epochs:'))
    Batch_size = 32
    N_Epochs = 50
    N_train_samples = count_files(train_path)
    N_test_samples = count_files(test_path)

    train_datagen = get_train_datagen()
    test_datagen = get_test_datagen()

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    # train the model on the new data for a few epochs
    print('Training top...')

    # class_weight = {0: 0.40, 1: 0.40, 2: 0.20}
    class_weight = 'auto'

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(N_train_samples / Batch_size),
        epochs=N_Epochs,
        verbose=1,
        callbacks=get_callbacks(model, group, position, train_type='top'),
        validation_data=test_generator,
        validation_steps=np.ceil(N_test_samples / Batch_size),
        class_weight=class_weight,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)

    weights_path = 'models/{group}/{position}/{model}/bottleneck_fc_model.h5'.format(position=position, group=group,
                                                                                     model=model)
    full_model.save_weights(weights_path)
    print('Model top trained.')


def fine_tune(model, group, position, weights_path):
    """

    :type model: object
    """
    train_path = 'data/{position}/train_224x224_w_flip/{group}/train/'.format(position=position, group=group)
    test_path = 'data/{position}/train_224x224_w_flip/{group}/test/'.format(position=position, group=group)
    N_train_samples = count_files(train_path)
    N_test_samples = count_files(test_path)

    print('Please input top training parameters: \n')
    # Batch_size = int(input('Batch size: '))
    # N_Epochs = int(input('Epochs:'))
    Batch_size = 32
    N_Epochs = 100

    train_datagen = get_train_datagen()
    test_datagen = get_test_datagen()

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    print('Loading model...')
    full_model = get_model(model)
    full_model.load_weights(weights_path)
    print('model weights loaded.')

    for i, layer in enumerate(full_model.layers):
        print(i, layer.name)

    N_layers_to_finetune = int(input('# of last layers to finetune:'))
    for layer in full_model.layers[-N_layers_to_finetune:]:
        layer.trainable = True
    for layer in full_model.layers[:-N_layers_to_finetune]:
        layer.trainable = False

    full_model.compile(
        optimizer=optimizers.Adam(lr=5e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    print('Fine-tuning last {} layers...'.format(N_layers_to_finetune))

    # class_weight={0:0.40, 1:0.40, 2:0.20}
    class_weight = 'auto'

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(N_train_samples / Batch_size),
        epochs=N_Epochs,
        verbose=1,
        callbacks=get_callbacks(model, group, position, train_type='ft'),
        validation_data=test_generator,
        validation_steps=np.ceil(N_test_samples / Batch_size),
        class_weight=class_weight,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)

    weights_path = 'models/{group}/{position}/{model}/finetuned_model.h5'.format(group=group, position=position,
                                                                                 model=model)
    full_model.save_weights(weights_path)
    print('Model fine-tuned.')


def ft_notop(model, group, position):
    train_path = 'data/{position}/train_224x224_w_flip/{group}/train/'.format(position=position, group=group)
    test_path = 'data/{position}/train_224x224_w_flip/{group}/test/'.format(position=position, group=group)
    N_train_samples = count_files(train_path)
    N_test_samples = count_files(test_path)

    print('Please input top training parameters: \n')
    Batch_size = int(input('Batch size: '))
    N_Epochs = int(input('Epochs:'))

    train_datagen = get_train_datagen()
    test_datagen = get_test_datagen()

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    print('Loading model...')
    full_model = get_model(model)
    print('model weights loaded.')

    for i, layer in enumerate(full_model.layers):
        print(i, layer.name)

    N_layers_to_finetune = int(input('# of last layers to finetune:'))
    for layer in full_model.layers[-N_layers_to_finetune:]:
        layer.trainable = True
    for layer in full_model.layers[:-N_layers_to_finetune]:
        layer.trainable = False

    full_model.compile(
        optimizer=optimizers.adam(lr=5e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    print('Fine-tuning last {} layers...'.format(N_layers_to_finetune))

    # class_weight = {0: 0.40, 1: 0.40, 2: 0.20}
    class_weight = 'auto'

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(N_train_samples / Batch_size),
        epochs=N_Epochs,
        verbose=1,
        callbacks=get_callbacks(model, group, position, train_type='ft_notop'),
        validation_data=test_generator,
        validation_steps=np.ceil(N_test_samples / Batch_size),
        class_weight=class_weight,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)

    weights_path = 'models/{group}/{position}/{model}/finetuned_notop_model.h5'.format(group=group, position=position,
                                                                                       model=model)
    full_model.save_weights(weights_path)
    print('Model fine-tuned.')


def train_from_scratch(group, position):
    train_path = 'data/{position}/train_224x224_w_flip/{group}/train/'.format(position=position, group=group)
    test_path = 'data/{position}/train_224x224_w_flip/{group}/test/'.format(position=position, group=group)
    N_train_samples = count_files(train_path)
    N_test_samples = count_files(test_path)

    full_model = Sequential()
    full_model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(3, 224, 224)))
    full_model.add(keras.layers.Activation('relu'))
    full_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    full_model.add(keras.layers.Conv2D(32, (3, 3)))
    full_model.add(keras.layers.Activation('relu'))
    full_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    full_model.add(keras.layers.Conv2D(64, (3, 3)))
    full_model.add(keras.layers.Activation('relu'))
    full_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    full_model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    full_model.add(keras.layers.Dense(64))
    full_model.add(keras.layers.Activation('relu'))
    full_model.add(keras.layers.Dropout(0.5))
    full_model.add(keras.layers.Dense(1))
    full_model.add(keras.layers.Activation('softmax'))

    full_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    Batch_size = 32
    N_Epochs = 100

    datagen = get_datagen()

    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=Batch_size,
        shuffle=True)

    full_model.compile(
        optimizer=optimizers.rmsprop(),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    print('Training from scratch')

    class_weight = 'auto'

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(N_train_samples / Batch_size),
        epochs=N_Epochs,
        verbose=1,
        callbacks=get_callbacks('scratch', group, position, train_type='ft'),
        validation_data=test_generator,
        validation_steps=np.ceil(N_test_samples / Batch_size),
        class_weight=class_weight,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)

    weights_path = 'models/{group}/{position}/{model}/finetuned_model.h5'.format(group=group, position=position,
                                                                          model=model)
    full_model.save_weights(weights_path)
    print('Model trained.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', help='The network eg. resnet50')
    parser.add_argument('--group', default='F_Adult', help='Demographic group')
    parser.add_argument('--position', default='PA', help='patient position')
    parser.add_argument('--train_top', action='store_true', help='train top')
    parser.add_argument('--finetune', action='store_true', help='finetune')
    parser.add_argument('--finetune_notop', action='store_true', help='finetune from random init top')

    args = parser.parse_args()
    assert_validity(args)
    model_path = prep_dir(args)
    weights_path = model_path + 'bottleneck_fc_model.h5'

    if args.model == 'scratch':
        train_from_scratch(args.group, args.position)
    if args.train_top:
        train_top(args.model, args.group, args.position)
    if args.finetune:
        fine_tune(args.model, args.group, args.position, weights_path)
    if args.finetune_notop:
        ft_notop(args.model, args.group, args.position)


if __name__ == '__main__':
    main()
