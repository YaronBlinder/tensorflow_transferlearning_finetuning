import argparse
import glob
import os

import keras.layers
import numpy as np
import tensorflow as tf
from keras import optimizers, callbacks, initializers
from keras.applications import ResNet50, VGG16, VGG19, Xception, InceptionV3
# from keras.initializers import glorot_normal
from keras.models import Model, Sequential
from keras.utils.training_utils import multi_gpu_model

import multi_gpu_callbacks
from densenet121 import densenet121_model
from densenet161 import densenet161_model
from densenet169 import densenet169_model
# from keras.preprocessing.image import ImageDataGenerator
from extended_keras_image import ImageDataGenerator, standardize, scale_im, radical_preprocess, \
    random_90deg_rotation, random_crop
from metrics import mcor, recall, f1

# from keras.applications.imagenet_utils import preprocess_input


N_classes = 2


def assert_validity(args):
    valid_models = ['resnet50',
                    'vgg16',
                    'vgg19',
                    'scratch',
                    'inception_v3',
                    'xception',
                    'densenet121',
                    'densenet161',
                    'densenet169']
    valid_groups = [
        'F_Ped', 'M_ped',
        'F_YA', 'M_YA',
        'F_Adult', 'M_Adult',
        'F_Ger', 'M_Ger', 'all']
    valid_positions = ['PA', 'LAT']

    assert args.model in valid_models, '{} not a valid model name'.format(args.model)
    assert args.group in valid_groups, '{} not a valid group'.format(args.group)
    assert args.position in valid_positions, '{} not a valid position'.format(args.position)


def prep_dir(args):
    model_path = 'models/{group}/{position}/{model}/{top}/n_dense_{n_dense}/dropout_{dropout}/'.format(
        position=args.position,
        group=args.group,
        model=args.model,
        top=args.top,
        n_dense=args.n_dense,
        dropout=args.dropout)
    TBlog_path = 'TBlog/' + model_path
    weights_path = 'weights/' + model_path
    os.makedirs(TBlog_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)


def get_base_model(model, pooling=None):
    if model == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=keras.layers.Input(shape=(224, 224, 3)))

    elif model == 'vgg16':
        base_model = VGG16(
            include_top=False,
            weights='imagenet',
            pooling=pooling,
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

    elif model == 'densenet121':
        base_model = densenet121_model(
            img_rows=224,
            img_cols=224,
            color_type=3,
            num_classes=N_classes)

    elif model == 'densenet161':
        base_model = densenet161_model(
            img_rows=224,
            img_cols=224,
            color_type=3,
            num_classes=N_classes)

    elif model == 'densenet169':
        base_model = densenet169_model(
            img_rows=224,
            img_cols=224,
            color_type=3,
            num_classes=N_classes)

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


def get_callbacks(model, top, group, position, train_type, n_dense=512, dropout=False, **kwargs):
    """
    :return: A list of `keras.callbacks.Callback` instances to apply during training.

    """

    model_path = 'models/{group}/{position}/{model}/{top}/n_dense_{n_dense}/dropout_{dropout}/'.format(
        position=position,
        group=group,
        model=model,
        top=top,
        n_dense=n_dense,
        dropout=dropout)
    TBlog_path = 'TBlog/' + model_path
    weights_path = 'weights/' + model_path
    G = kwargs.get('G', None)
    if G > 1:
        base_model = kwargs.get('base_model', None)
        return [
            multi_gpu_callbacks.MultiGPUCheckpointCallback(
                filepath=weights_path + '{epoch:02d}-{val_acc:.2f}.hdf5',
                base_model=base_model,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only=True),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12,
                verbose=1),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=5,
                verbose=1),
            # callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
            # callbacks.TensorBoard(
            #     log_dir=TBlog_path,
            #     histogram_freq=1,
            #     write_graph=True,
            #     write_images=True)
        ]
    else:
        return [
            callbacks.ModelCheckpoint(
                filepath=weights_path + '{epoch:02d}-{val_acc:.2f}.hdf5',
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only=True),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12,
                verbose=1),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=5,
                verbose=1),
            # callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
            # callbacks.TensorBoard(
            #     log_dir=TBlog_path,
            #     histogram_freq=1,
            #     write_graph=True,
            #     write_images=True)
        ]


def get_model(model, top, freeze_base=False, n_dense=1024, dropout=True, pooling=None):
    assert top in ['chollet', 'waya', 'linear', 'pooled_linear', 'test'], 'top selection invalid'

    base_model = get_base_model(model, pooling=pooling)

    if model in ['densenet121', 'densenet161', 'densenet169']:
        full_model = base_model
    else:
        x = base_model.output
        if pooling is None:
            x = keras.layers.Flatten()(x)

        if top == 'chollet':
            x = keras.layers.Dense(n_dense, activation="relu")(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(n_dense, activation="relu")(x)
        elif top == 'waya':
            x = keras.layers.Dense(n_dense)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.advanced_activations.LeakyReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
        elif top == 'linear':
            x = keras.layers.Dense(n_dense)(x)
            x = keras.layers.Dropout(0.5)(x)
        elif top == 'test':
            x = keras.layers.Dense(n_dense)(x)
            x = keras.layers.Dense(n_dense)(x)
            if dropout:
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


def get_train_datagen(model, size, position):
    datagen = ImageDataGenerator()
    datagen.config['position'] = position
    if model in ['vgg16', 'vgg19', 'resnet50', 'densenet121', 'densenet161', 'densenet169']:
        size = 224
    elif model in ['inception_v3', 'xception']:
        size = 299
    else:
        pass
    datagen.config['random_crop_ratio'] = 0.9
    datagen.config['size'] = size
    datagen.set_pipeline([random_crop, scale_im, radical_preprocess, random_90deg_rotation, standardize])
    return datagen


def get_test_datagen(model, size, position):
    datagen = ImageDataGenerator()
    datagen.config['position'] = position
    if model in ['vgg16', 'vgg19', 'resnet50', 'densenet121', 'densenet161', 'densenet169']:
        size = 224
    elif model in ['inception_v3', 'xception']:
        size = 299
    else:
        pass
    datagen.config['size'] = size
    datagen.set_pipeline([scale_im, radical_preprocess, standardize])
    return datagen


def train_top(model, top, group, position, size, n_epochs, n_dense, dropout, pooling, G):
    print('Loading model...')
    full_model = get_model(model, top, freeze_base=True, n_dense=n_dense, dropout=dropout, pooling=pooling)
    full_model.compile(
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.5),
        # optimizer=optimizers.Adam(lr=1e-4),
        # optimizer=optimizers.rmsprop(),
        loss='binary_crossentropy',
        # metrics=['accuracy']
        metrics=[mcor, recall, f1])

    train_path = 'data/{position}_{size}/{group}/train/'.format(position=position, size=size, group=group)
    test_path = 'data/{position}_{size}/{group}/test/'.format(position=position, size=size, group=group)

    if model in ['densenet121', 'densenet161', 'densenet169']:
        batch_size = 16
    else:
        batch_size = 32
    n_train_samples = count_files(train_path)
    n_test_samples = count_files(test_path)

    print(train_path)
    train_datagen = get_train_datagen(model, size, position)
    test_datagen = get_test_datagen(model, size, position)

    if model in ['xception', 'inception_v3']:
        target_size = (299, 299)
    else:
        target_size = (224, 224)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=True)

    # train the model on the new data for a few epochs
    print('Training top...')
    print(
        'Network:{model}\n'.format(model=model),
        'Top:{top}\n'.format(top=top),
        'Group:{group}\n'.format(group=group),
        'Position:{position}\n'.format(position=position),
        'Im_size:{size}\n'.format(size=size),
        'N_epochs:{n_epochs}\n'.format(n_epochs=n_epochs),
        'N_dense:{n_dense}\n'.format(n_dense=n_dense),
        'Dropout:{dropout}\n'.format(dropout=dropout),
        'Pooling:{pooling}'.format(pooling=pooling))

    class_weight = None
    train_type = 'top'

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(np.ceil(n_train_samples / batch_size)),
        epochs=n_epochs,
        verbose=1,
        callbacks=get_callbacks(model, top, group, position, train_type, G, n_dense, dropout),
        validation_data=test_generator,
        validation_steps=int(np.ceil(n_test_samples / batch_size)),
        class_weight=class_weight,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=True,
        initial_epoch=0)

    weights_path = 'weights/models/{group}/{position}/{model}/{top}/n_dense_{n_dense}/dropout_{dropout}/top_trained.h5'.format(
        position=position,
        group=group,
        model=model,
        top=top,
        n_dense=n_dense,
        dropout=dropout)

    full_model.save_weights(weights_path)
    print('Model top trained.')


def fine_tune(model, top, group, position, size, weights_path):
    train_path = 'data/{position}_512_16/{group}/train/'.format(position=position, group=group)
    test_path = 'data/{position}_512_16/{group}/test/'.format(position=position, group=group)
    n_train_samples = count_files(train_path)
    n_test_samples = count_files(test_path)

    print('Please input top training parameters: \n')
    # batch_size = int(input('Batch size: '))
    # n_epochs = int(input('Epochs:'))
    batch_size = 32
    n_epochs = 100

    train_datagen = get_train_datagen(model, size, position)
    test_datagen = get_test_datagen(model, size, position)

    if model in ['xception', 'inception_v3']:
        target_size = (299, 299)
    else:
        target_size = (224, 224)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=True)

    print('Loading model...')
    full_model = get_model(model, top)
    full_model.load_weights(weights_path)
    print('model weights loaded.')

    for i, layer in enumerate(full_model.layers):
        print(i, layer.name)

    # N_layers_to_finetune = int(input('# of last layers to finetune:'))
    if model == 'resnet50':
        N_layers_to_finetune = 17
    elif model == 'vgg16':
        N_layers_to_finetune = 10
    elif model == 'vgg19':
        N_layers_to_finetune = 11
    else:
        assert False, 'you should not be here'
    for layer in full_model.layers[-N_layers_to_finetune:]:
        layer.trainable = True
    for layer in full_model.layers[:-N_layers_to_finetune]:
        layer.trainable = False

    full_model.compile(
        optimizer=optimizers.Adam(lr=1e-5),
        loss='binary_crossentropy',
        # metrics=['accuracy', f1_score, precision_score, recall_score])
        metrics=['accuracy'])

    print('Fine-tuning last {} layers...'.format(N_layers_to_finetune))

    # class_weight={0:0.40, 1:0.40, 2:0.20}
    class_weight = "auto"

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(n_train_samples / batch_size),
        epochs=n_epochs,
        verbose=1,
        callbacks=get_callbacks(model, top, group, position, train_type='ft', G),
        validation_data=test_generator,
        validation_steps=np.ceil(n_test_samples / batch_size),
        class_weight=class_weight,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)

    weights_path = 'weights/{group}_{position}_{model}_finetuned_model.h5'.format(group=group, position=position,
                                                                                  model=model)
    full_model.save_weights(weights_path)
    print('Model fine-tuned.')


def train_all(model, top, group, position, size, n_epochs, n_dense, dropout, pooling, G):
    print('Loading model...')
    print("[INFO] training with {} GPUs...".format(G))

    # train_path = 'data/{position}_{size}/{group}/train/'.format(position=position, size=size, group=group)
    # test_path = 'data/{position}_{size}/{group}/test/'.format(position=position, size=size, group=group)

    train_path = '/Radical_data/data/all/16_bit/train/'
    test_path = '/Radical_data/data/all/16_bit/test/'
    # train_path = '/Radical_data/data/all/trial/train/'
    # test_path = '/Radical_data/data/all/trial/test/'

    if model in ['densenet121', 'densenet161', 'densenet169']:
        batch_size = 16
    else:
        batch_size = 32
    n_train_samples = count_files(train_path)
    n_test_samples = count_files(test_path)

    train_datagen = get_train_datagen(model, size, position)
    test_datagen = get_test_datagen(model, size, position)

    if model in ['xception', 'inception_v3']:
        target_size = (299, 299)
    else:
        target_size = (224, 224)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size * G,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size * G,
        shuffle=True)

    # train the model on the new data for a few epochs
    print('Training top...')
    print(
        'Network:{model}\n'.format(model=model),
        'Top:{top}\n'.format(top=top),
        'Group:{group}\n'.format(group=group),
        'Position:{position}\n'.format(position=position),
        'Im_size:{size}\n'.format(size=size),
        'N_epochs:{n_epochs}\n'.format(n_epochs=n_epochs),
        'N_dense:{n_dense}\n'.format(n_dense=n_dense),
        'Dropout:{dropout}\n'.format(dropout=dropout),
        'Pooling:{pooling}'.format(pooling=pooling))

    class_weight = None
    train_type = 'top'

    if G > 1:
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            full_model = get_model(model, top, freeze_base=False, n_dense=n_dense, dropout=dropout, pooling=pooling)
        # make the model parallel
        gpu_full_model = multi_gpu_model(full_model, gpus=G)
        gpu_full_model.compile(
            # optimizer=optimizers.SGD(lr=1e-4, momentum=0.5),
            optimizer=optimizers.Adam(lr=1e-2),
            # optimizer=optimizers.rmsprop(),
            loss='binary_crossentropy',
            # metrics=['accuracy']
            metrics=[mcor, recall, f1])

        gpu_full_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=int(np.ceil(n_train_samples / (batch_size * G))),
            epochs=n_epochs,
            verbose=1,
            callbacks=get_callbacks(model, top, group, position, train_type, n_dense, dropout, G=G,
                                    base_model=full_model),
            validation_data=test_generator,
            validation_steps=int(np.ceil(n_test_samples / (batch_size * G))),
            class_weight=class_weight,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            initial_epoch=0)
    else:
        full_model = get_model(model, top, freeze_base=False, n_dense=n_dense, dropout=dropout, pooling=pooling)
        full_model.compile(
            # optimizer=optimizers.SGD(lr=1e-4, momentum=0.5),
            optimizer=optimizers.Adam(lr=1e-2),
            # optimizer=optimizers.rmsprop(),
            loss='binary_crossentropy',
            # metrics=['accuracy']
            metrics=[mcor, recall, f1]) \
 \
                full_model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=int(np.ceil(n_train_samples / (batch_size * G))),
                    epochs=n_epochs,
                    verbose=1,
                    callbacks=get_callbacks(model, top, group, position, train_type, G, n_dense, dropout),
                    validation_data=test_generator,
                    validation_steps=int(np.ceil(n_test_samples / (batch_size * G))),
                    class_weight=class_weight,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    initial_epoch=0)

    weights_path = 'weights/models/{group}/{position}/{model}/{top}/n_dense_{n_dense}/dropout_{dropout}/fully_trained.h5'.format(
        position=position,
        group=group,
        model=model,
        top=top,
        n_dense=n_dense,
        dropout=dropout)

    full_model.save_weights(weights_path)
    print('Model trained.')


def train_from_scratch(group, position, size, selu=False):
    train_path = 'data/{position}_{size}/{group}/train/'.format(position=position, size=size, group=group)
    test_path = 'data/{position}_{size}/{group}/test/'.format(position=position, size=size, group=group)
    n_train_samples = count_files(train_path)
    n_test_samples = count_files(test_path)

    if not selu:
        full_model = Sequential()
        full_model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(size, size, 3)))
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
        full_model.add(keras.layers.Dense(N_classes))
        full_model.add(keras.layers.Activation('softmax'))

        full_model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
            metrics=['accuracy'])

    else:
        full_model = Sequential()
        full_model.add(
            keras.layers.Conv2D(32, (3, 3), input_shape=(size, size, 3),
                                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=np.sqrt(1 / 9))))
        full_model.add(keras.layers.Activation('selu'))
        full_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        full_model.add(keras.layers.Conv2D(32, (3, 3),
                                           kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                        stddev=np.sqrt(1 / 9))))
        full_model.add(keras.layers.Activation('selu'))
        full_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        full_model.add(keras.layers.Conv2D(64, (3, 3),
                                           kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                        stddev=np.sqrt(1 / 9))))
        full_model.add(keras.layers.Activation('selu'))
        full_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        full_model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
        full_model.add(keras.layers.Dense(64, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1)))
        full_model.add(keras.layers.Activation('selu'))
        full_model.add(keras.layers.Dropout(0.5))
        full_model.add(keras.layers.Dense(N_classes))
        full_model.add(keras.layers.Activation('softmax'))

        full_model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
            metrics=['accuracy'])

    batch_size = 32
    n_epochs = 100

    train_datagen = get_train_datagen('scratch', size, position)
    test_datagen = get_test_datagen('scratch', size, position)

    target_size = (size, size)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        image_reader='cv2',
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=True)

    print('Training from scratch')

    # class_weight = {0: 1.5, 1: 1}
    class_weight = 'auto'

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(np.ceil(n_train_samples / batch_size)),
        epochs=n_epochs,
        verbose=1,
        callbacks=get_callbacks('scratch', 'ft_notop', group, position, train_type='ft_notop'),
        # callbacks=get_callbacks(model, top, group, position, train_type, n_dense, dropout),
        validation_data=test_generator,
        validation_steps=int(np.ceil(n_test_samples / batch_size)),
        class_weight=class_weight,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=True,
        initial_epoch=0)

    weights_path = 'weights/{group}_{position}_{model}_trained_model.h5'.format(group=group, position=position,
                                                                                model='scratch')
    full_model.save_weights(weights_path)
    print('Model trained.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg16', help='The network eg. resnet50')
    parser.add_argument('--size', default=512, help='Image size')
    parser.add_argument('--top', default='linear', help='Top classifier')
    parser.add_argument('--group', default='M_Adult', help='Demographic group')
    parser.add_argument('--position', default='PA', help='patient position')
    parser.add_argument('--train_top', action='store_true', help='train top')
    parser.add_argument('--train_all', action='store_true', help='train top')
    parser.add_argument('--finetune', action='store_true', help='finetune')
    parser.add_argument('--epochs', default=100, help='# of epochs for top training')
    parser.add_argument('--n_dense', default=512, help='size of dense layer')
    parser.add_argument('--dropout', action='store_true', help='flag for adding a dropout layer')
    parser.add_argument('--pooling', default=None, help='type of global pooling layer')
    parser.add_argument('--selu', action='store_true', help='flag for selu model (from scratch')
    parser.add_argument('--gpus', type=int, default=1, help='# of GPUs to use for training')

    args = parser.parse_args()
    assert_validity(args)
    prep_dir(args)

    n_epochs = int(args.epochs)
    size = int(args.size)
    n_dense = int(args.n_dense)
    G = int(args.gpus)

    if args.model == 'scratch':
        selu = True if args.selu is not None else False
        train_from_scratch(args.group, args.position, size, selu)
    if args.train_top:
        train_top(args.model, args.top, args.group, args.position, size, n_epochs, n_dense, args.dropout, args.pooling,
                  G)
    if args.train_all:
        train_all(args.model, args.top, args.group, args.position, size, n_epochs, n_dense, args.dropout, args.pooling,
                  G)
    if args.finetune:
        weights_path = 'weights/models/{group}/{position}/{model}/{top}/n_dense_{n_dense}/dropout_{dropout}/top_trained.h5'.format(
            position=position,
            group=group,
            model=model,
            top=top,
            n_dense=n_dense,
            dropout=dropout)
        fine_tune(args.model, args.top, args.group, args.position, size, weights_path)


if __name__ == '__main__':
    main()
