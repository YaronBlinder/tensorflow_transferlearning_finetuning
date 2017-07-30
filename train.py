import argparse
import glob
import os

import keras.layers
import numpy as np
from keras import optimizers, callbacks
from keras.applications import ResNet50, VGG16, VGG19, Xception, InceptionV3
# from keras.initializers import glorot_normal
from keras.models import Model, Sequential

# from keras.preprocessing.image import ImageDataGenerator
from extended_keras_image import ImageDataGenerator, standardize, scale_im, radical_preprocess, \
    imagenet_preprocess, random_90deg_rotation

# from keras.applications.imagenet_utils import preprocess_input

# This version uses the finetuning example from Keras documentation
# instead of the bottleneck feature generation

N_classes = 2


def assert_validity(args):
    valid_models = ['resnet50', 'vgg16', 'vgg19', 'scratch', 'inception_v3', 'xception']
    valid_groups = [
        'F_Ped', 'M_ped',
        'F_YA', 'M_YA',
        'F_Adult', 'M_Adult',
        'F_Ger', 'M_Ger']
    valid_positions = ['PA', 'LAT']

    assert args.model in valid_models, '{} not a valid model name'.format(args.model)
    assert args.group in valid_groups, '{} not a valid group'.format(args.group)
    assert args.position in valid_positions, '{} not a valid position'.format(args.position)


def prep_dir(args):
    group, model, position, top = args.group, args.model, args.position, args.top
    TBlog_path = 'TBlog/models/{group}/{position}/{model}/{top}/'.format(group=group, position=position, model=model,
                                                                         top=top)
    os.makedirs(TBlog_path, exist_ok=True)
    os.makedirs('weights', exist_ok=True)


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


def get_callbacks(model, top, group, position, train_type):
    """
    :return: A list of `keras.callbacks.Callback` instances to apply during training.

    """
    path = 'models/{group}/{position}/{model}/{top}/{train_type}/'.format(
        group=group,
        position=position,
        model=model,
        top=top,
        train_type=train_type)
    return [
        # callbacks.ModelCheckpoint(
        #     filepath=path + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
        #     monitor='val_acc',
        #     verbose=1,
        #     save_best_only=True),
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
        callbacks.TensorBoard(
            log_dir='TBlog/' + path,
            histogram_freq=1,
            write_graph=True,
            write_images=True)
    ]


def get_model(model, top, freeze_base=False):
    assert top in ['chollet', 'waya', 'linear'], 'top selection invalid'

    base_model = get_base_model(model)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    if top == 'chollet':
        x = keras.layers.Dense(1024, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(1024, activation="relu")(x)
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


def get_train_datagen(model, position, size):
    datagen = ImageDataGenerator()
    datagen.config['position'] = position
    if model in ['vgg16', 'vgg19', 'resnet50']:
        size = 224
        datagen.config['size'] = size
        datagen.set_pipeline([scale_im, imagenet_preprocess, random_90deg_rotation, standardize])
    else:
        datagen.config['size'] = size
        datagen.set_pipeline([scale_im, radical_preprocess, random_90deg_rotation, standardize])
    return datagen


def get_test_datagen(model, position, size):
    datagen = ImageDataGenerator()
    datagen.config['position'] = position
    if model in ['vgg16', 'vgg19', 'resnet50']:
        size = 224
        datagen.config['size'] = size
        datagen.set_pipeline([scale_im, imagenet_preprocess, standardize])
    else:
        datagen.config['size'] = size
        datagen.set_pipeline([scale_im, radical_preprocess, standardize])
    return datagen


def train_top(model, top, group, position, size, n_epochs):
    print('Loading model...')
    full_model = get_model(model, top, freeze_base=True)
    full_model.compile(
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.5),
        # optimizer=optimizers.Adam(lr=1e-4),
        # optimizer=optimizers.rmsprop(),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    train_path = 'data/{position}_{size}_16/{group}/train/'.format(position=position, size=size, group=group)
    test_path = 'data/{position}_{size}_16/{group}/test/'.format(position=position, size=size, group=group)

    batch_size = 32
    n_train_samples = count_files(train_path)
    n_test_samples = count_files(test_path)

    print(train_path)
    train_datagen = get_train_datagen(model, position)
    test_datagen = get_test_datagen(model, position)

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

    class_weight = 'auto'

    full_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(n_train_samples / batch_size),
        epochs=n_epochs,
        verbose=1,
        callbacks=get_callbacks(model, top, group, position, train_type='top'),
        validation_data=test_generator,
        validation_steps=np.ceil(n_test_samples / batch_size),
        class_weight=class_weight,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)

    weights_path = 'weights/{group}_{position}_{model}_{top}_top_trained.h5'.format(position=position, group=group,
                                                                                    model=model, top=top)
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

    train_datagen = get_train_datagen(model, position)
    test_datagen = get_test_datagen(model, position)

    if model in ['xception', 'inception_v3']:
        target_size = (299, 299)
    else:
        target_size = (224, 224)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        # target_size=(224, 224),

        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        # target_size=(224, 224),
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
        callbacks=get_callbacks(model, top, group, position, train_type='ft'),
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


def train_from_scratch(group, position, size):
    train_path = 'data/{position}_{size}_16/{group}/train/'.format(position=position, size=size, group=group)
    test_path = 'data/{position}_{size}_16/{group}/test/'.format(position=position, size=size, group=group)
    n_train_samples = count_files(train_path)
    n_test_samples = count_files(test_path)

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
        steps_per_epoch=np.ceil(n_train_samples / batch_size),
        epochs=n_epochs,
        verbose=1,
        callbacks=get_callbacks('scratch', 'ft_notop', group, position, train_type='ft_notop'),
        validation_data=test_generator,
        validation_steps=np.ceil(n_test_samples / batch_size),
        class_weight=class_weight,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
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
    parser.add_argument('--finetune', action='store_true', help='finetune')
    parser.add_argument('--epochs', default=100, help='# of epochs for top training')

    args = parser.parse_args()
    assert_validity(args)
    weights_path = 'weights/{group}_{position}_{model}_{top}_top_trained.h5'.format(position=args.position,
                                                                                    group=args.group,
                                                                                    model=args.model, top=args.top)
    n_epochs = int(args.epochs)
    size = int(args.size)

    if args.model == 'scratch':
        train_from_scratch(args.group, args.position, size)
    if args.train_top:
        train_top(args.model, args.top, args.group, args.position, size, n_epochs)
    if args.finetune:
        fine_tune(args.model, args.top, args.group, args.position, size, weights_path)


if __name__ == '__main__':
    main()
