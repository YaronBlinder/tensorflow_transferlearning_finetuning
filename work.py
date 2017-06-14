import os
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, callbacks
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense
from keras.models import Model
import argparse
import tensorflow as tf
import numpy as np
import glob

N_classes = 5
Batch_size = 32
N_layers_to_finetune = 33
N_epochs = 50

# def choose_net(network):
#     MAP = {
#         # 'vggf'     : vggf,
#         # 'caffenet' : caffenet,
#         # 'vgg16'    : vgg16,
#         # 'vgg19'    : vgg19,
#         # 'googlenet': googlenet,
#         'resnet50' : resnet50,
#         # 'resnet152': resnet152,
#         # 'inceptionv3': inceptionv3,
#     }
#
#     if network == 'caffenet':
#         size = 227
#     elif network == 'inceptionv3':
#         size = 299
#     else:
#         size = 224
#
#     #placeholder to pass image
#     input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')
#
#     # return MAP[network](input_image), input_image


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


def get_callbacks():
    """
    :return: A list of `keras.callbacks.Callback` instances to apply during training.

    """
    return [
        # callbacks.ModelCheckpoint(
        #     model_checkpoint, monitor='val_acc', verbose=1, save_best_only=True),
        callbacks.EarlyStopping(monitor='val_loss', patience=12, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.6, patience=2, verbose=1),
        # callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
        callbacks.TensorBoard(log_dir='TBlog/', histogram_freq=4,
                              write_graph=True, write_images=True)
    ]


def generate_bn_features(train_path, test_path):
    model = ResNet50(weights='imagenet', include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))
    batch_size = Batch_size
    n_steps_train = np.ceil(count_files(train_path) / batch_size)
    n_steps_test = np.ceil(count_files(test_path) / batch_size)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

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
    np.save('weights/bottleneck_features_train', bottleneck_features_train)
    np.save('weights/train_classes', train_generator.classes)

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
    np.save('weights/bottleneck_features_test', bottleneck_features_test)
    np.save('weights/test_classes', test_generator.classes)


def train_top_only(model, weights_path, train_path):
    model = ResNet50(weights='imagenet', include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))
    train_data = np.load('weights/bottleneck_features_train.npy')
    train_labels = one_hot_labels(np.load('weights/train_classes.npy'))
    test_data = np.load('weights/bottleneck_features_test.npy')
    test_labels = one_hot_labels(np.load('weights/test_classes.npy'))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', name='fcc_0'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(N_classes, activation='softmax', name='class_id'))
    print('Model bottom loaded.')

    top_model.compile(
        optimizer='SGD',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    top_model.fit(
        x=train_data,
        y=train_labels,
        epochs=50,
        batch_size=Batch_size,
        validation_data=(test_data, test_labels),
        callbacks=get_callbacks(),
        verbose=1)

    top_model.save_weights(weights_path)
    print('Model top trained.')


# def train_top_from_scratch(model, weights_path, train_path):
    


def fine_tune(model, weights_path, train_path, test_path):

    N_train_samples = count_files(train_path)
    N_test_samples = count_files(test_path)

    base_model = ResNet50(weights='imagenet', include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))
    print('Model bottom loaded.')
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', name='fcc_0'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(N_classes, activation='softmax', name='class_id'))
    top_model.load_weights(weights_path)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    for layer in model.layers[-N_layers_to_finetune:]:
        layer.trainable = False
    for layer in model.layers[:-N_layers_to_finetune]:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy',
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
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(N_train_samples/Batch_size),
        epochs=N_epochs,
        verbose=1,
        callbacks=get_callbacks(),
        validation_data=test_generator,
        validation_steps=np.ceil(N_test_samples/Batch_size),
        class_weight=None,
        max_q_size=10,
        workers=4,
        pickle_safe=False,
        initial_epoch=0)


    # (generator=train_generator,
    #                     steps_per_epoch=math.ceil(len(generator.index) / batch_size),
    #                     epochs=nb_epoch,
    #                     verbose=1,
    #                     callbacks=get_callbacks(),
    #                     validation_data=valid_generator,
    #                     validation_steps=math.ceil(len(generator.valid_index) / batch_size),
    #                     class_weight=class_weights,
    #                     workers=4,
    #                     pickle_safe=True)


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
    parser.add_argument(
        '--weights', default='weights/bottleneck_fc_model.h5', help='Path for top layer weights')

    # Some commented potential arguments
    # parser.add_argument('--img_path', default='misc/sample.jpg',  help='Path to input image')
    # parser.add_argument('--evaluate', default=False,  help='Flag to evaluate over full test set')
    # parser.add_argument('--img_list',  help='Path to the test image list')
    # parser.add_argument('--gt_labels', help='Path to the ground truth test labels')

    args = parser.parse_args()

    # valid = validate_arguments(args)
    # net, inp_im  = choose_net(args.network)

    train_path = 'data/train_224x224/' + args.group + '/train/'
    test_path = 'data/train_224x224/' + args.group + '/test/'
    if args.generate_bn_features:
        generate_bn_features(train_path, test_path)

    if args.train_top_only:
        if not os.path.exists('weights/bottleneck_features_train.npy'):
            print('Bottleneck features file not found! Generate first.')
        else:
            train_top_only(args.model, args.weights, train_path)

    elif args.finetune:
        if not os.path.exists(args.weights):
            print('Weights file not found! Train top first.')
        else:
            print('Fine tuning:')
            fine_tune(args.model, args.weights, train_path, test_path)
    else:
        print('No retraining selected.')

    # if args.evaluate:
    #     evaluate(net, args.img_list, inp_im, args.gt_labels, args.network)
    # else:
    #     predict(net, args.img_path, inp_im, args.network)


if __name__ == '__main__':
    main()
