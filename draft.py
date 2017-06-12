import os
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense
import argparse
import tensorflow as tf
import numpy as np


N_classes = 5

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


def generate_bn_features(train_path, test_path):
    model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
    batch_size = 16
    datagen = ImageDataGenerator(
     rescale=1./255,
     )
    train_generator = datagen.flow_from_directory(
            train_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True)
    bottleneck_features_train = model.predict_generator(train_generator, 1)
    # save the output as a Numpy array
    np.save('weights/bottleneck_features_train', bottleneck_features_train)
    np.save('weights/train_classes', train_generator.classes)

    test_generator = datagen.flow_from_directory(
            test_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='sparse',  # this means our generator will only yield batches of data, no labels
            shuffle=True)
    bottleneck_features_validation = model.predict_generator(test_generator, 1)
    np.save('weights/bottleneck_features_validation', bottleneck_features_validation)
    np.save('weights/test_classes', test_generator.classes)

def train_top_only(model, weights_path, train_path):
    model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
    train_data = np.load('weights/bottleneck_features_train.npy')
    # the features were saved in order, so recreating the labels is easy
    train_labels = np.array([0] * 1000 + [1] * 1000) #Make sure this matches
    validation_data = np.load('weights/bottleneck_features_validation.npy')
    validation_labels = np.array([0] * 400 + [1] * 400)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', name='fcc_0'))
    top_model.add(Dropout(0.5))
    # top_model.add(Dense(1, activation='sigmoid'))
    top_model.add(Dense(N_classes, activation='softmax', name='class_id'))

    print('Model bottom loaded.')

    model.compile(optimizer='SGD',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(train_data, train_labels,
        epochs=50,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels))

    model.save_weights('weights/bottleneck_fc_model.h5')
    print('Model top trained.')


    # train_datagen = ImageDataGenerator(
    #     zoom_range=0.1,
    #     horizontal_flip=True)
    #
    # train_generator = train_datagen.flow_from_directory(
    #     train_path,
    #     target_size=(224, 224))
    #
    # model.fit_generator(
    #     train_generator,
    #     epochs=50,
    #     batch_size=batch_size,
    #     validation_data=(validation_data, validation_labels))

    return(model)


def fine_tune(model, weights_path, train_path):
    model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
    print('Model bottom loaded.')
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    top_model.load_weights(weights_path)
    model.add(top_model)

    #TODO: Decide how many layers to freeze/unfreeze and retrain
    # return(model)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', help='The network eg. resnet50')
    parser.add_argument('--group', default='F_Adult', help='The network eg. resnet50')
    # parser.add_argument('--train_path', default='train', help='Path to training set')
    # parser.add_argument('--test_path', default='test', help='Path to test set')
    parser.add_argument('--generate_bn_features', action='store_true',  help='Flag to generate bottleneck features')
    parser.add_argument('--train_top_only', action='store_true',  help='Flag to retrain')
    parser.add_argument('--finetune', action='store_true', help='Flag to fine tune')
    parser.add_argument('--weights', default='weights/bottleneck_fc_model.h5', help='Path for top layer weights')

    # parser.add_argument('--img_path', default='misc/sample.jpg',  help='Path to input image')
    # parser.add_argument('--evaluate', default=False,  help='Flag to evaluate over full validation set')
    # parser.add_argument('--img_list',  help='Path to the validation image list')
    # parser.add_argument('--gt_labels', help='Path to the ground truth validation labels')

    args = parser.parse_args()
    # valid = validate_arguments(args)
    # net, inp_im  = choose_net(args.network)

    train_path = 'data/'+args.group+'/train/'
    test_path = 'data/'+args.group+'/test/'
    if args.generate_bn_features:
        generate_bn_features(train_path, test_path)

    if args.train_top_only:
        #TODO: check if weights file exists
        train_top_only(args.model, args.weights, train_path)

    elif args.finetune:
        #TODO: check if weights file exists
        print('Fine tuning:')
        fine_tune(args.model, args.weights, train_path)
    else:
        print('No retraining selected.') #TODO: include in validation step

    # if args.evaluate:
    #     evaluate(net, args.img_list, inp_im, args.gt_labels, args.network)
    # else:
    #     predict(net, args.img_path, inp_im, args.network)

if __name__ == '__main__':
    main()
