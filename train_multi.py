import argparse
import clfs
import keras.layers
import multi_gpu_callbacks
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from generate_from_df import generator_from_df
from keras import optimizers, callbacks
from keras.utils.training_utils import multi_gpu_model
from sklearn.model_selection import train_test_split

num_age_groups = 4


def get_callbacks(gpus):
    """
    :return: A list of `keras.callbacks.Callback` instances to apply during training.

    """
    TBlog_path = 'TBlog/'
    weights_path = 'weights/'
    if gpus > 1:
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
                filepath=weights_path + '{epoch:02d}-{val_triage_acc:.2f}.hdf5',
                monitor='val_triage_acc',
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
            callbacks.TensorBoard(
                log_dir=TBlog_path,
                histogram_freq=0,
                write_graph=True,
                write_images=True)
        ]


def prep_dir():
    TBlog_path = 'TBlog/'
    weights_path = 'weights/'
    os.makedirs(TBlog_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)


def get_train_datagen(train_df, datapath, batch_size, target_size=224):
    datagen = generator_from_df(df=train_df,
                                batch_size=batch_size,
                                target_size=target_size,
                                datapath=datapath,
                                train=True)
    return datagen


def get_test_datagen(test_df, datapath, batch_size, target_size=224):
    datagen = generator_from_df(df=test_df,
                                batch_size=batch_size,
                                target_size=target_size,
                                datapath=datapath,
                                train=False)
    return datagen


def get_base_model():
    # base_model = densenet121_model(
    #     img_rows=224,
    #     img_cols=224,
    #     color_type=1,
    #     num_classes=2)
    base_model = clfs.get_model(model='densenet121', top='linear')

    return base_model


def get_model(weights_path):
    base_model = get_base_model()
    base_model.load_weights(weights_path)
    base_model.layers.pop()
    x = base_model.layers[-1].output
    gender_prediction = keras.layers.Dense(1, activation='sigmoid', name='gender')(x)
    agegroup_prediction = keras.layers.Dense(num_age_groups, activation='softmax', name='age')(x)
    triage_prediction = keras.layers.Dense(1, activation='sigmoid', name='triage')(x)
    full_model = keras.Model(base_model.input, [triage_prediction, agegroup_prediction, gender_prediction])

    return full_model


def train(batch_size, n_epochs, gpus, df, datapath):
    print('Loading model...')
    print("[INFO] training with {} GPUs...".format(gpus))

    X_train, X_test, y_train, y_test = train_test_split(df['filepath'], df.drop('filepath', axis=1), test_size=0.1)
    df_test = y_test.join(X_test)
    df_train = y_train.join(X_train)

    n_train_samples = df_train.shape[0]
    n_test_samples = df_test.shape[0]

    target_size = 224

    train_datagen = get_train_datagen(df_train, datapath, batch_size=batch_size * gpus, target_size=target_size)
    test_datagen = get_test_datagen(df_test, datapath, batch_size=batch_size * gpus, target_size=target_size)

    pretrained_weights_path = 'weights/PA.hdf5'

    # train the model on the new data for a few epochs
    print('Loading model...')

    class_weight = None

    if gpus > 1:
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            full_model = get_model(pretrained_weights_path)
        # make the model parallel
        gpu_full_model = multi_gpu_model(full_model, gpus=gpus)
        gpu_full_model.compile(
            # optimizer=optimizers.SGD(lr=1e-4, momentum=0.5),
            optimizer=optimizers.Adam(lr=1e-2),
            # optimizer=optimizers.rmsprop(),
            loss=['binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'],
            metrics=['accuracy', 'accuracy', 'accuracy'])

        gpu_full_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=int(np.ceil(n_train_samples / (batch_size * gpus))),
            epochs=n_epochs,
            verbose=1,
            callbacks=get_callbacks(gpus),
            validation_data=test_generator,
            validation_steps=int(np.ceil(n_test_samples / (batch_size * gpus))),
            class_weight=class_weight,
            max_queue_size=10,
            workers=4,
            use_multiprocessing=False,
            initial_epoch=0)
    else:
        full_model = get_model(pretrained_weights_path)
        print('Model loaded')
        full_model.compile(
            # optimizer=optimizers.SGD(lr=1e-4, momentum=0.5),
            optimizer=optimizers.Adam(lr=1e-2),
            # optimizer=optimizers.rmsprop(),
            loss=['binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'],
            metrics=['accuracy'])
        print('Model compiled')
        full_model.fit_generator(
            generator=train_datagen,
            steps_per_epoch=int(np.ceil(n_train_samples / (batch_size * gpus))),
            epochs=n_epochs,
            verbose=1,
            callbacks=get_callbacks(gpus),
            validation_data=test_datagen,
            validation_steps=int(np.ceil(n_test_samples / (batch_size * gpus))),
            class_weight=class_weight,
            max_queue_size=10,
            workers=4,
            use_multiprocessing=False,
            initial_epoch=0)

    weights_path = 'weights/multi_trained.h5'

    full_model.save_weights(weights_path)
    print('Model trained.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, help='Image size')
    parser.add_argument('--epochs', default=100, help='# of epochs for top training')
    parser.add_argument('--gpus', type=int, default=1, help='# of GPUs to use for training')
    parser.add_argument('--df', type=str, default='multitask.csv', help='Path to dataframe with filepaths and labels')
    parser.add_argument('--datapath', type=str, required=True, help='Path to image data')

    args = parser.parse_args()
    prep_dir()

    batch_size = int(args.batch_size)
    n_epochs = int(args.epochs)
    gpus = int(args.gpus)
    df = pd.read_csv(args.df)
    datapath = args.datapath

    train(batch_size=batch_size,
          n_epochs=n_epochs,
          gpus=gpus,
          df=df,
          datapath=datapath)


if __name__ == '__main__':
    main()
