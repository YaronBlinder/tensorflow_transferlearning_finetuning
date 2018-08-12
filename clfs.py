import keras.layers
from densenet121 import densenet121_model
from keras.applications import ResNet50, VGG16, VGG19, Xception, InceptionV3
from keras.models import Model


N_classes = 2


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


def get_model(model, top, freeze_base=False, n_dense=512, dropout=True, pooling=None):
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


def radical_preprocess(x, position, *args, **kwargs):
    if position == 'PA':
        # ds_mean = 38679.2766871 #calculated
        # ds_std = 26824.8858495
        # ds_mean = 134.39976334 #cxr8
        # ds_std = 114.044506656 #cxr8
        ds_mean = 24705.6761615 #cxr8 + big_batch_all
        ds_std = 36862.05965 #cxr8 + big_batch_all
    elif position == 'LAT':
        ds_mean = 34024.5927414
        ds_std = 33591.1099547
    else:
        print('You should not be here')
    x = x.astype('float32')
    x -= ds_mean
    x /= ds_std
    return x


