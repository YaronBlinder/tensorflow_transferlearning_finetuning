from cv2 import resize, imread
import numpy as np
import scipy.ndimage as ndi

def random_crop(x, random_crop_ratio, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[0], x.shape[1]
    random_crop_size = [int(np.round(w * random_crop_ratio)), int(np.round(h * random_crop_ratio))]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped = x[offsetw:offsetw + random_crop_size[0], offseth:offseth + random_crop_size[1], :]
    return cropped


def scale_im(x, size, *args, **kwargs):
    resized = resize(x, (size, size))
    return resized


def radical_preprocess(x, position='PA', *args, **kwargs):
    if position == 'PA':
        # ds_mean = 38679.2766871 #calculated
        # ds_std = 26824.8858495
        # ds_mean = 134.39976334 #cxr8
        # ds_std = 114.044506656 #cxr8
        ds_mean = 34996.7451362  # cxr8 + big_batch_all
        ds_std = 29177.7923993  # cxr8 + big_batch_all
    elif position == 'LAT':
        ds_mean = 34024.5927414
        ds_std = 33591.1099547


    else:
        print('You should not be here')

    x = x.astype('float32')
    x -= ds_mean
    x /= ds_std

    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def random_90deg_rotation(x, row_index=0, col_index=1, channel_index=0, fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * 90 * np.random.randint(0, 4)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    if len(x.shape) > 2:  # non-grayscale on TF
        x = np.rollaxis(x, channel_index, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [
            ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=0, mode=fill_mode,
                                               cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index + 1)
    else:
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        x = ndi.interpolation.affine_transform(x, final_affine_matrix, final_offset, order=0, mode=fill_mode, cval=cval)
        x = x.reshape((x.shape[0], x.shape[1], 1))
    return x


def preprocess_train(imfile, target_size=224, crop_ratio=0.9):
    im = imread(imfile, 0)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    elif im.shape[2] == 1:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    im = random_crop(im, crop_ratio)
    im = scale_im(im, target_size)
    im = radical_preprocess(im)
    im = random_90deg_rotation(im)
    print(imfile, im.shape)
    return(im)


def preprocess_test(imfile, target_size=224):
    im = imread(imfile, 0)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    elif im.shape[2] == 1:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    im = scale_im(im, target_size)
    im = radical_preprocess(im)
    return(im)


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
            callbacks.TensorBoard(
                log_dir=TBlog_path,
                histogram_freq=0,
                write_graph=True,
                write_images=True)
        ]


def generator_from_df(df, batch_size, target_size, datapath, train=False):
    """Generator that yields (X, Y).
    If features is not None, assume it is the path to a bcolz array
    that can be indexed by the same indexing of the input df.
    Assume input DataFrame df has columns 'imgpath' and 'target', where
    'imgpath' is full path to image file.
    https://github.com/fchollet/keras/issues/1627
    https://github.com/fchollet/keras/issues/1638
    Be forewarned if/when you modify this function: some errors will
    not be explicit, appearing only as a generic:
      ValueError: output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: None
    It usually means something in your infinite loop is not doing what
    you think it is, so the loop crashes and returns None.  Check your
    DataFrame in this function with various print statements to see if
    it is doing what you think it is doing.
    Again, error messages will not be too helpful here--if in doubt,
    print().
    """
    # Each epoch will only process an integral number of batch_size
    # but with the shuffling of df at the top of each epoch, we will
    # see all training samples eventually, but will skip an amount
    # less than batch_size during each epoch.
    nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)

    # At the start of *each* epoch, this next print statement will
    # appear once for *each* worker specified in the call to
    # model.fit_generator(...,workers=nworkers,...)!
    #     print("""
    # Initialize generator:
    #   batch_size = %d
    #   nbatches = %d
    #   df.shape = %s
    # """ % (batch_size, nbatches, str(df.shape)))

    count = 1
    epoch = 0

    df['filepath'] = df.filepath.apply(lambda filepath : datapath+filepath)
    # New epoch.
    while 1:

        # The advantage of the DataFrame holding the image file name
        # and the labels is that the entire df fits into memory and
        # can be easily shuffled at the start of each epoch.
        #
        # Shuffle each epoch using the tricky pandas .sample() way.
        df = df.sample(frac=1)  # frac=1 is same as shuffling df.

        epoch += 1
        i, j = 0, batch_size

        # Mini-batches within epoch.
        mini_batches_completed = 0
        for _ in range(nbatches):

            # Callbacks are more elegant but this print statement is
            # included to be explicit.
            # print("Top of generator for loop, epoch / count / i / j = "\
            #       "%d / %d / %d / %d" % (epoch, count, i, j))

            sub = df.iloc[i:j]

            try:

                # preprocess_input()
                # https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py#L389
                if train:
                    X = np.array([preprocess_train(f, target_size) for f in sub.filepath])
                else:
                    X = np.array([preprocess_test(f, target_size) for f in sub.filepath])

                Y_label2 = sub.label_2.values
                Y_agegroup = sub.ohe_P_YA_A_G.values
                Y_genderM = sub.gender_M.values

                mini_batches_completed += 1
                yield X, [Y_label2, Y_agegroup, Y_genderM]


            except IOError as err:

                # A type of lazy person's regularization: with
                # millions of images, if there are a few bad ones, no
                # need to find them, just skip their mini-batch if
                # they throw an error and move on to the next
                # mini-batch.  With the shuffling of the df at the top
                # of each epoch, the bad apples will be in a different
                # mini-batch next time around.  Yes, they will
                # probably crash that mini-batch, too, but so what?
                # This is easier than finding bad files each time.

                # Let's decrement count in anticipation of the
                # increment coming up--this one won't count, so to
                # speak.
                count -= 1

                # Actually, we could make this a try...except...else
                # with the count increment.  Homework assignment left
                # to the reader.

            i = j
            j += batch_size
            count += 1