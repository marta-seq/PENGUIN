import copy
import os
import random
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from UNET import UNET

import ImageParser as IP
import os
# import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from tqdm import tqdm

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from tf.keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from DeepL import score_testset_classification, plot_summary_loss, plot_summary_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tifffile
from patchify import unpatchify
import cv2


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def patch(filesTrain, PatchSize:int = 128, step:int = 128):
    all_img_patches = []
    for i in tqdm(range(len(filesTrain))):
            full_image = filesTrain[i]
            ch = full_image.shape[2]
            patches_img = patchify(full_image, (PatchSize, PatchSize,ch),
                                   step=step)  # Step=256 for 256 patches means no overlap
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :, :]
                    all_img_patches.append(single_patch_img)

    # This will split the image into small images of shape [3,3]
    images = np.array(all_img_patches)
    images = np.expand_dims(images, -1)
    print(images.shape)
    return images



def autoencoderV1(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate
    import tensorflow as tf
    tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))


        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        # model.add(Conv2D(39, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(IMG_CHANNELS, (3, 3), activation='relu', padding='same'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.summary()
        return model

# https://keras.io/examples/vision/oxford_pets_image_segmentation/
def unet_xception(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    from tensorflow.keras import layers


    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(IMG_CHANNELS, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    return model

# # https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py

def train_predict_autoencoder(images_train,images_test, dir = 'breast_cancer_imc/deep_results/',
                              batch_size = 8, EPOCHS = 5,
                              PatchSize = 16, step = 16):

    X_train, X_val, y_train, y_val = train_test_split(images_train, images_train, test_size=0.20, random_state=SEED)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=SEED)

    X_train = patch(X_train,PatchSize, step)
    X_train = np.squeeze(X_train)
    X_train = X_train.astype(np.float16, copy=False)
    print(X_train.shape)

    X_val = patch(X_val, PatchSize, step)
    X_val = np.squeeze(X_val)
    X_val = X_val.astype(np.float16, copy=False)

    print(X_train.shape)

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    PATIENCE_ES = 50
    PATIENCE_LR = 30

    es = EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE_ES, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_LR, min_lr=0.00001, verbose=1)
    # save the best performing model using validation result
    checkpointer = ModelCheckpoint(filepath=dir + 'CheckPoint/saved_weights.hdf5',
                                   monitor='val_loss', verbose=0,
                                   save_best_only=True)

    callbacks = [es, reduce_lr, checkpointer]


    # model = UNET(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = autoencoderV1(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


    history = model.fit(X_train, X_train,
                        validation_data= (X_val, X_val),
                        batch_size=batch_size, epochs=EPOCHS,
                        callbacks=callbacks) #, use_multiprocessing=True, workers= 6

    del X_train, y_train
    del X_val, y_val

    ##########
    # #Load model and predict
    # model = get_model()
    # #model.load_weights('mitochondria_gpu_tf1.4.hdf5')
    # model.load_weights('mitochondria_50_plus_100_epochs.hdf5')
    images_denoise = []
    ch = images_test[0].shape[2]
    for i in tqdm(range(len(images_test))):
        predict_img = images_test[i]
        # patches = patchify(predict_img, (PatchSize, PatchSize,ch),
        #                    step=step)
        # predicted_patches = []
        # for i in range(patches.shape[0]):
        #     for j in range(patches.shape[1]):
        #         print(i,j)
        #         single_patch = patches[i,j,:,:,:]
        #         print(single_patch.shape)
        #         # single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        #         # single_patch_norm = np.expand_dims(single_patch, axis=1)
        #         # print(single_patch_norm.shape)
        #         # single_patch_input=np.expand_dims(single_patch_norm, 0)
        #         # Predict and threshold for values above 0.5 probability
        #         # single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
        #         # print(single_patch_input.shape)
        #         single_patch_prediction = (model.predict(single_patch))
        #         print(single_patch_prediction.shape)
        #         predicted_patches.append(single_patch_prediction)
        #
        # predicted_patches = np.array(predicted_patches)
        # # predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], PatchSize,PatchSize,ch) )
        # predicted_patches_reshaped = np.reshape(predicted_patches, patches.shape)
        #
        # print('predi img', predict_img.shape)
        # print('patches', patches.shape)
        # print('pred patches', predicted_patches.shape)
        # print('pred patches reshaped', predicted_patches_reshaped.shape)
        # reconstructed_image = unpatchify(predicted_patches_reshaped, predict_img.shape)


        #
        # print('predicted_patches', predicted_patches.shape)
        # print('predict_img', predict_img.shape)
        # predicted_patches_reshaped = np.reshape(predicted_patches, patches.shape) # patches.shape[1],
        # print('predicted_patches_reshaped', predicted_patches_reshaped.shape)
        # print('patches original', patches.shape)
        # reconstructed_image = unpatchify(predicted_patches_reshaped, (PatchSize, PatchSize,ch)) # predict_img.shape
        # # reconstructed_image = prediction_full_image(model, predict_img,PatchSize)


        #This will split the image into small images of shape [3,3]
        patches = patchify(predict_img, (PatchSize, PatchSize,ch), step=PatchSize)  #Step=256 for 256 patches means no overlap

        predicted_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                print(i,j)

                single_patch = patches[i,j,:,:]
                # single_patch_norm = np.expand_dims(np.array(single_patch),2)
                # single_patch_input=np.expand_dims(single_patch, 0)

                #Predict and threshold for values above 0.5 probability
                # single_patch_prediction = (model.predict(single_patch)[0,:,:,0] > 0.5).astype(np.uint8)

                single_patch_prediction = (model.predict(single_patch)[0]> 0.5).astype(np.uint8)
                print('single_patch_prediction',single_patch_prediction.shape) # (1, 128, 128, 39)

                predicted_patches.append(single_patch_prediction)

        predicted_patches = np.array(predicted_patches)
        print('predict_patches', predicted_patches.shape)
        predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], PatchSize,PatchSize, ch))
        predicted_patches_reshaped = np.expand_dims(np.array(predicted_patches_reshaped),2)
        print('predict_patches reshaped', predicted_patches_reshaped.shape) # (4, 4, 1, 128, 128, 39)
        # (n_rows, n_cols, 1, patch_height, patch_width, N)
        # merging back patches
        # https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
        image_height, image_width, channel_count = predict_img.shape
        patch_height, patch_width, step = PatchSize, PatchSize, step
        patch_shape = (patch_height, patch_width, channel_count)
        output_height = image_height - (image_height - patch_height) % step
        output_width = image_width - (image_width - patch_width) % step
        output_shape = (output_height, output_width, channel_count)
        reconstructed_image = unpatchify(predicted_patches_reshaped, output_shape)
        images_denoise.append(reconstructed_image)

    # tf.keras.clear_session()
    tf.keras.backend.clear_session()
    print('finish train')
    return images_denoise




def train_predict_autoencoder_per_channel(images_train,images_test, dir = 'breast_cancer_imc/deep_results/',
                                          batch_size = 8, EPOCHS = 5, PatchSize = 32, step = 32):


    X_train, X_val, y_train, y_val = train_test_split(images_train, images_train, test_size=0.20, random_state=SEED)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=SEED)

    X_train = patch(X_train,PatchSize, step)
    X_train = np.squeeze(X_train)
    X_train = X_train.astype(np.float16, copy=False)

    X_val = patch(X_val, PatchSize, step)
    X_val = np.squeeze(X_val)
    X_val = X_val.astype(np.float16, copy=False)

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    PATIENCE_ES = 50
    PATIENCE_LR = 30


    es = EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE_ES, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_LR, min_lr=0.00001, verbose=1)
    # save the best performing model using validation result
    checkpointer = ModelCheckpoint(filepath=dir + 'CheckPoint/saved_weights.hdf5',
                                   monitor='val_loss', verbose=0,
                                   save_best_only=True)

    callbacks = [es, reduce_lr, checkpointer]
    images_denoise = [None] * len(images_test)

    for channel in range(IMG_CHANNELS):
        X_train_ch = X_train[...,channel, np.newaxis]
        X_val_ch = X_val[...,channel, np.newaxis]

        model = unet_xception(IMG_HEIGHT, IMG_WIDTH, 1)
        # model = autoencoderV1(IMG_HEIGHT, IMG_WIDTH, 1)


        history = model.fit(X_train_ch, X_train_ch,
                            validation_data= (X_val_ch, X_val_ch),
                            batch_size=batch_size, epochs=EPOCHS,
                            callbacks=callbacks) #, use_multiprocessing=True, workers= 6

        del X_train_ch
        del X_val_ch
        ##########
        # #Load model and predict
        # model = get_model()
        # #model.load_weights('mitochondria_gpu_tf1.4.hdf5')
        # model.load_weights('mitochondria_50_plus_100_epochs.hdf5')

        for i_test in tqdm(range(len(images_test))):
            predict_img = images_test[i_test][...,channel, np.newaxis]
            print('image testttt', predict_img.shape)
            patches = patchify(predict_img, (PatchSize, PatchSize,1), step=PatchSize)  #Step=256 for 256 patches means no overlap
            predicted_patches = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    print(i,j)

                    single_patch = patches[i,j,:,:]
                    # single_patch_norm = np.expand_dims(np.array(single_patch),2)
                    # single_patch_input=np.expand_dims(single_patch, 0)

                    #Predict and threshold for values above 0.5 probability
                    # single_patch_prediction = (model.predict(single_patch)[0,:,:,0] > 0.5).astype(np.uint8)

                    single_patch_prediction = (model.predict(single_patch)[0]> 0.5).astype(np.uint8)
                    print('single_patch_prediction',single_patch_prediction.shape) # (1, 128, 128, 39)
                    print('image testttt', predict_img.shape)

                    predicted_patches.append(single_patch_prediction)

            predicted_patches = np.array(predicted_patches)
            print('predict_patches', predicted_patches.shape)
            predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], PatchSize,PatchSize))
            predicted_patches_reshaped = np.expand_dims(np.array(predicted_patches_reshaped),2)
            predicted_patches_reshaped = predicted_patches_reshaped[...,np.newaxis]
            print('predict_patches reshaped', predicted_patches_reshaped.shape) # (4, 4, 1, 128, 128, 39)
            # (n_rows, n_cols, 1, patch_height, patch_width, N)
            # merging back patches
            # https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
            image_height, image_width, channel_count = predict_img.shape
            patch_height, patch_width, step = PatchSize, PatchSize, step
            patch_shape = (patch_height, patch_width, channel_count)
            output_height = image_height - (image_height - patch_height) % step
            output_width = image_width - (image_width - patch_width) % step
            output_shape = (output_height, output_width, channel_count)
            reconstructed_image = unpatchify(predicted_patches_reshaped, output_shape)
            print(reconstructed_image.shape)
            if channel == 0:
                images_denoise[i_test] = reconstructed_image
            else:
                images_denoise[i_test] = np.dstack((images_denoise[i_test],reconstructed_image))
        # tf.keras.clear_session()
        tf.keras.backend.clear_session()
        print('finish train')
    print(images_denoise[0].shape)
    return images_denoise
