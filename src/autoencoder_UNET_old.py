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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

######
# DATA FILES
######
# did not got the metabrc dataset based on metadata. just all theimages listed

path_stacks = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/' \
              'breast_data/imc_metabric2022/MBTMEIMCPublic/Images/'
res = os.listdir(path_stacks)
# select the fullstack
filesTrain = [x for x in res if x.__contains__('FullStack')] # 794
filesTrain = [str(path_stacks + x) for x in filesTrain ][:20]
# https://github.com/bnsreenu/python_for_microscopists/blob/master/219_unet_small_dataset_using_functional_blocks.py
# https://www.youtube.com/watch?v=GAYJ81M58y8&ab_channel=DigitalSreeni


import os
# import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(filesTrain, filesTrain, test_size=0.20, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=SEED)

filesTrain = X_train
filesTest = X_test
filesMasks = y_train
filesMasksTest = y_test

all_img_patches = []
all_mask_patches = []
PatchSize = 128
Step = 128
print(filesTrain)
def patch(filesTrain):
    for i in range(len(filesTrain)):
        file_exists = os.path.exists(filesTrain[i])
        if file_exists:  #
            large_image = IP.parse_image(filesTrain[i])
            # large_image = tiff.imread(filesTrain[i])
            # large_mask = tiff.imread(filesMasks[i])
            print(large_image.shape)

            imgNorm = IP.normalize_by_channel(large_image)
            imgNorm = large_image

            ch = large_image.shape[2]
            patches_img = patchify(imgNorm, (PatchSize, PatchSize,ch),
                                   step=Step)  # Step=256 for 256 patches means no overlap
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :, :]
                    all_img_patches.append(single_patch_img)

    # This will split the image into small images of shape [3,3]
    images = np.array(all_img_patches)
    images = np.expand_dims(images, -1)
    print(images.shape)
    return images

X_train = patch(X_train)
# # Sanity check, view few mages
# import random
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# image_example  = X_train[image_number,:,:,:,27,:]
# plt.imshow(np.reshape(image_example, (PatchSize, PatchSize)), cmap='gray')
# plt.subplot(122)
# plt.imshow(np.reshape(y_train[image_number], (PatchSize, PatchSize,1)), cmap='gray')
# plt.show(block=True)

print(X_train.shape)
X_train = np.squeeze(X_train)
X_train = X_train.astype(np.float16, copy=False)

X_val = patch(X_val)
X_val = np.squeeze(X_val)
X_val = X_val.astype(np.float16, copy=False)

print(X_train.shape)
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

print(X_train.shape)
# input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# model = build_unet(input_shape)
# model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from DeepL import score_testset_classification, plot_summary_loss, plot_summary_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



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
        model.add(Conv2D(39, (3, 3), activation='relu', padding='same'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.summary()
        return model

PATIENCE_ES = 50
PATIENCE_LR = 30

dir = 'breast_cancer_imc/deep_results/'
es = EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE_ES, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_LR, min_lr=0.00001, verbose=1)
# save the best performing model using validation result
checkpointer = ModelCheckpoint(filepath=dir + 'CheckPoint/saved_weights.hdf5',
                               monitor='val_loss', verbose=0,
                               save_best_only=True)

callbacks = [es, reduce_lr, checkpointer]


# model = UNET(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = autoencoderV1(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
batch_size = 8
EPOCHS = 5

history = model.fit(X_train, X_train,
                    validation_data= (X_val, X_val),
                    batch_size=batch_size, epochs=EPOCHS,
                    callbacks=callbacks) #, use_multiprocessing=True, workers= 6

del X_train, y_train
del X_val, y_val

# # plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# # val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# # plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# # val_acc = history.history['val_accuracy']
#
# plt.plot(epochs, acc, 'y', label='Training acc')
# # plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# # IOU
# X_test = patch(filesTest)
# X_test = np.squeeze(X_test)
# X_test = X_test.astype(np.float32, copy=False)
# print('X_tes', X_test.shape)
# y_pred = model.predict(X_test)
# y_pred_thresholded = y_pred > 0.5
#
# intersection = np.logical_and(X_test, y_pred_thresholded)
# union = np.logical_or(X_test, y_pred_thresholded)
# iou_score = np.sum(intersection) / np.sum(union)
# print("IoU socre is: ", iou_score)

# test_img_number = random.randint(0, len(X_test))
# print(test_img_number)
#
# for test_img_number in range(len(X_test)):
#     print(test_img_number)
#     test_img = X_test[test_img_number]   # (128, 128, 31)
#     print(test_img.shape)
#     test_img = test_img[np.newaxis, ...]  # new dimension is axis 0
#
#     print(test_img.shape)
#
#     prediction = model.predict(test_img)
#
#     print(prediction.shape)
#
#     import tifffile
#
#     denoise_img2_t = np.moveaxis(test_img, -1, 0)
#     tifffile.imwrite('test_autoencoder_original{}.tiff'.format(test_img_number), denoise_img2_t,
#                      photometric="minisblack")
#     denoise_img2_t = np.moveaxis(prediction , -1, 0)
#     tifffile.imwrite('test_autoencoder_prediction{}.tiff'.format(test_img_number), denoise_img2_t,
#                      photometric="minisblack")
#
#

# https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py
import cv2
def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], patch_size):   #Steps of 256
        for j in range(0, image.shape[1], patch_size):  #Steps of 256
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_norm = np.expand_dims(np.array(single_patch), axis=1)
            # single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            # single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8) binary --- mask
            single_patch_prediction = model.predict(single_patch_input)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])

            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img

##########
# #Load model and predict
# model = get_model()
# #model.load_weights('mitochondria_gpu_tf1.4.hdf5')
# model.load_weights('mitochondria_50_plus_100_epochs.hdf5')
import tifffile
#
# for test_img_number in range(len(filesTest)):
#     print(test_img_number)
#     test_img = X_test[test_img_number]   # (128, 128, 31)
#     print(test_img.shape)
#     test_img = test_img[np.newaxis, ...]  # new dimension is axis 0
#     print(test_img.shape)
#     prediction = prediction(model, test_img, patch_size=PatchSize)
#     print(prediction.shape)
#     denoise_img2_t = np.moveaxis(test_img, -1, 0)
#     tifffile.imwrite('test_autoencoder_original{}.tiff'.format(test_img_number), denoise_img2_t,
#                      photometric="minisblack")
#     denoise_img2_t = np.moveaxis(prediction , -1, 0)
#     tifffile.imwrite('test_autoencoder_prediction{}.tiff'.format(test_img_number), denoise_img2_t,
#                      photometric="minisblack")

from patchify import unpatchify
for test_img_number in range(len(filesTest)):
    test_img = IP.parse_image(filesTest[test_img_number])
    print(test_img_number)
    ch = test_img.shape[2]

    imgNorm = IP.normalize_by_channel(test_img)
    imgNorm = test_img
    patches = patchify(imgNorm, (PatchSize, PatchSize,ch),
                           step=Step)
    print(patches.shape)
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            print(i,j)
            single_patch = patches[i,j,:,:,:]
            print(single_patch.shape)
            # single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            # single_patch_norm = np.expand_dims(single_patch, axis=1)
            # print(single_patch_norm.shape)
            # single_patch_input=np.expand_dims(single_patch_norm, 0)
            # Predict and threshold for values above 0.5 probability
            # single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            # print(single_patch_input.shape)
            single_patch_prediction = (model.predict(single_patch))
            print(single_patch_prediction.shape)
        predicted_patches.append(single_patch_prediction)

    predicted_patches = np.array(predicted_patches)
    print(predicted_patches.shape)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], PatchSize,PatchSize, ch) ) # patches.shape[1],
    reconstructed_image = unpatchify(predicted_patches_reshaped, test_img.shape)

    denoise_img2_t = np.moveaxis(test_img, -1, 0)
    tifffile.imwrite('test_autoencoder_original{}.tiff'.format(test_img_number), denoise_img2_t,
                     photometric="minisblack")
    denoise_img2_t = np.moveaxis(reconstructed_image, -1, 0)
    tifffile.imwrite('test_autoencoder_prediction{}.tiff'.format(test_img_number), denoise_img2_t,
                     photometric="minisblack")


print('finish train')
# tf.keras.clear_session()
tf.keras.backend.clear_session()
print('finish train')

