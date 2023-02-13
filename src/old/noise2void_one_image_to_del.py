# https://github.com/juglab/n2v/blob/main/examples/2D/denoising2D_SEM/01_training.ipynb
# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile
import tifffile

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import ImageParser as IP
img_path = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/' \
           'PreprocessAnalysis/metabric22/MB3121_663_FullStack.tiff'

img_arr = IP.parse_image(img_path) # (679, 665, 39)
clean_arr = np.empty(img_arr.shape)
# one channel at time
for channel in range(img_arr.shape[2]):
    print(channel)
    img = img_arr[:,:,channel]
    # the data generator from them creates two adicional dimensions (1, 2500, 1690, 1)
    img = img[np.newaxis,..., np.newaxis]

    # We create our DataGenerator-object.
    # It will help us load data and extract patches for training and validation.
    datagen = N2V_DataGenerator()

    # We load all the '.tif' files from the 'data' directory.
    # If you want to load other types of files see the RGB example.
    # The function will return a list of images (numpy arrays).
    # imgs = datagen.load_imgs_from_directory(directory = "data/")

    # Let's look at the shape of the images.
    # print(imgs[0].shape,imgs[1].shape) [(1, 2500, 1690, 1),...]
    # The function automatically added two extra dimensions to the images:
    # One at the beginning, is used to hold a potential stack of images such as a movie.
    # One at the end, represents channels.

    #generatepatches different images to train and validation
    # We will use the first image to extract training patches and store them in 'X'
    patch_shape = (64,64)
    X = datagen.generate_patches_from_list([img], shape=patch_shape)

    # # We will use the second image to extract validation patches.
    # X_val = datagen.generate_patches_from_list(imgs[1:], shape=patch_shape)

    # Patches are created so they do not overlap.
    # (Note: this is not the case if you specify a number of patches. See the docstring for details!)
    # Non-overlapping patches would also allow us to split them into a training and validation set
    # per image. This might be an interesting alternative to the split we performed above.

    # Train

    # train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
    # is shown once per epoch.
    config = N2VConfig(X, unet_kern_size=3,
                       train_steps_per_epoch=int(X.shape[0]/128), train_epochs=20, train_loss='mse', batch_norm=True,
                       train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64),
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)

    # Let's look at the parameters stored in the config-object.
    vars(config)
    # When creating the config-object, we provide the training data X.
    # From X we extract mean and std that will be used to normalize all data before
    # it is processed by the network. We also extract the dimensionality and number of channels
    # from X.
    #
    # Compared to supervised training (i.e. traditional CARE), we recommend to use N2V
    # with an increased train_batch_size and batch_norm. To keep the network from learning
    # the identity we have to manipulate the input pixels during training.
    # For this we have the parameter n2v_manipulator with default value 'uniform_withCP'.
    # Most pixel manipulators will compute the replacement value based on a neighborhood.
    # With n2v_neighborhood_radius we can control its size.


    # a name used to identify the model
    model_name = 'n2v_2D'
    # the base directory in which our model will live
    basedir = 'models'
    # We are now creating our network model.
    model = N2V(config, model_name, basedir=basedir)


    # We are ready to start training now. # todo put X_val
    history = model.train(X, X)

    # model.export_TF(name='Noise2Void - 2D SEM Example',
    #                 description='This is the 2D Noise2Void example trained on SEM data in python.',
    #                 authors=["Tim-Oliver Buchholz", "Alexander Krull", "Florian Jug"],
    #                 test_img=X[0,...,0], axes='YX',
    #                 patch_shape=patch_shape)
    #
    # # A previously trained model is loaded by creating a new N2V-object without providing a 'config'.
    # model_name = 'n2v_2D'
    # basedir = 'models'
    # model = N2V(config=None, name=model_name, basedir=basedir)

    # In case you do not want to load the weights that lead to lowest validation loss during
    # training but the latest computed weights, you can execute the following line:

    # model.load_weights('weights_last.h5')

    # Here we process the data.
    # The parameter 'n_tiles' can be used if images are to big for the GPU memory.
    # If we do not provide the n_tiles' parameter the system will automatically try to find an appropriate tiling.
    # This can take longer.
    # img = img[np.newaxis, ..., np.newaxis]
    img = img_arr[:,:,channel]
    print(img.shape)

    pred_train = model.predict(img, axes='YX')
    print(pred_train.shape)
    clean_arr[:,:,channel] = pred_train

clean_arr = np.float32(clean_arr)
print(clean_arr.shape)
denoise_img2_t = np.moveaxis(clean_arr, -1, 0)
tifffile.imwrite('noise2void_1image.tiff', denoise_img2_t,
                 photometric="minisblack")


