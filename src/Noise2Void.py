import numpy as np
import random
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(0)
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

#### noise2void
from n2v.models import N2VConfig, N2V
from n2v.models import N2V

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Let's look at the parameters stored in the config-object.
# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

# https://github.com/bnsreenu/python_for_microscopists/blob/master/293_denoising_RGB_images_using_deep%20learning.ipynb
# https://github.com/juglab/n2v/blob/main/examples/2D/denoising2D_SEM/01_training.ipynb

# class N2V:
#     def __init__(self):
#         self.model = None
#

# per channel
def noise2void(img_list:list, patch_size:int = 64, train_epochs:int = 100,
               train_batch_size:int =16) -> list: # returns list because images may have different shapes
    clean_arr = copy.deepcopy(img_list)
    CH = img_list[0].shape[2]

    # one channel at time
    for channel in range(CH):
        print(channel)
        # get all
        images_ch = []
        for img in img_list:
            img_ch = img[:,:,channel]
            # the data generator from them creates two aditional dimensions (1, 2500, 1690, 1)
            img_ax = img_ch[np.newaxis,..., np.newaxis]
            images_ch.append(img_ax)

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
        n_files = int(len(img_list) * 0.8)
        print(n_files)
        print(len(images_ch))
        patch_shape = (patch_size,patch_size)
        X = datagen.generate_patches_from_list(images_ch[:n_files], shape=patch_shape)
        X = np.float32(X)
        X_val = datagen.generate_patches_from_list(images_ch[n_files:], shape=patch_shape)
        X_val = np.float32(X_val)
        # Patches are created so they do not overlap.
        # (Note: this is not the case if you specify a number of patches. See the docstring for details!)
        # Non-overlapping patches would also allow us to split them into a training and validation set
        # per image. This might be an interesting alternative to the split we performed above.

        # Train

        # train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
        # is shown once per epoch.
        config = N2VConfig(X, unet_kern_size=3,
                           train_steps_per_epoch=int(X.shape[0]/128),
                           train_epochs=train_epochs,
                           train_loss='mse', batch_norm=True,
                           train_batch_size=train_batch_size, n2v_perc_pix=0.198,
                           n2v_patch_shape=patch_shape,
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
        # tf.debugging.set_log_device_placement(True)
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        model = N2V(config, model_name, basedir=basedir)


        # We are ready to start training now. # todo put X_val
        history = model.train(X, X_val)

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
        # img = img_arr[:,:,channel]
        del X
        del X_val
        # print(len(images_ch))
        for i in range(len(images_ch)-1):
            # print(images_ch[i].shape)
            pred_train = model.predict(images_ch[i][0,...,0], axes='YX')
            # print(np.array(pred_train).shape)
            # print(i)
            # print(channel)
            clean_arr[i][...,channel] = np.array(pred_train)
    # print(clean_arr)
    return clean_arr


    # def predictN2V(self,img):
    #     clean_arr = copy.deepcopy(img)
    #     # print(len(images_ch))
    #     pred_train = self.model.predict(images_ch[i][0,...,0], axes='YX')
    #
    #         clean_arr[i][...,channel] = np.array(pred_train)
    #     print(clean_arr)
    #     return clean_arr
    #


# per channel one image at time
def noise2void_one_image(img:np.ndarray, patch_size:int = 64, train_batch:int = 32, train_epochs:int = 20) -> np.ndarray:
    n2v_img = np.empty(img.shape)
    print(img.shape)
    for ch in range(img.shape[2]):
        img_ch = img[np.newaxis,:,:,ch, np.newaxis] #, np.newaxis   img[np.newaxis,:,:,ch]
        print(img_ch.shape)
        datagen = N2V_DataGenerator()
        patch_size = patch_size
        # Patches are extracted from all images and combined into a single numpy array
        patch_shape = (patch_size,patch_size)
        patches = datagen.generate_patches_from_list([img_ch], shape=patch_shape)
        # Patches are created so they do not overlap.
        # (Note: this is not the case if you specify a number of patches. See the docstring for details!)
        # Non-overlapping patches enable us to split them into a training and validation set.

        train_val_split = int(patches.shape[0] * 0.8)
        X = patches[:train_val_split]
        X_val = patches[train_val_split:]

        # train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
        # is shown once per epoch.
        train_batch = train_batch
        config = N2VConfig(X, unet_kern_size=3,unet_n_first=64, unet_n_depth=3,
                           train_steps_per_epoch=int(X.shape[0]/train_batch), train_epochs=train_epochs, train_loss='mse',
                           batch_norm=True, train_batch_size=train_batch,
                           n2v_perc_pix=0.198, n2v_patch_shape=(patch_size, patch_size),
                           n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, single_net_per_channel=False)

        # Let's look at the parameters stored in the config-object.
        vars(config)

        # a name used to identify the model --> change this to something sensible!
        model_name = 'n2v_2D_stars'
        # the base directory in which our model will live
        basedir = 'N2V/models'
        # We are now creating our network model.
        model = N2V(config, model_name, basedir=basedir)

        # We are ready to start training now.
        history = model.train(X, X_val)
        img_ch = img_ch[0]
        print(img_ch.shape)
        pred = model.predict(img_ch[...,0], axes='YX')
        n2v_img[:,:,ch] = pred
    return n2v_img

