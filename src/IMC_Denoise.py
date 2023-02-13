
import random
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tp
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from IMC_Denoise_package.IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise_package.IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
from IMC_Denoise_package.IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tifffile
import ImagePreprocessFilters as IPrep
import ImageParser as IP
import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# https://github.com/PENGLU-WashU/IMC_Denoise/blob/main/Jupyter_Notebook_examples/IMC_Denoise_Train_and_Predict.ipynb

path_IMC_DENOISE = '/home/martinha/PycharmProjects/phd/Preprocess_IMC/data_IMC_DENOISE/metabric22/'

def create_dir_as_requested(files):
    #     Data_structure example:
    # |---Raw_image_directory
    # |---|---Tissue1
    # |---|---|---Channel1_img.tiff
    # |---|---|---Channel2_img.tiff
    # ...
    # |---|---|---Channel_n_img.tiff
    # |---|---Tissue2
    # |---|---|---Channel1_img.tiff
    # |---|---|---Channel2_img.tiff
    # ...
    # |---|---|---Channel_n_img.tif

    for i in range(len(files)):
        image_path = files[i]
        image = IP.parse_image(image_path)
        print(image.shape)
        # create folder for the image
        name_image = image_path.rsplit('/',1)[-1][:-5]
        print(name_image)
        path_res = path_IMC_DENOISE + name_image
        if not os.path.exists(path_res):
            os.makedirs(path_res)

        for channel in range(image.shape[2]):
            img_ch = image[...,channel]
            img_name = os.path.join(path_res, 'Channel{}_img.tiff'.format(channel))
            tifffile.imwrite(img_name, img_ch,photometric="minisblack")



def apply_IMCDenoise_full(images_test, n_channels = 39,
                          n_neighbours = 4,
                          n_iter = 3,
                          window_size = 3):
    # n_neighbour and n_lambda are the parameters from DIMR algorithm for hot pixel removal
    # in the training set generation process. 4 and 5 are their defaults.
    # If the defaults are changed, the corresponding parameter should be declared in
    # DeepSNiF_DataGenerator(). Otherwise, they can be omitted.
    #
    # The DeepSNiF_DataGenerator class search all the CD38 images in raw image directory,
    # split them into multiple 64x64 patches, and then augment the generated data.
    # Note the very sparse patches are removed in this process.
    Raw_directory = path_IMC_DENOISE
    images_denoise = [None] * len(images_test)

    for channel in range(5): #n_channels
        channel_name = 'Channel{}_img'.format(channel)
        # n_neighbours = 4 # Larger n enables removing more consecutive hot pixels.
        # n_iter = 3 # Iteration number for DIMR
        # window_size = 3 # Slide window size. For IMC images, window_size = 3 is fine.

        DataGenerator = DeepSNiF_DataGenerator(channel_name = channel_name, n_neighbours = n_neighbours,
                                               n_iter = n_iter, window_size = window_size, ratio_thresh = 0.92) # they say that the default is 0.5 but it is 0.8
        # i think threshold is important because if the sparsity its below that threshold the patch is omitted
        # this leads to some channels not having patches
        generated_patches = DataGenerator.generate_patches_from_directory(load_directory = Raw_directory)
        print('The shape of the generated training set is ' + str(generated_patches.shape) + '.')

        # Define parameters for DeepSNiF training
        train_epoches = 200 # training epoches, which should be about 200 for a good training result. The default is 200.
        train_initial_lr = 1e-3 # inital learning rate. The default is 1e-3.
        train_batch_size = 128 # training batch size. For a GPU with smaller memory, it can be tuned smaller. The default is 256.
        pixel_mask_percent = 0.2 # percentage of the masked pixels in each patch. The default is 0.2.
        val_set_percent = 0.15 # percentage of validation set. The default is 0.15.
        loss_function = "I_divergence" # loss function used. The default is "I_divergence".
        weights_name = None # trained network weights saved here. If None, the weights will not be saved.
        loss_name = None # training and validation losses saved here, either .mat or .npz format. If not defined, the losses will not be saved.
        weights_save_directory = None # location where 'weights_name' and 'loss_name' saved.
        # If the value is None, the files will be saved in a sub-directory named "trained_weights" of  the current file folder.
        is_load_weights = False # Use the trained model directly. Will not read from saved one.
        lambda_HF = 3e-6 # HF regularization parameter
        deepsnif = DeepSNiF(train_epoches = train_epoches,
                            train_learning_rate = train_initial_lr,
                            train_batch_size = train_batch_size,
                            mask_perc_pix = pixel_mask_percent,
                            val_perc = val_set_percent,
                            loss_func = loss_function,
                            weights_name = weights_name,
                            loss_name = loss_name,
                            weights_dir = weights_save_directory,
                            is_load_weights = is_load_weights,
                            lambda_HF = lambda_HF)

        train_loss, val_loss = deepsnif.train(generated_patches)

        # # plot training and validation
        # plt.figure(figsize=(8, 5))
        # plt.plot(np.array(range(len(train_loss))),train_loss, color='red', marker='^', linewidth=2, markersize=8)
        # plt.plot(np.array(range(len(val_loss))),val_loss, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=8)
        # plt.xlabel('Epoches')
        # plt.ylabel('BCE losses')
        # plt.legend(['training loss', 'val loss'])
        # plt.show()


        for i in range(len(images_test)):
            print('channel', channel)
            img = images_test[i]
            img_ch = img[...,channel]
            reconstructed_image = deepsnif.perform_IMC_Denoise(img_ch, n_neighbours = n_neighbours,
                                                               n_iter = n_iter, window_size = window_size)
            if channel == 0:
                images_denoise[i] = reconstructed_image
            else:
                images_denoise[i] = np.dstack((images_denoise[i],reconstructed_image))

    return images_denoise




# Perform the DIMR algorithm only if the SNR of the raw image is high.
def apply_dimr_1img_1ch(img):
    n_neighbours = 4 # Larger n enables removing more consecutive hot pixels.
    n_iter = 3 # Iteration number for DIMR
    window_size = 3
    Img_DIMR = DIMR(n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size).perform_DIMR(img)
    return Img_DIMR


def apply_DIMR(image_full,
               n_neighbours = 4,
               n_iter = 3,
               window_size = 5):
    # not the scheme they got. just one image with all markers

    # for each marker in image
    img_dimr = np.empty(image_full.shape)
    for ch in range(image_full.shape[2]):
        img_ch = image_full[:,:,ch]
        dmr = DIMR(n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size)
        img_dimr_ch =dmr.perform_DIMR(img_ch)
        img_dimr[:,:,ch] = img_dimr_ch
    print(img_dimr)
    return img_dimr



# perform DIMR and DeepSNiF algorithms for low SNR raw images.
def dimr_deepsnif_1img_1ch(img, deepsnif):
    n_neighbours = 4 # Larger n enables removing more consecutive hot pixels.
    n_iter = 3 # Iteration number for DIMR
    window_size = 3
    Img_DIMR_DeepSNiF = deepsnif.perform_IMC_Denoise(img, n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size)
    return Img_DIMR_DeepSNiF