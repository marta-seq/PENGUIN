import numpy as np
import random
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(1)
tf.get_logger().setLevel('ERROR')
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tifffile
import ImagePreprocessFilters as IPrep
import ImageParser as IP
import tqdm



from IMC_Denoise_package.IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise_package.IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
from IMC_Denoise_package.IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

path_IMC_DENOISE = '/home/martinha/PycharmProjects/phd/Preprocess_IMC/data_IMC_DENOISE//'
#
# def create_dir_as_requested(files):
#     #     Data_structure example:
#     # |---Raw_image_directory
#     # |---|---Tissue1
#     # |---|---|---Channel1_img.tiff
#     # |---|---|---Channel2_img.tiff
#     # ...
#     # |---|---|---Channel_n_img.tiff
#     # |---|---Tissue2
#     # |---|---|---Channel1_img.tiff
#     # |---|---|---Channel2_img.tiff
#     # ...
#     # |---|---|---Channel_n_img.tif
#
#     for i in range(len(files)):
#         image_path = files[i]
#         image = IP.parse_image(image_path)
#         print(image.shape)
#         # create folder for the image
#         name_image = image_path.rsplit('/',1)[-1][:-5]
#         print(name_image)
#         path_res = path_IMC_DENOISE + name_image
#         if not os.path.exists(path_res):
#             os.makedirs(path_res)
#
#         for channel in range(image.shape[2]):
#             img_ch = image[...,channel]
#             img_name = os.path.join(path_res, 'Channel{}_img.tiff'.format(channel))
#             tifffile.imwrite(img_name, img_ch,photometric="minisblack")
#


