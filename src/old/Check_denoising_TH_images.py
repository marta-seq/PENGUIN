"""
Check the differences from denoising IMC pictures

"""

import os
import random
import ImageParser as IP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

######
# 1. DATA FILES
######

data_dir = 'breast_cancer_imc/breast_data/'

path_data_meta_basel = os.path.join(data_dir, 'Basel_PatientMetadataALTERED.csv')
path_data_meta_zur = os.path.join(data_dir, 'Data_publication/ZurichTMA/Zuri_PatientMetadata.csv')
path_images = os.path.join(data_dir, 'OMEnMasks/ome/')

######
# 2. OPEN AND READ IMAGES
######

basel = pd.read_csv(path_data_meta_basel)
# basel = basel.loc[basel['diseasestatus']=='tumor']
names_images = basel['FileName_FullStack']
files_images = [str(path_images + sub) for sub in names_images]

zur = pd.read_csv(path_data_meta_zur)
names_imagesZur = zur['FileName_FullStack']
files_imagesZur = [str(path_images + sub) for sub in names_imagesZur]

# PARSE images to get numpy arrays into shape (, , 52)
images = map(IP.parse_image, files_images)
imagesZur = map(IP.parse_image, files_imagesZur)

#select good channels
channels_to_keep = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 32, 33, 34, 37, 38, 40, 41, 42,
                    43, 44, 45, 46, 47, 48]
channels_would_be_good = [25, 29, 31, 35, 36, 39]

imagesCH = map(lambda p: IP.simple_reduce_channels(p, channels_to_keep),images)
imagesCHZur = map(lambda p: IP.simple_reduce_channels(p, channels_to_keep),imagesZur)


######
# 3. TAKE OUTLIERS OUT . Outliers are removed through saturation of all pixels with
# values lower than the 1st and higher than the 99th percentile.
######

# change values to check differences

up_limit = 99  # 99
down_limit = 1 # 1
imgNoOutlier = map(lambda p:IP.remove_outliers(p,up_limit,down_limit), imagesCH)
imgNoOutlierZur = map(lambda p:IP.remove_outliers(p,up_limit,down_limit), imagesCHZur)

images = np.array(imgNoOutlier)
imagesZur = np.array(imgNoOutlierZur)

# put here the code for multiple images
# code for the originals


# see pictures
image_number = random.randint(0, len(images))
plt.figure(figsize=(12, 6))
plt.subplot(121)
image_example = imgNoOutlier[image_number,:,:,:,27,:]
plt.imshow(image_example, cmap='gray')
plt.show(block=True)





######
# NORMALIZE PER CHANNEL BASED ON MIN MAX PER CHANNEL OVER ALL DATASET
######

v_min, v_max = IP.get_maximum_minimum_per_channel_over_dataset(imgNoOutlier)
imgNorm = map(lambda p: IP.normalize_by_channel_based_on_dataset(p, v_min, v_max), imgNoOutlier)


v_minZur, v_maxZur = IP.get_maximum_minimum_per_channel_over_dataset(imgNoOutlier)
imgNormZur = map(lambda p: IP.normalize_by_channel_based_on_dataset(p, v_minZur, v_maxZur), imgNoOutlierZur)


# todo check other normalization techniques

######
# SIMPLE RESIZE PICTURES TO A STANDARD SIZE
######
# check better ways to resize??? and try patches
# RESIZE PIC with tensorflow resize_with_pad
INP_SIZE = (425, 425)
print('resizing images to {}'.format(INP_SIZE))
img_resize_train = map(lambda p: IP.resize_dataset(p, INP_SIZE), imgNormTrain)
img_resize_val = map(lambda p: IP.resize_dataset(p, INP_SIZE), imgNormVal)
img_resize_test = map(lambda p: IP.resize_dataset(p, INP_SIZE), imgNormTest)






# # logging.disable(logging.WARNING)
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# # from silence_tensorflow import silence_tensorflow
# # silence_tensorflow()
#
# # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')
#
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
#
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# # from tf.keras import backend as K