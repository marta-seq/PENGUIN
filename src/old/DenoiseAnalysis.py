"""
Check the differences from denoising IMC pictures

# todo save images with preprocess 99 and 1
# todo do it in ricky images and save them also
# save images with other thresholds
# get some info about pixel values histogram
# apply imc denoise

"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents [1]
sys.path.append(str(package_root_directory))

import copy
import os
import random
import utils.ImageParser as IP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'
# logging.disable(logging.WARNING)
import tensorflow as tf

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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

######
# 1. DATA FILES
######
channelsBodenmiller2020 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 32, 33, 34, 37, 38, 40, 41,
                                              42, 43, 44, 45, 46, 47, 48]
channels_would_be_good = [25, 29, 31, 35, 36, 39]
channelsBodenmiller2020_V2 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29,30,31, 32, 33, 34, 35, 36,37, 38, 39, 40, 41,
                              42, 43, 44, 45, 46, 47, 48] # join both lists

channels_METABRIC2020 = [1,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
    ,37,38,39,40,41,42,43]
channels_METABRIC2022 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]

data_dir = 'breast_cancer_imc/breast_data/'

data = {
    'Bodenmiller2020_Basel':
        {'path_meta':os.path.join(data_dir, 'bodenmiller2020/Basel_PatientMetadataALTERED.csv'),
         'path_images':os.path.join(data_dir, 'bodenmiller2020/OMEnMasks/ome/'),
         'channels': channelsBodenmiller2020_V2}, # use of all chennels even the ones with bad wuality

    'Bodenmiller2020_Zur':
        {'path_meta':os.path.join(data_dir, 'bodenmiller2020/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv'),
         'path_images':os.path.join(data_dir, 'bodenmiller2020/OMEnMasks/ome/'),
         'channels':channelsBodenmiller2020_V2},

    'Metabric2020':
        {'path_meta':os.path.join(data_dir, ''),
         'path_images':os.path.join(data_dir, 'imc_metabric/to_public_repository/full_stacks/'),
         'channels': channels_METABRIC2020},

    'Metabric2022':
        {'path_meta':os.path.join(data_dir, ''),
         'path_images':os.path.join(data_dir, 'imc_metabric2022/MBTMEIMCPublic/Images/'),
         'channels': channels_METABRIC2022},
        }


def get_images(dataset, remove_outlier, normalized, up_limit, down_limit):
    path_meta = data[dataset]['path_meta']
    path_images = data[dataset]['path_images']
    channels_to_keep = data[dataset]['channels']

    if dataset.__contains__('Bodenmiller'):
        file_metadata = pd.read_csv(path_meta)
        names_images = file_metadata['FileName_FullStack']
        files_images = [str(path_images + sub) for sub in names_images]
    elif dataset == 'Metabric2022':
        res = os.listdir(path_images)
        names_images = [x for x in res if x.__contains__('FullStack')] # 794
        print(len(names_images))
        files_images = [str(path_images + sub) for sub in names_images]
    else: # metabric 2020
        res = os.listdir(path_images)
        names_images = [x for x in res if x.__contains__('fullstack')] # 794
        print(len(names_images))
        files_images = [str(path_images + sub) for sub in names_images]

    images = map(IP.parse_image, files_images)
    imagesCH = map(lambda p: IP.simple_reduce_channels(p, channels_to_keep), images)
    # imagesCH = images

    if remove_outlier:
        # Outliers are removed through saturation of all pixels with values lower than the 1st and higher than the 99th percentile.
        imgNoOutlier = map(lambda p:IP.remove_outliers(p,up_limit,down_limit), imagesCH)
    else:
        imgNoOutlier = imagesCH

    if normalized =='by_dataset':
        imgNoOutlier2 = copy.deepcopy(imgNoOutlier)
        v_min, v_max = IP.get_maximum_minimum_per_channel_over_dataset(imgNoOutlier2)
        imgNorm = map(lambda p: IP.normalize_by_channel_based_on_dataset(p, v_min, v_max), imgNoOutlier)

    elif normalized == 'by_image':
        imgNorm = map(IP.normalize_by_channel, imgNoOutlier)

    else:
        imgNorm = copy.deepcopy(imgNoOutlier)

    imgNorm = list(imgNorm)
    return imgNorm

# save files
def create_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

def save_img_per_channel(list_of_images,path_to_write):
    dim_channel = np.array(list_of_images[0]).shape[2] # channels of the first image

    create_dir(path_to_write)
    # save per channel
    for channel in range(dim_channel):
        x = channel
        print(x)
        path_tiff = path_to_write + '/ch' + str(x) + '/'
        create_dir(path_tiff)
        i=0
        for img in list_of_images:
            arr = img[:,:,x]
            image_channel = np.array(arr)
            name_tiff = path_tiff + str(i) + '.tiff'
            with tifffile.TiffWriter(name_tiff) as tif:
                tif.write(image_channel, metadata={'channel':x, 'patient':i}, photometric='minisblack')
            i+=1

def save_img(list_of_images,path_to_write):
    create_dir(path_to_write)
    # save per image
    x=0
    for img in list_of_images:
        img = np.moveaxis(img, -1, 0)
        print(img.shape)
        name_tiff = path_to_write + '/full_image_' + str(x) + '.tiff'
        with tifffile.TiffWriter(name_tiff) as tif:
            tif.write(img, metadata={'patient':x}, photometric='minisblack')
            x+=1

if __name__ == '__main__':
    dataset = 'Metabric2022' # 'Bodenmiller2020_Basel' 'Bodenmiller2020_Zur' 'Metabric2020' 'Metabric2022'
    remove_outlier = True
    up_limit = 95
    down_limit = 1
    normalized = 'by_image' # 'by_image' 'by_dataset'
    print(up_limit)
    imgList = get_images(dataset, remove_outlier, normalized, up_limit, down_limit)

    # save per channel
    path_to_write = 'breast_cancer_imc/resultsPreprocess/'
    if remove_outlier:
        out = 'outlier_{}_{}V2'.format(up_limit,down_limit)
    else:
        out = 'no_outlierV2'

    final_path = os.path.join(path_to_write, '{}/{}_{}/per_channel'.format(dataset,out,normalized))
    save_img_per_channel(list_of_images = imgList,path_to_write = final_path)

    final_path = os.path.join(path_to_write, '{}/{}_{}/full_image'.format(dataset,out,normalized))

    save_img(list_of_images = imgList,path_to_write = final_path)



# # see pictures
# image_number = random.randint(0, len(images))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# image_example = imgNoOutlier[image_number,:,:,:,27,:]
# plt.imshow(image_example, cmap='gray')
# plt.show(block=True)




