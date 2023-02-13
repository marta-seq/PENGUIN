import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import ImagePreprocessFilters as IPrep
import ImageParser as IP
import IMC_Denoise as ID

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

dataset = 'metabric22'  # boden20 metabric20

path = '/home/martinha/PycharmProjects/phd/Preprocess_IMC/data/'
imgs_to_apply = path + dataset
path_for_results = '/home/martinha/PycharmProjects/phd/Preprocess_IMC/resultsPreprocess/'

# several channels per
# image
# several images per folder

# get_list of images to do
list_img_path = os.listdir(imgs_to_apply)
# read images
files = [str(imgs_to_apply + '/' + sub) for sub in list_img_path if 'tif' in sub]

# apply only DIMR
# Perform the DIMR algorithm only if the SNR of the raw image is high
# dont know what does this mean .
# try to applied to all

# asit is er channel and they dont say anything about normalizing. I din't
images = map(IP.parse_image, files)
# images_dimr = map(lambda p: ID.apply_DIMR(p, n_neighbours=5, n_iter=20, window_size=3), images)

# # to create dir as requested ( only need to do this once)
ID.create_dir_as_requested(files)
images_dimr = ID.apply_IMCDenoise_full(list(images), n_channels = 39,
                                                     n_neighbours=4, n_iter=3,
                                                     window_size=3)
i_dimr = list(images_dimr)

path_res = path_for_results + dataset + '/IMC_Denoise/imc_denoise_4nei_3iter_3window_200epochs_5channels/'
if not os.path.exists(path_res):
    os.makedirs(path_res)
names_save = [str(path_res + sub) for sub in list_img_path if 'tif' in sub]

images_test = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), i_dimr, names_save)

images_test = list(images_test)
print(images_test)
from ScoreNoise import calculate_psnr_snr_save
names_save_psnr = [str(sub[:-5] + 'psnr.txt') for sub in names_save]

psnr = map(lambda p, f, file: calculate_psnr_snr_save(p, f, file),
           i_dimr, images_test, names_save_psnr)
print(list(psnr))



#

