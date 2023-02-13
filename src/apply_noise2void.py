import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import ImagePreprocessFilters as IPrep
import ImageParser as IP

dataset = 'metabric22'  # boden20 metabric20

path = '/home/martinha/PycharmProjects/phd/Preprocess_IMC/data/'
imgs_to_apply = path + dataset
path_for_results = '/home/martinha/PycharmProjects/phd/Preprocess_IMC/resultsPreprocess/'

# several channels per image
# several images per folder

# get_list of images to do
list_img_path = os.listdir(imgs_to_apply)
# read images
files = [str(imgs_to_apply + '/' + sub) for sub in list_img_path if 'tif' in sub]
# PARSE images to get numpy arrays into shape (, ,n_channels)
images = map(IP.parse_image, files)
# PERCENTILE SATURATION OUTLIERS
up_limit = 99  # 99
down_limit = 1  # 1
imgsOut = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images)
# NORMALIZE PER CHANNEL with function from OpenCV
imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgsOut)

imgs_norm = list(imgs_norm)  # to be used more again

# apply noise2void
from Noise2Void import noise2void,noise2void_one_image
clean_imgs = noise2void(img_list = imgs_norm, patch_size = 64, train_epochs = 100, train_batch_size =16)

# # noise2void 1 image
# clean_imgs = []
# for image in imgs_norm:
#     clean_imgs.append(noise2void_one_image(image, patch_size= 64, train_batch = 32))

# save images
path_res = path_for_results + dataset + '/N2V/noise2void_all_64pat_100epochs_16batch/'
if not os.path.exists(path_res):
    os.makedirs(path_res)
names_save = [str(path_res + sub) for sub in list_img_path if 'tif' in sub]

# for i in range(len(clean_imgs)):
#     IPrep.save_images(clean_imgs[i], names_save[i], ch_last=True)
images_test = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), clean_imgs, names_save)
#     print('saved')

print(list(images_test))
from ScoreNoise import calculate_psnr_snr_save
# calculate_psnr_snr_save(image1_true, image2_test, save_file)

names_save_psnr = [str(sub[:-5] + 'psnr.txt') for sub in names_save]

psnr = map(lambda p, f, file: calculate_psnr_snr_save(p, f, file),
           imgs_norm, images_test, names_save_psnr)
print(list(psnr))
#

for i in range(len(clean_imgs)):
    calculate_psnr_snr_save(imgs_norm[i],clean_imgs[i], names_save_psnr[i], ch_last=True)
    print('PSNR')




















