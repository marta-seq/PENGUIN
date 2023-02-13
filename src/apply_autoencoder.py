import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import ImagePreprocessFilters as IPrep
import ImageParser as IP


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
# PARSE images to get numpy arrays into shape (, ,n_channels)
images = map(IP.parse_image, files)
# PERCENTILE SATURATION OUTLIERS
up_limit = 99  # 99
down_limit = 1  # 1
imgsOut = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images)
# NORMALIZE PER CHANNEL with function from OpenCV
imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgsOut)

imgs_norm = list(imgs_norm)  # to be used more again

from AutoEncoder import train_predict_autoencoder, train_predict_autoencoder_per_channel

# reconstructed_autoenc = train_predict_autoencoder(images_train = imgs_norm,
#                                                  images_test = imgs_norm,  # will train and solved it the same images
#                                                  dir = 'Preprocess_IMC/resultsPreprocess/autoencoder_results/',
#                                                  batch_size = 16,
#                                                  EPOCHS = 100)



reconstructed_autoenc = train_predict_autoencoder_per_channel(images_train= imgs_norm,
                                                              images_test= imgs_norm,
                                                              dir = 'Preprocess_IMC/resultsPreprocess/autoencoder_results/',
                                                              batch_size = 8,
                                                              EPOCHS = 50,
                                                              PatchSize = 16, step = 16)

path_res = path_for_results + dataset + '/autoencoderUNET_results/v1_bs16_epochs50_16_per_ch/'
if not os.path.exists(path_res):
    os.makedirs(path_res)
names_save = [str(path_res + sub) for sub in list_img_path if 'tif' in sub]

images_test = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), reconstructed_autoenc, names_save)
print('saved')

print(list(images_test))
# imgs = list(imgs_filtered)# they have different shapes


def calculate_psnr_snr(image1_true, image2_test, save_file):
    # Assumes image1 and image2 are numpy arrays with the same shape and dtype
    # do this by channel
    mse = np.mean((image1_true - image2_test) ** 2, axis=(0, 1))
    snr = np.mean(image1_true ** 2, axis=(0, 1)) / mse
    psnr = 10 * np.log10(np.amax(image1_true) ** 2 / mse)

    with open(save_file, 'w') as fp:
        fp.write('\n'.join([str(psnr.round(4))]))
    return psnr


names_save_psnr = [str(sub[:-5] + 'psnr.txt') for sub in names_save]

psnr = map(lambda p, f, file: calculate_psnr_snr(p, f, file),
           imgs_norm, images_test, names_save_psnr)
print(list(psnr))

# save psnr and snr


