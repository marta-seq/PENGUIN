import os
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


imgs_norm2 = map(lambda p: IPrep.out_ratio(p, th=0.4), imgs_norm)


# PERCENTILE FILTERS
imgs_filtered = map(lambda p: IPrep.percentile_filter(p, window_size=3, percentile=50, transf_bool=True), imgs_norm2)
# if 2 consecutive median filters
# imgs_filtered = map(lambda p: IPrep.percentile_filter(p, window_size=3, percentile=50, transf_bool=True), imgs_filtered)

# Percentile filters with different percentiles per Channel
# percentil_list = [50,50,50,50,50,50,50,50,25,25,25,50,50,50,50,50,25,50,25,50,50,50,50,25,25,
#                   50,50,50,50,50,50,50,50,50,50,50,50,25,25]
# imgs_filtered = map(lambda p: IPrep.percentile_filter_changedpercentiles(p, window_size=3,
#                                                                          percentiles=percentil_list,
#                                                                          transf_bool=True), imgs_norm)

# hybrid median
# imgs_filtered = map(lambda p: IPrep.hybrid_median_filter(p, window_size=5, percentile=50, transf_bool = True ), imgs_norm)

# adaptive median filter
# imgs_filtered = map(lambda p: IPrep.adaptive_median_filter(p, max_size = 7, transf_bool = True), imgs_norm)

# hybrid version ChatGPT?


# morphological Filter
# imgs_filtered = map(lambda p: IPrep.morphological_filter(p, structuring_element_size=3), imgs_norm)

# BODENMILLER
# imgs_filtered = map(lambda p: IPrep.modified_hot_pixelsBodenmiller(p, thres=0.2, window_size= 3), imgs_norm)  # ,

# GAUSSIAN
# imgs_filtered = map(lambda p: IPrep.gaussian_filter(p, sigma=0.3), imgs_norm)

# Non local means
# imgs_filtered = map(lambda p: IPrep.non_local_means_filter(p, patch_size=5, patch_distance=6), imgs_norm)

# bilateral
# imgs_filtered = map(lambda p: IPrep.bilateral_filter(p, sigma=None), imgs_norm)

# TV chambolle
# imgs_filtered = map(lambda p: IPrep.total_variation_filter(p, weight=0.3), imgs_norm)

# wavelet
# imgs_filtered = map(lambda p: IPrep.wavelet_filter(p), imgs_norm)

# anisotropin filter n_iter = 50  equation = 2    equation 2 favours wide regions over smaller ones. See [1]_ for details.
# imgs_filtered = map(lambda p: IPrep.anisotropic_filtering(p, niter=5, kappa=50, gamma=0.2, option=1), imgs_norm)

# bm3d
# imgs_filtered = map(lambda p: IPrep.bm3d_filter(p, sigma_psd=0.1), imgs_norm)

# save images
path_res = path_for_results + dataset + '/th04_median50/'
if not os.path.exists(path_res):
    os.makedirs(path_res)
names_save = [str(path_res + sub) for sub in list_img_path if 'tif' in sub]

images_test = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)
print('saved')


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


# from skimage.metrics import peak_signal_noise_ratio as psnr
# psnr(noisy_image, denoised_image)
# metrics =

# get all the PSNR f the original and from the filtered ones
# get from channel?
