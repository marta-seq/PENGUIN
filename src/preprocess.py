import copy

import matplotlib.pyplot as plt
import numpy as np

import ImageParser as IP
from scipy import ndimage as nd


def show_pic(img, channels=[22, 28, 35], path_fig=None):
    plt.imshow(img[:, :, channels])  # [22,28,35] Cd45 Ki67 panCK
    plt.show()
    if path_fig:
        plt.savefig(path_fig)


def show_all_channels(img, path_png):
    n_channels = img.shape[2]  # number of images
    # sqrt of n _channels.
    # if is  int. is straigt that. if is plot n_lines = vaue + 1. n_columns =value
    sqr = np.sqrt(n_channels)
    if sqr.is_integer():
        n_row = sqr
        n_col = sqr
    else:
        n_row = int(sqr) + 1
        n_col = int(sqr)
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()

    for ch, ax in zip(range(n_channels), axs):
        img_one_ch = img[:, :, ch]
        ax.title(ch)
        ax.imshow(img_one_ch)

    plt.savefig(path_png)

    plt.show()


# img_path = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/' \
#            'PreprocessAnalysis/metabric22/MB0111_223_FullStack.tiff'
img_path = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/' \
           'PreprocessAnalysis/metabric22/MB3121_663_FullStack.tiff'
# read images
# maybe select useful channels

img_arr = IP.parse_image(img_path)


from skimage.metrics import peak_signal_noise_ratio
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)


# outlier percentile saturation
img_out = IP.remove_outliers(img_arr, up_limit=99, down_limit=1)

# use filters to denoise

# gaussianblur and median filterblur the images.
# use non local means
# https://github.com/bnsreenu/python_for_microscopists/blob/master/022-denoising.py
# https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html
from skimage.restoration import denoise_nl_means, estimate_sigma
import tifffile

sigma_est = np.mean(estimate_sigma(img_out, multichannel=True))
# denoise_img = denoise_nl_means(img_out, h=1.15 * sigma_est, fast_mode=False,
#                                patch_size=5, patch_distance=3, multichannel=True)
patch_kw = dict(patch_size=5,
                patch_distance=3,
                multichannel=True)

denoise_img = denoise_nl_means(img_out, h=0.9 * sigma_est, fast_mode=True,
                               # h=1.15 * sigma_est
                               **patch_kw)

# with tifffile.TiffWriter(name_tiff) as tif:
#     tif.write(img, metadata={'patient':x}, photometric='minisblack')



##################################################################################
### moved to IamgePreprocessing
# normalize and then boden

def normalize_by_channel(img: np.array) -> np.array:
    # Assuming you're working with image data of shape (W, H, 3), you should probably ' \
    #             'normalize over each channel (axis=2) separately, as mentioned in the other answer.
    # keepdims makes the result shape (1, 1, 3) instead of (3,). This doesn't matter here, but
    # would matter if you wanted to normalize over a different axis.
    v = img
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    # print(v_min)
    # print(v_max)
    new_img = (v - v_min) / (v_max - v_min)
    # new_img = new_img * (v_max - v_min) + v_min
    # new_img = (v_max - v_min)/(v_max - v_min)*(v-v_max)+v_max
    return new_img


def filter_hot_pixelsBodenmiller(img: np.ndarray, thres: float) -> np.ndarray:
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10
    from scipy.ndimage import maximum_filter
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)


img_norm_den = normalize_by_channel(img_out)
img_norm_boden_den05 = filter_hot_pixelsBodenmiller(img_norm_den, thres=0.5)

denoise_img2_t = np.moveaxis(img_norm_boden_den05, -1, 0)
tifffile.imwrite('denoise_boden50.tiff', denoise_img2_t,
                 photometric="minisblack")

##################################################################################



# This function finds the hot or dead pixels in a 2D dataset.
# tolerance is the number of standard deviations used to cutoff the hot pixels
# If you want to ignore the edges and greatly speed up the code, then set
# worry_about_edges to False.
#
# The function returns a list of hot pixels and also an image with with hot pixels removed
img_out = IP.remove_outliers(img_arr, up_limit=99, down_limit=1)
img_out = normalize_by_channel(img_out)

from scipy.ndimage import median_filter

blurred = median_filter(img_arr, size=3)
difference = img_arr - blurred
threshold = 10 * np.std(difference)

# find the hot pixels, but ignore the edges
hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
hot_pixels = np.array(hot_pixels) + 1  # because we ignored the first row and first column

fixed_image = np.copy(img_out)  # This is the image with the hot pixels removed
for y,x in zip(hot_pixels[0],hot_pixels[1]):
    fixed_image[y,x]=blurred[y,x]


# def find_outlier_pixels(data,tolerance=3,worry_about_edges=True):
#     #This function finds the hot or dead pixels in a 2D dataset.
#     #tolerance is the number of standard deviations used to cutoff the hot pixels
#     #If you want to ignore the edges and greatly speed up the code, then set
#     #worry_about_edges to False.
#     #
#     #The function returns a list of hot pixels and also an image with with hot pixels removed
#
#     from scipy.ndimage import median_filter
#     blurred = median_filter(Z, size=2)
#     difference = data - blurred
#     threshold = 10*np.std(difference)
#
#     #find the hot pixels, but ignore the edges
#     hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
#     hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column
#
#     fixed_image = np.copy(data) #This is the image with the hot pixels removed
#     for y,x in zip(hot_pixels[0],hot_pixels[1]):
#         fixed_image[y,x]=blurred[y,x]
#
#     if worry_about_edges == True:
#         height,width = np.shape(data)
#
#         ###Now get the pixels on the edges (but not the corners)###
#
#         #left and right sides
#         for index in range(1,height-1):
#             #left side:
#             med  = np.median(data[index-1:index+2,0:2])
#             diff = np.abs(data[index,0] - med)
#             if diff>threshold:
#                 hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
#                 fixed_image[index,0] = med
#
#             #right side:
#             med  = np.median(data[index-1:index+2,-2:])
#             diff = np.abs(data[index,-1] - med)
#             if diff>threshold:
#                 hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
#                 fixed_image[index,-1] = med
#
#         #Then the top and bottom
#         for index in range(1,width-1):
#             #bottom:
#             med  = np.median(data[0:2,index-1:index+2])
#             diff = np.abs(data[0,index] - med)
#             if diff>threshold:
#                 hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
#                 fixed_image[0,index] = med
#
#             #top:
#             med  = np.median(data[-2:,index-1:index+2])
#             diff = np.abs(data[-1,index] - med)
#             if diff>threshold:
#                 hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
#                 fixed_image[-1,index] = med
#
#         ###Then the corners###
#
#         #bottom left
#         med  = np.median(data[0:2,0:2])
#         diff = np.abs(data[0,0] - med)
#         if diff>threshold:
#             hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
#             fixed_image[0,0] = med
#
#         #bottom right
#         med  = np.median(data[0:2,-2:])
#         diff = np.abs(data[0,-1] - med)
#         if diff>threshold:
#             hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
#             fixed_image[0,-1] = med
#
#         #top left
#         med  = np.median(data[-2:,0:2])
#         diff = np.abs(data[-1,0] - med)
#         if diff>threshold:
#             hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
#             fixed_image[-1,0] = med
#
#         #top right
#         med  = np.median(data[-2:,-2:])
#         diff = np.abs(data[-1,-1] - med)
#         if diff>threshold:
#             hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
#             fixed_image[-1,-1] = med
#
#     return hot_pixels,fixed_image
#

# hot_pixels,fixed_image = find_outlier_pixels(Z)
# # https://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python
# plt.figure(figsize=(8,4))
# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122)
# #Then plot image original
# ax1.set_title('Raw data with hot pixels')
# ax1.imshow(img_out,interpolation='nearest',origin='lower')
#
# #Now we try to find the hot pixels
# blurred_Z = nd.gaussian_filter(img_out, sigma=1)
# difference = img_out - blurred_Z
#
# ax2.set_title('Difference with hot pixels identified')
# ax2.imshow(difference,interpolation='nearest',origin='lower')
#
# threshold = 15
# hot_pixels = np.nonzero((difference>threshold) | (difference<-threshold))


tifffile.imwrite('denoise_img.tiff', denoise_img,
                 photometric="minisblack")

# gaussian blur
# bodenmiller function
# try autoencoder


# gaussian_img = nd.gaussian_filter(img_out, sigma=3)
# plt.imsave("images/gaussian.jpg", gaussian_img)


import skimage

skimage.restoration.denoise_bilateral(img_out)

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import (
    filters, measure, morphology, segmentation
)
from skimage.data import protein_transport

# https://scikit-image.org/docs/dev/auto_examples/applications/plot_fluorescence_nuclear_envelope.html
smooth = filters.gaussian(img_out, sigma=0.7)

thresh_value = filters.threshold_otsu(smooth)
thresh = smooth > thresh_value  # is boolean false and true

# substitue True values from the thresh from the values from img_out
bool = np.where(thresh, 1, 0)
not_bool = np.ma.MaskedArray(img_out, mask=thresh)


# def take_out_single_pixels(img, neighboor_pixel=1):
#     import scipy.ndimage
#     if neighboor_pixel==1:
#         kernel = np.ones((1,3,3))
#     else:
#         kernel = np.ones((1,5,5))
#     kernel[0,2,2] = 0
#     tmp = scipy.ndimage.convolve(img, kernel, mode='constant')
#     out = np.logical_and(tmp > 0, img)
#     data = np.ma.MaskedArray(img, mask=out)
#     return data

def filter_isolated_cells(array, struct):
    import scipy.ndimage as ndimage

    """ Return array with completely isolated single cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :return: Array with minimum region size > 1
    """

    filtered_array = np.copy(array)
    array_bool = array[array != 0] = 1
    print(np.unique(array_bool))
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array_bool, id_regions, range(num_ids + 1)))
    print(id_sizes)
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


# Run function on sample array
new_array = np.copy(img_out)
for ch in range(img_out.shape[2]):
    new_array[ch] = filter_isolated_cells(img_out[ch], struct=np.ones((3, 3)))



img_out = IP.remove_outliers(img_arr, up_limit=99, down_limit=1)

np.count_nonzero(img_arr) # 3741194
# transform to bool (if it has values or not)
img_bool = np.where(img_arr > 0, 1, 0)
# np.count_nonzero(img_bool) # 3741194 should be the same and it is

############################################################################
# passed to ImagePreprocess
from scipy.ndimage import percentile_filter
kernel = np.ones((3, 3, 1))
percentile_blur = percentile_filter(img_bool, percentile=25, footprint=kernel)
# np.count_nonzero(percentile_blur)
# 1691740 took out 2049454  remained 45% pixels

bl = percentile_blur.astype(bool) # False is Zero
# np.count_nonzero(bl) 1691740

# data = copy.deepcopy(img_arr)
data = np.where(bl == False, 0, img_arr)
# or
# data[~bl] = 0
# 1581592 todo !!!
denoise_img2_t = np.moveaxis(data, -1, 0)
tifffile.imwrite('bol_percent25_3_3.tiff', denoise_img2_t,
                 photometric="minisblack")

#############################################################################





# # gaussian
#


# This function finds the hot or dead pixels in a 2D dataset.
# tolerance is the number of standard deviations used to cutoff the hot pixels
# If you want to ignore the edges and greatly speed up the code, then set
# worry_about_edges to False.
#
# The function returns a list of hot pixels and also an image with with hot pixels removed

blurred = median_filter(img_arr, size=3)
difference = img_arr - blurred
threshold = 10 * np.std(difference)

# find the hot pixels, but ignore the edges
hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
hot_pixels = np.array(hot_pixels) + 1  # because we ignored the first row and first column

fixed_image = np.copy(img_out)  # This is the image with the hot pixels removed



# to segment (did not move)
# https://scikit-image.org/docs/dev/auto_examples/applications/plot_fluorescence_nuclear_envelope.html
smooth = filters.gaussian(img_out, sigma=0.7)

thresh_value = filters.threshold_otsu(smooth)
thresh = smooth > thresh_value  # is boolean false and true

# substitue True values from the thresh from the values from img_out
bool = np.where(thresh, 1, 0)
not_bool = np.ma.MaskedArray(img_out, mask=thresh)





# def take_out_single_pixels(img, neighboor_pixel=1):
#     import scipy.ndimage
#     if neighboor_pixel==1:
#         kernel = np.ones((1,3,3))
#     else:
#         kernel = np.ones((1,5,5))
#     kernel[0,2,2] = 0
#     tmp = scipy.ndimage.convolve(img, kernel, mode='constant')
#     out = np.logical_and(tmp > 0, img)
#     data = np.ma.MaskedArray(img, mask=out)
#     return data

#
#
#






#############################################################################
# moved to ImagePreprocess
# np.count_nonzero(img_arr) # 3741194
# transform to bool (if it has values or not)
img_bool = np.where(img_arr > 0, 1, 0)
# np.count_nonzero(img_bool) # 3741194 should be the same and it is

from scipy.ndimage import percentile_filter, gaussian_filter
kernel = np.ones((3, 3, 1))

gaussian_blur = gaussian_filter(img_bool, sigma=0.2)
# 0.1 zeros are the same.
# 0.2 diminshes the zeros.   # 161311 non zero
# 0.3 none is zero
# np.count_nonzero(percentile_blur)
# 1691740 took out 2049454  remained 45% pixels

bl = gaussian_blur.astype(bool) # False is Zero
# np.count_nonzero(bl) 1691740

# data = copy.deepcopy(img_arr)
data = np.where(bl == False, 0, img_arr)
# or
# data[~bl] = 0
# 1581592 todo !!!
denoise_img2_t = np.moveaxis(data, -1, 0)
tifffile.imwrite('bol_percent25_3_3.tiff', denoise_img2_t,
                 photometric="minisblack")

#############################################################################








#
#
# # bodenmiller with 7 7 r 3 3

# the way I was doing

def filter_hot_pixelsBodenmiller(img: np.ndarray, thres: float) -> np.ndarray:
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10
    from scipy.ndimage import maximum_filter
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)


img_norm_den = normalize_by_channel(img_out)
img_norm_boden_den05 = filter_hot_pixelsBodenmiller(img_norm_den, thres=0.5)

denoise_img2_t = np.moveaxis(img_norm_boden_den05, -1, 0)
tifffile.imwrite('denoise_boden50.tiff', denoise_img2_t,
                 photometric="minisblack")


###############################################################################
# moved to ImagePreprocess
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
denoise_TV = denoise_tv_chambolle(img_arr, weight=0.3, multichannel=True)
###############################################################################

###############################################################################
# moved to ImagePreprocess
import bm3d

# for each marker in image
img_bm3d = np.empty(img_arr.shape)
for ch in range(img_arr.shape[2]):
    Img = img_arr[:,:,ch]
    BM3D_denoised_image = bm3d.bm3d(Img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    #BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

    img_bm3d[:,:,ch] = BM3D_denoised_image


img_dimr = np.float32(img_bm3d)
denoise_img2_t = np.moveaxis(img_dimr, -1, 0)
tifffile.imwrite('img_dimr_nei4_it3.tiff', denoise_img2_t,
                 photometric="minisblack")
###############################################################################


###############################################################################
# moved to apply filters
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
wavelet = np.empty(img_arr.shape)
for ch in range(img_arr.shape[2]):
    Img = img_arr[:,:,ch]
    wav = wavelet_smoothed = denoise_wavelet(Img, multichannel=False,
                                                 method='BayesShrink', mode='soft',
                                                 rescale_sigma=True)

    wavelet[:,:,ch] = wav


img_dimr = np.float32(wavelet)
denoise_img2_t = np.moveaxis(img_dimr, -1, 0)
tifffile.imwrite('wavelet.tiff', denoise_img2_t,
                 photometric="minisblack")
###############################################################################

###############################################################################
# moved to apply filters

dl = np.empty(img_arr.shape)
for ch in range(img_arr.shape[2]):
    Img = img_arr[:,:,ch]
    denoise_bilateral = denoise_bilateral(Img, sigma_spatial=15,
                                          multichannel=False)

    dl[:,:,ch] = denoise_bilateral


img_dimr = np.float32(dl)
denoise_img2_t = np.moveaxis(img_dimr, -1, 0)
tifffile.imwrite('denoise_bilateral.tiff', denoise_img2_t,
                 photometric="minisblack")

###############################################################################

###############################################################################
# moved to ImagePreprocess

from medpy.filter.smoothing import anisotropic_diffusion
# niter= number of iterations
#kappa = Conduction coefficient (20 to 100)
#gamma = speed of diffusion (<=0.25)
#Option: Perona Malik equation 1 or 2. A value of 3 is for Turkey's biweight function

dl = np.empty(img_arr.shape)
for ch in range(img_arr.shape[2]):
    Img = img_arr[:,:,ch]
    img_aniso_filtered = anisotropic_diffusion(Img, niter=50, kappa=50, gamma=0.2, option=2)

    dl[:,:,ch] = img_aniso_filtered

img_dimr = np.float32(dl)
denoise_img2_t = np.moveaxis(img_dimr, -1, 0)
tifffile.imwrite('img_aniso_filtered.tiff', denoise_img2_t,
                 photometric="minisblack")

###############################################################################




# # moved to apply_noise2void
# #### noise2void
# from n2v.models import N2VConfig, N2V
# from n2v.models import N2V
#
#
# # Let's look at the parameters stored in the config-object.
#
# # A previously trained model is loaded by creating a new N2V-object without providing a 'config'.
# from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
#
#
#
# dl = np.empty(img_arr.shape)
# for ch in range(img_arr.shape[2]):
#     Img = img_arr[:,:,ch]
#     datagen = N2V_DataGenerator()
#
#     patch_size = 64
#     # Patches are extracted from all images and combined into a single numpy array
#     patch_shape = (patch_size,patch_size)
#     patches = datagen.generate_patches_from_list(Img, shape=patch_shape)
#     # Patches are created so they do not overlap.
#     # (Note: this is not the case if you specify a number of patches. See the docstring for details!)
#     # Non-overlapping patches enable us to split them into a training and validation set.
#     train_val_split = int(patches.shape[0] * 0.8)
#     X = patches[:train_val_split]
#     X_val = patches[train_val_split:]
#
#     # train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
#     # is shown once per epoch.
#     train_batch = 32
#     config = N2VConfig(X, unet_kern_size=3,
#                        unet_n_first=64, unet_n_depth=3, train_steps_per_epoch=int(X.shape[0]/train_batch), train_epochs=20, train_loss='mse',
#                        batch_norm=True, train_batch_size=train_batch, n2v_perc_pix=0.198, n2v_patch_shape=(patch_size, patch_size),
#                        n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, single_net_per_channel=False)
#
#     # Let's look at the parameters stored in the config-object.
#     vars(config)
#
#     # a name used to identify the model --> change this to something sensible!
#     model_name = 'n2v_2D_stars'
#     # the base directory in which our model will live
#     basedir = 'models'
#     # We are now creating our network model.
#     model = N2V(config, model_name, basedir=basedir)
#
#     # We are ready to start training now.
#     history = model.train(X, X_val)
#     pred = model.predict(Img, axes='YX')
#     dl[:,:,ch] = pred
#
# img_dimr = np.float32(dl)
# denoise_img2_t = np.moveaxis(img_dimr, -1, 0)
# tifffile.imwrite('noise2void.tiff', denoise_img2_t,
#                  photometric="minisblack")

# # moved to aplyIMC_denoise
#
# ####### IMC denoise
# from IMC_Denoise.IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
# # https://github.com/PENGLU-WashU/IMC_Denoise/blob/main/Jupyter_Notebook_examples/IMC_Denoise_Train_and_Predict.ipynb
# n_neighbours = 4 # Larger n enables removing more consecutive hot pixels.
# n_iter = 3 # Iteration number for DIMR
# window_size = 3 # Slide window size. For IMC images, window_size = 3 is fine.
#
# # for each marker in image
# img_dimr = np.empty(img_arr.shape)
# for ch in range(img_arr.shape[2]):
#     Img_raw = img_arr[:,:,ch]
#     print(Img_raw)
#     Img_DIMR = DIMR(n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size).perform_DIMR(Img_raw)
#     img_dimr[:,:,ch] = Img_DIMR
#     print(img_dimr)
#     print(Img_DIMR)
# img_dimr = np.float32(img_dimr)
#
# denoise_img2_t = np.moveaxis(img_dimr, -1, 0)
# tifffile.imwrite('img_dimr_nei4_it3.tiff', denoise_img2_t,
#                  photometric="minisblack")
# # para treinar preciso de mais patches e n sei se quero xb
# # n ha trained para todos os markers
#
# # from IMC_Denoise.IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
# #
# # train_epoches = 50 # training epoches, which should be about 200 for a good training result. The default is 200.
# # train_initial_lr = 1e-3 # inital learning rate. The default is 1e-3.
# # train_batch_size = 128 # training batch size. For a GPU with smaller memory, it can be tuned smaller. The default is 256.
# # pixel_mask_percent = 0.2 # percentage of the masked pixels in each patch. The default is 0.2.
# # val_set_percent = 0.15 # percentage of validation set. The default is 0.15.
# # loss_function = "I_divergence" # loss function used. The default is "I_divergence".
# # weights_name = None # trained network weights saved here. If None, the weights will not be saved.
# # loss_name = None # training and validation losses saved here, either .mat or .npz format. If not defined, the losses will not be saved.
# # weights_save_directory = None # location where 'weights_name' and 'loss_name' saved.
# # # If the value is None, the files will be saved in a sub-directory named "trained_weights" of  the current file folder.
# # is_load_weights = False # Use the trained model directly. Will not read from saved one.
# # lambda_HF = 3e-6 # HF regularization parameter
# # deepsnif = DeepSNiF(train_epoches = train_epoches,
# #                     train_learning_rate = train_initial_lr,
# #                     train_batch_size = train_batch_size,
# #                     mask_perc_pix = pixel_mask_percent,
# #                     val_perc = val_set_percent,
# #                     loss_func = loss_function,
# #                     weights_name = weights_name,
# #                     loss_name = loss_name,
# #                     weights_dir = weights_save_directory,
# #                     is_load_weights = is_load_weights,
# #                     lambda_HF = lambda_HF)
# # train_loss, val_loss = deepsnif.train(generated_patches)
# #
# #
# # # perform DIMR and DeepSNiF algorithms for low SNR raw images.
# # Img_DIMR_DeepSNiF = deepsnif.perform_IMC_Denoise(mg_raw, n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size)
# # plt.figure(figsize = (10,8))
# # plt.imshow(Img_DIMR_DeepSNiF, vmin = 0, vmax = 0.5*np.max(Img_DIMR_DeepSNiF), cmap = 'jet')
# # plt.colorbar()
# # plt.show()
