

import copy

import matplotlib.pyplot as plt
import numpy as np
import os
import ImageParser as IP
import ImagePreprocessFilters as IPrep

from scipy import ndimage as nd

# open Marieke images
# apply something
# get scores accordingly to the ones she has preprocessed
# save on something results/Marieke


# open Marieke images
dir_to_apply = '/home/martinha/PycharmProjects/phd/Preprocess_IMC/data/Marieke2/2.ROIs_raw'
list_pat = os.listdir(dir_to_apply)

for pat in list_pat:
    imgs_to_apply = os.path.join(dir_to_apply, pat)
    path_for_results = os.path.join('/home/martinha/PycharmProjects/phd/Preprocess_IMC/Marieke_resultsPreprocessAll/',
                                    pat)

    # get_list of images to do
    # find all images with ome.tiff and ome_Threshold
    list_img_path = os.listdir(imgs_to_apply)
    files_original = [str(imgs_to_apply + '/' + sub) for sub in list_img_path if '.ome.tiff' in sub]
    files_true = [str(imgs_to_apply + '/' + sub) for sub in list_img_path if 'ome_Threshold.tiff' in sub]

    images = map(IP.parse_image, files_original)
    # PERCENTILE SATURATION OUTLIERS
    up_limit = 99  # 99
    down_limit = 1  # 1
    imgsOut = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images)
    # NORMALIZE PER CHANNEL with function from OpenCV
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgsOut)

    # remove th 0.8
    imgs_norm2 = map(lambda p: IPrep.out_ratio(p, th=0.4), imgs_norm)
    # PERCENTILE FILTERS
    imgs_filtered = map(lambda p: IPrep.percentile_filter(p, window_size=3, percentile=50, transf_bool=True), imgs_norm2)
    # if 2 consecutive median filters
    # imgs_filtered = map(lambda p: IPrep.percentile_filter(p, window_size=3, percentile=50, transf_bool=True), imgs_filtered2)

    # Percentile filters with different percentiles per Channel
    # percentil_list = [50,50,50,50,50,50,50,50,25,25,25,50,50,50,50,50,25,50,25,50,50,50,50,25,25,
    #                   50,50,50,50,50,50,50,50,50,50,50,50,25,25]
    # imgs_filtered = map(lambda p: IPrep.percentile_filter_changedpercentiles(p, window_size=3,
    #                                                                          percentiles=percentil_list,
    #                                                                          transf_bool=True), imgs_norm)

    # hybrid median
    # imgs_filtered = map(lambda p: IPrep.hybrid_median_filter(p, window_size=3, percentile=50, transf_bool = True ), imgs_norm)

    # adaptive median filter
    # imgs_filtered = map(lambda p: IPrep.adaptive_median_filter(p, max_size = 7, transf_bool = True), imgs_norm)

    # morphological Filter
    # imgs_filtered = map(lambda p: IPrep.morphological_filter(p, structuring_element_size=3), imgs_norm)

    # BODENMILLER
    # imgs_filtered = map(lambda p: IPrep.modified_hot_pixelsBodenmiller(p, thres=0.2, window_size= 3), imgs_norm)  # ,

    # save images
    path_res = path_for_results + '/percentile_filter_window3_th04_p50/'
    if not os.path.exists(path_res):
        os.makedirs(path_res)
    names_save = [str(path_res + sub[:-9] + 'th04_p50.tiff') for sub in list_img_path if '.ome.tiff' in sub]

    images_test = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)
    images_test = list(images_test)
    print(images_test)
    print('saved')

    def calculate_psnr_snr(image1_true, image2_test, save_file):
        # Assumes image1 and image2 are numpy arrays with the same shape and dtype
        # do this by channel
        mse = np.mean((image1_true - image2_test) ** 2, axis=(0, 1))
        snr = np.mean(image1_true ** 2, axis=(0, 1)) / mse
        psnr = 10 * np.log10(np.amax(image1_true) ** 2 / mse)

        n_pixels_changed = np.sum(image2_test != image1_true)
        # print(np.where(img_out != img))

        with open(save_file, 'w') as fp:
            fp.write('\n'.join([str(psnr.round(4)), 'pixels changed {}'.format(n_pixels_changed)]))
            fp.write()

        return psnr


    names_save_psnr = [str(sub[:-5] + 'psnr.txt') for sub in names_save]
    imgs_true = list(map(IP.parse_image,files_true))
    psnr = map(lambda p, f, file: calculate_psnr_snr(p, f, file),
               imgs_true, images_test, names_save_psnr)
    print(list(psnr))
