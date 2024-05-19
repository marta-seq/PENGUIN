""""
Example script for images with one channel in each file
Please adapt accordingly
"""

import os
import sys
from glob import glob
import numpy as np
from src.file_specs import FileSpecifics
import src.ImagePreprocessFilters as IPrep
import src.ImageParser as IP


def preprocess_image(file_paths, path_for_results, up_limit=99, down_limit=1, threshold=None, percentile=50,
                     binary_masks=False):
    images_original = list(map(IP.parse_image, file_paths))

    if len(images_original[0].shape) == 2:  # Check if the shape is 2D
        images_original = [np.expand_dims(img, axis=-1) for img in images_original]

    # PERCENTILE SATURATION OUTLIERS
    imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)

    # NORMALIZE PER CHANNEL with function from OpenCV
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)

    # THRESHOLDING
    if isinstance(threshold, float):
        imgs_filtered = list(map(lambda p: IPrep.out_ratio2(p, th=threshold), imgs_norm))
    elif threshold is None:
        imgs_filtered = imgs_norm
    elif threshold in ['otsu', 'isodata', 'Li', 'Yen', 'triangle', 'mean']:
        threshold_fn = getattr(IPrep, f'th_{threshold}')
        imgs_filtered = list(map(threshold_fn, imgs_norm))
    elif threshold == 'local':
        imgs_filtered = list(map(lambda p: IPrep.th_local(p, block_size=3, method='gaussian'), imgs_norm))
    else:
        raise ValueError(f"Invalid threshold type: {threshold}")

    if percentile is not None:
        imgs_filtered = map(
            lambda p: IPrep.percentile_filter(p, window_size=3, percentile=percentile, transf_bool=True),
            imgs_filtered)
        imgs_filtered = list(imgs_filtered)

    if binary_masks:
        imgs_filtered = [np.where(a > 0, 1, 0) for a in imgs_filtered]

    names_save = [os.path.join(path_for_results, os.path.basename(os.path.dirname(sub)), os.path.basename(sub)) for
                  sub in file_paths]
    map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)
    print('Images saved at ', path_for_results)


if __name__ == "__main__":
    folder_path = 'data_test/one_ch/'
    path_for_results = 'results_percentile/'

    # normalization outliers
    up_limit = 99
    down_limit = 1

    # Thresholding
    threshold = None
    percentile = 50
    binary_masks = False

    # load files
    files = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], '*.ome.tiff'))]
    num_images = len(files)
    print(f"Number of images identified: {num_images}")
    if num_images == 0:
        sys.exit(1)

    channel_names = set([name.split("_")[-1].split(".ome.tiff")[0] for name in files])

    # for channel in channel_names:
    #     files_channel = [file for file in files if str(channel + '.ome.tiff') in str(file)]
    #
    #     paths_save = [str(path_for_results + os.path.basename(os.path.dirname(sub))) for sub in files_channel]
    #
    #     preprocess_image(files_channel, path_for_results, up_limit, down_limit, threshold, percentile, binary_masks)
    #     print(f'Channel: {channel}, Percentile: {percentile}, thresholding: {threshold}')

    channel_names = ['CD45', 'CD68','CD31','Bcatenin', 'Vimentin']
    thresholds = [0.1,None, 0.1,0.1, None]
    percentiles = [0.5,0.5,0.5,0.5,0.5]

    for channel, th, perc in zip(channel_names, thresholds, percentiles):
        files_channel = [file for file in files if str(channel + '.ome.tiff') in str(file)]

        paths_save = [str(path_for_results + os.path.basename(os.path.dirname(sub))) for sub in files_channel]

        preprocess_image(files_channel, path_for_results, up_limit, down_limit, th, perc, binary_masks)
        print(f'Channel: {channel}, Percentile: {perc}, Thresholding: {th}')

