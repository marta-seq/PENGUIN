""""
Example script for images with stacks of channel per image
Please adapt accordingly
"""

import os
import sys
from glob import glob
import numpy as np
from src.file_specs import FileSpecifics
import src.ImagePreprocessFilters as IPrep
import src.ImageParser as IP

def preprocess_image(img, thresholds, percentiles):
    filtered_img = np.empty(img.shape)
    for ch in range(img.shape[2]):
        img_ch = img[:, :, ch]

        # Thresholding
        th = thresholds[ch]
        if th is not None:
            img_ch = np.where(img_ch >= th, img_ch, 0)

        # Percentile filtering
        perc = percentiles[ch]
        if perc is not None:
            img_ch = img_ch[..., np.newaxis]
            img_ch = IPrep.percentile_filter(img_ch, window_size=3, percentile=perc, transf_bool=True)
            img_ch = img_ch.squeeze()

        filtered_img[:, :, ch] = img_ch
    return filtered_img


if __name__ == "__main__":
    folder_path = 'data_test/all_ch/METABRIC22_sample/'
    # folder_path = 'data_test/all_ch/stacks_with_names/'
    path_for_results = 'results_percentile/'

    # normalization outliers
    up_limit = 99
    down_limit = 1
    binary_masks = False

    # Load files
    files = glob(os.path.join(folder_path, '*.tiff'))
    num_images = len(files)
    print(f"Number of images identified: {num_images}")
    if num_images == 0:
        sys.exit(1)

    # Parse image channels
    specs = FileSpecifics(files[0], multitiff=True)
    channel_names = specs.channel_names
    print('Channel names: ', channel_names)
    num_channels = len(channel_names)

    # Calculate thresholds and percentiles
    thresholds = [0.1 for _ in range(num_channels) ]
    percentiles = [0.5 for _ in range(num_channels)]

    images_original = list(map(IP.parse_image_pages, files))

    # Preprocessing
    imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)
    filtered_images = map(lambda i: preprocess_image(i, thresholds, percentiles), imgs_norm)
    imgs_filtered = list(filtered_images)

    # Apply binary masks if needed
    if binary_masks:
        imgs_filtered = [np.where(a > 0, 1, 0) for a in imgs_filtered]

    # Save images
    names_save = [os.path.join(path_for_results, os.path.basename(sub)) for sub in files]
    if isinstance(self.channel_names[0], str):
        images_final = map(
            lambda p, f: IPrep.save_img_ch_names_pages(p, f, ch_last=True, channel_names=self.channel_names),
            imgs_filtered, names_save)

    else:
        # will not save channel names
        images_final = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)

    print(f'Images saved at {self.path_for_results}')

