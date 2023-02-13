"""
Try to smooth pixels

One alternative is to remove hot pixels? like check if the the pixel has neighbors.

My alternative ( check the literature also!!!) is to substitute every pixel value
with the mean (or median) value of a window size. then a pixel alone surrounded by
zero values will fade away
"""

from apeer_ometiff_library import io
import numpy as np
import copy
import tifffile

import utils.ImageParser as IP

# open one image

def smooth(img,distance_p = 2):
    # Smooth
    smooth_img = copy.deepcopy(img)
    # distance_p = 2  # 2 for window 5 to 5
    for i in range(img.shape[2]):  # range in channels
        ch = img[:, :, i]  # 2 dim

        for pixel_x in range(img.shape[0]):
            for pixel_y in range(img.shape[1]):
                pixel = ch[pixel_x, pixel_y]

                # taking care of corners of image
                if pixel_x - distance_p < 0:
                    min_pixel_x = 0
                else:
                    min_pixel_x = pixel_x - distance_p
                if pixel_x + distance_p > img.shape[0]:
                    max_pixel_x = img.shape[0]
                else:
                    max_pixel_x = pixel_x + distance_p

                if pixel_y - distance_p < 0:
                    min_pixel_y = 0
                else:
                    min_pixel_y = pixel_y - distance_p
                if pixel_y + distance_p > img.shape[1]:
                    max_pixel_y = img.shape[1]
                else:
                    max_pixel_y = pixel_y + distance_p

                # get mean values
                window = ch[min_pixel_x:max_pixel_x, min_pixel_y:max_pixel_y]
                mean = np.mean(window)
                # mean = np.percentile(window, 25) # percentile 25
                # median?
                # weighted average?

                # take care of corners because will give a error
                smooth_img[pixel_x, pixel_y, i] = mean

    print(smooth_img.shape)
    return smooth_img

img_path = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/imc_metabric2022/MBTMEIMCPublic/Images/MB0000_64_FullStack.tiff'
img = IP.parse_image(img_path)
distance_p = 2 # window 5 5      distance1:window 3 3
fun = 'mean'
smooth_img = smooth(img)


path_to_write = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/resultsPreprocess/smooth/'
name_image = img_path.rsplit("/",1)[1] # the name of the image

smooth_img_write = np.moveaxis(smooth_img, -1, 0)
name_tiff = path_to_write + name_image[:len(name_image)-5] + 'distance_p{}_fun_{}.tiff'.format(distance_p,fun)
with tifffile.TiffWriter(name_tiff) as tif:
    tif.write(smooth_img_write, metadata={'img':name_image, 'smooth_distance_p':distance_p, 'smooth_fun':fun,
                             'normalization':'none', 'outlier':'none'}, photometric='minisblack')

# OUTLIER REMOVE
smooth_img_out = IP.remove_outliers(smooth_img, up_limit=99, down_limit=1)
smooth_out_write = np.moveaxis(smooth_img_out, -1, 0)
name_tiff = name_tiff[:len(name_tiff)-5] + 'outlier.tiff'
with tifffile.TiffWriter(name_tiff) as tif:
    tif.write(smooth_out_write, metadata={'img':name_image, 'smooth_distance_p':distance_p, 'smooth_fun':fun,
                                    'normalization':'none', 'outlier':'99_1'}, photometric='minisblack')

# NORMALIZATION
smooth_img_norm = IP.normalize_by_channel(smooth_img_out)
smooth_norm_write = np.moveaxis(smooth_img_norm, -1, 0)
name_tiff = name_tiff[:len(name_tiff)-5] + 'outlier_norm.tiff'
with tifffile.TiffWriter(name_tiff) as tif:
    tif.write(smooth_norm_write, metadata={'img':name_image, 'smooth_distance_p':distance_p, 'smooth_fun':fun,
                                    'normalization':'perchannel', 'outlier':'99_1'}, photometric='minisblack')



# # the original
# name_tiff = path_to_write + name_image
# img_write = np.moveaxis(img, -1, 0)
# with tifffile.TiffWriter(name_tiff) as tif:
#     tif.write(img_write, metadata={'img':name_image, 'smooth_distance_p':None, 'smooth_fun':None,
#                                     'normalization':'none', 'outlier':'none'}, photometric='minisblack')
#
# # OUTLIER REMOVE
# smooth_img_out = IP.remove_outliers(img, up_limit=99, down_limit=1)
# smooth_img_out_write = np.moveaxis(smooth_img_out, -1, 0)
# name_tiff = name_tiff[:len(name_tiff)-5] + 'outlier.tiff'
# with tifffile.TiffWriter(name_tiff) as tif:
#     tif.write(smooth_img_out_write, metadata={'img':name_image, 'smooth_distance_p':distance_p, 'smooth_fun':'mean',
#                                         'normalization':'none', 'outlier':'99_1'}, photometric='minisblack')
#
# # NORMALIZATION
# smooth_img_norm = IP.normalize_by_channel(smooth_img_out)
# smooth_img_out_write = np.moveaxis(smooth_img_norm, -1, 0)
# name_tiff = name_tiff[:len(name_tiff)-5] + 'outlier_norm.tiff'
# with tifffile.TiffWriter(name_tiff) as tif:
#     tif.write(smooth_img_out_write, metadata={'img':name_image, 'smooth_distance_p':distance_p, 'smooth_fun':'mean',
#                                          'normalization':'perchannel', 'outlier':'99_1'}, photometric='minisblack')
#
#



def filter_hot_pixelsBodenmiller(img: np.ndarray, thres: float) -> np.ndarray:
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10
    from scipy.ndimage import maximum_filter
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)

T = 0.01 # threshold
smooth_img_Boden = filter_hot_pixelsBodenmiller(img, thres=T)
smooth_img_write = np.moveaxis(smooth_img_Boden, -1, 0)
name_tiff = path_to_write + name_image[:len(name_image)-5] + 'Bodenmiller{}.tiff'.format(T)
with tifffile.TiffWriter(name_tiff) as tif:
    tif.write(smooth_img_write, metadata={'img':name_image, 'threshold':T,
                                          'normalization':'none', 'outlier':'none'}, photometric='minisblack')

# OUTLIER REMOVE
smooth_img_out = IP.remove_outliers(smooth_img_Boden, up_limit=99, down_limit=1)
smooth_out_write = np.moveaxis(smooth_img_out, -1, 0)
name_tiff = name_tiff[:len(name_tiff)-5] + 'outlier.tiff'
with tifffile.TiffWriter(name_tiff) as tif:
    tif.write(smooth_out_write, metadata={'img':name_image, 'threshold':T,
                                          'normalization':'none', 'outlier':'99_1'}, photometric='minisblack')

# NORMALIZATION
smooth_img_norm = IP.normalize_by_channel(smooth_img_out)
smooth_norm_write = np.moveaxis(smooth_img_norm, -1, 0)
name_tiff = name_tiff[:len(name_tiff)-5] + 'outlier_norm.tiff'
with tifffile.TiffWriter(name_tiff) as tif:
    tif.write(smooth_norm_write, metadata={'img':name_image, 'threshold':T,
                                           'normalization':'perchannel', 'outlier':'99_1'}, photometric='minisblack')

