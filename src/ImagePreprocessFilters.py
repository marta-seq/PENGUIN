import copy
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from scipy.ndimage import percentile_filter
import tifffile


def remove_outliers(img: np.array, up_limit=99, down_limit=1) -> np.array:
    """
    Remove outliers in an image by saturating pixel values.
    This function sharpens images to facilitate pixel annotation by removing outliers.
    It saturates all pixels with values lower than the down limit (default 1st percentile)
    and higher than the up limit (default 99th percentile). The saturation is performed per channel.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.
    up_limit : int, optional
        The upper limit percentile for saturation. All pixel values above this percentile
        are set to this value. Default is 99.
    down_limit : int, optional
        The lower limit percentile for saturation. All pixel values below this percentile
        are set to this value. Default is 1.

    Returns
    -------
    np.ndarray
        The image with outliers removed through saturation.

    Examples
    --------
    >>> img = np.random.rand(100, 100, 3) * 255
    >>> img_saturated = remove_outliers(img, up_limit=99, down_limit=1)
    >>> print(img_saturated.shape)

    Notes
    -----
    The function modifies the image in-place and saturates pixel values based on the specified
    percentile limits per channel.
    """
    # Check if the image has 3 dimensions (height, width, channels)
    if len(img.shape) < 3:
        # If not, add an extra dimension to make it a 3-channel image
        img = np.expand_dims(img, axis=-1)

    imOutlier = img
    for i in range(img.shape[2]):
        ch = img[:, :, i]
        p_99 = np.percentile(ch, up_limit)
        p_01 = np.percentile(ch, down_limit)
        if p_99 > 0:
            ch = np.where(ch > p_99, p_99, ch) # instead of substitube by 0
        ch = np.where(ch < p_01, p_01, ch)
        imOutlier[:, :, i] = ch

    return imOutlier

def normalize_channel_cv2_minmax(img: np.array) -> np.array:
    """
    Normalize each channel of the image using the OpenCV min-max normalization function.

    This function processes each channel of the input image independently,
    normalizing pixel values to the range [0, 1] using OpenCV's `cv2.normalize` method.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array with shape (H, W, C), where H is the height,
        W is the width, and C is the number of channels.

    Returns
    -------
    np.ndarray
        The image with each channel min-max normalized to the range [0, 1].

    Examples
    --------
    >>> img = np.random.rand(100, 100, 3) * 255
    >>> normalized_img = normalize_channel_cv2_minmax(img)
    >>> print(normalized_img.shape)
    >>> print(normalized_img.min(), normalized_img.max())

    Notes
    -----
    This normalization ensures that the minimum value of each channel is 0 and
    the maximum value is 1.
    """

    normalized = np.zeros(img.shape)
    for ch in range(img.shape[2]):
        out_img = img[:, :, ch]
        out = np.zeros(out_img.shape, np.double)
        normalized_ch = cv2.normalize(out_img, out, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        normalized[:, :, ch] = normalized_ch
    return normalized

def out_ratio2(img: np.ndarray, th: float = 0.1) -> np.ndarray:
    """
    Apply thresholding to an image.

    This function sets all pixels below the specified threshold to 0.
    The input image should be a single-channel image if thresholding is to be applied per channel.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.
    th : float, optional
        The threshold value. All pixel values below this threshold will be set to 0.
        Default is 0.1.

    Returns
    -------
    np.ndarray
        The thresholded image, with all pixels below the threshold set to 0.

    Examples
    --------
    >>> img = np.random.rand(100, 100)
    >>> thresholded_img = out_ratio2(img, th=0.2)
    >>> print(thresholded_img.min(), thresholded_img.max())

    Notes
    -----
    The function performs element-wise thresholding, so it is suitable for both 2D and 3D arrays.

    """
    new_img = np.where(img >= th, img, 0)
    return new_img



def percentile_filter(img: np.ndarray, window_size: int = 3, percentile: int = 50,
                      transf_bool: bool = True, out_ratio: bool = False) -> np.ndarray:
    """
    Apply a percentile filter ( Scipy implementation) to the image.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array. If different values per channel this should be only the specific channel.
    window_size : int, optional
        The size of the kernel window. Default is 3, creating a 3x3 kernel.
    percentile : int, optional
        The percentile value to be applied to the array inside the window.
    transf_bool : bool, optional
        If True, the function will transform the image to boolean values (0 or 1) before applying the filter.
        The percentile filter is only applied to determine which pixels are noise and only those are set to 0.
        This prevents unnecessary blur by identifying noise and setting those values to zero.
        Default is True.

    Returns
    -------
    np.ndarray
        The image filtered with the percentile method.

    Notes
    -----
    - The function applies a percentile filter to the input image.
    - The kernel used for filtering is a square window of size (window_size, window_size).
    - If `transf_bool` is True, the image is transformed to boolean values (0 or 1) before filtering,
      preventing unnecessary blur by identifying noise and setting those values to zero.
    - If `transf_bool` is False, the filter is applied directly to the image.
    """

    kernel = np.ones((window_size, window_size, 1))
    # if disk instead square is better
    # from skimage.morphology import disk
    # kernel = disk(window_size) # disk(3) will create a window 5 *5 plus points on edges
    if transf_bool:

        img_to_apply = np.where(img > 0, 1, 0)

    else:
        # will apply the filter directly to the image the positive values from the image will be different
        img_to_apply = copy.deepcopy(img)

    nzero = np.count_nonzero(img_to_apply)

    percentile_blur = nd.percentile_filter(img_to_apply,
                                        percentile=percentile,
                                        footprint=kernel)
    nzero_filter = np.count_nonzero(percentile_blur)

    if transf_bool:
        bl = percentile_blur.astype(bool) # False is Zero
        percentile_blur = np.where(bl == False, 0, img)

    return percentile_blur

def save_images(img: np.array, name:str, ch_last:bool = True)-> np.array:
    """
    Simple Save a numpy array as a TIFF file.

    This function allows saving a numpy array as a TIFF file. The array can represent an image with multiple channels,
    and the function provides an option to specify whether the channels are the last axis of the array.

    Parameters
    ----------
    img : np.array
        The input image as a NumPy array.
    name : str
        The file path to save the image.
    ch_last : bool, optional
        Specifies whether the channels are the last axis of the numpy array.
        If True, the array shape is assumed to be (height, width, channels).
        If False, the array shape is assumed to be (channels, height, width).
        Default is True.

    Returns
    -------
    np.array
        The input image numpy array.

    Notes
    -----
    - The function saves the image as a TIFF file using the tifffile library.
    - It converts the input array to float32 before saving to ensure compatibility.
    """
    img_save = np.float32(img)
    if ch_last == True: # channel is the last axis
        img_save = np.moveaxis(img_save, -1, 0)

    tifffile.imwrite(name, img_save,photometric="minisblack")
    return img

def save_images_ch_names(img: np.array, name:str, ch_last:bool = True, channel_names:list=None)-> np.array:
    """
    Save a numpy array to a TIFF file with channel names as metadata.

    This function allows saving a numpy array as a TIFF file with optional channel names included as metadata.

    Parameters
    ----------
    img : np.array
        The input image as a NumPy array.
    name : str
        The file path to save the image.
    ch_last : bool, optional
        Specifies whether the channels are the last axis of the numpy array.
        If True, the array shape is assumed to be (height, width, channels).
        If False, the array shape is assumed to be (channels, height, width).
        Default is True.
    channel_names : list, optional
        A list containing channel names for each channel of the image.
        If provided, each channel name will be included as metadata in the saved TIFF file.
        Default is None.

    Returns
    -------
    np.array
        The input image numpy array.

    Notes
    -----
    - The function saves the image as a TIFF file using the tifffile library.
    - It converts the input array to float32 before saving to ensure compatibility.
    """

    img_save = np.float32(img)
    if ch_last == True: # channel is the last axis
        img_save = np.moveaxis(img_save, -1, 0)

    tifffile.imwrite(name, img_save,photometric="minisblack", metadata={'Channel': {'Name': channel_names}})
    return img

def save_img_ch_names_pages(img: np.array, name:str, ch_last:bool = True, channel_names:list=None)-> np.array:
    """
    Save a numpy array as a TIFF file with channel names in the tags of TIFF file pages.

    This function allows saving a numpy array as a multi-page TIFF file. Each page of the TIFF file corresponds
    to a channel of the input image, and the function provides an option to specify channel names for each page.

    Parameters
    ----------
    img : np.array
        The input image as a NumPy array.
    name : str
        The file path to save the image.
    ch_last : bool, optional
        Specifies whether the channels are the last axis of the numpy array.
        If True, the array shape is assumed to be (height, width, channels).
        If False, the array shape is assumed to be (channels, height, width).
        Default is True.
    channel_names : list, optional
        A list containing channel names for each channel of the image.
        If provided, each TIFF page will have a corresponding channel name in the tags.
        Default is None.

    Returns
    -------
    np.array
        The input image numpy array.

    Notes
    -----
    - The function saves the image as a multi-page TIFF file using the tifffile library.
    - It converts the input array to float32 before saving to ensure compatibility.
    - Channel names provided in the `channel_names` parameter will be included in the tags of TIFF file pages.
    """
    img_save = np.float32(img)
    if ch_last == True: # channel is the last axis
        img_save = np.moveaxis(img_save, -1, 0) # put the channel on first axis

    with tifffile.TiffWriter(name, bigtiff=True) as tiff:
        for i, page in enumerate(img_save):
            tiff.save(page,description=channel_names[i],
                      extratags = [(285,'s',None,channel_names[i], False)]) #, metadata=tags #  description=channel_names[i],
    return img_save



##########################################################################
#####Other thresholds
import skimage
def th_otsu(img: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to an image based on skimage implementation.

    This function uses Otsu's method to determine the optimal threshold value and
    sets all pixel values below this threshold to 0.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The thresholded image with pixels below the threshold set to 0.
    """
    th = skimage.filters.threshold_otsu(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_isodata(img: np.ndarray) -> np.ndarray:
    """
    Apply Isodata thresholding to an image based on skimage implementation.

    This function uses the Isodata method to determine the optimal threshold value and
    sets all pixel values below this threshold to 0.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The thresholded image with pixels below the threshold set to 0.
    """
    th = skimage.filters.threshold_isodata(img)
    new_img = np.where(img > th, img, 0)
    return new_img


def th_li(img: np.ndarray) -> np.ndarray:
    """
    Apply Li thresholding to an image based on skimage implementation.

    This function uses the Li method to determine the optimal threshold value and
    sets all pixel values below this threshold to 0.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The thresholded image with pixels below the threshold set to 0.
    """
    th = skimage.filters.threshold_li(img)
    new_img = np.where(img > th, img, 0)
    return new_img


def th_yen(img: np.ndarray) -> np.ndarray:
    """
    Apply Yen thresholding to an image based on skimage implementation.

    This function uses the Yen method to determine the optimal threshold value and
    sets all pixel values below this threshold to 0.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The thresholded image with pixels below the threshold set to 0.
    """
    th = skimage.filters.threshold_yen(img)
    new_img = np.where(img > th, img, 0)
    return new_img


def th_triangle(img: np.ndarray) -> np.ndarray:
    """
    Apply triangle thresholding to an image based on skimage implementation.

    This function uses the triangle method to determine the optimal threshold value and
    sets all pixel values below this threshold to 0.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The thresholded image with pixels below the threshold set to 0.
    """
    th = skimage.filters.threshold_triangle(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_mean(img: np.ndarray) -> np.ndarray:
    """
    Apply mean thresholding to an image based on skimage implementation..

    This function uses the mean method to determine the threshold value and
    sets all pixel values below this threshold to 0.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The thresholded image with pixels below the threshold set to 0.
    """
    th = skimage.filters.threshold_mean(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_local(img: np.ndarray, block_size: int = 3, method: str = 'gaussian') -> np.ndarray:
    """
    Apply local thresholding to an image based on local
    neighborhoods defined by the block size.

    Parameters
    ----------
    img : np.ndarray
        The input image as a NumPy array.
    block_size : int, optional
        The size of the local neighborhood for threshold calculation.
        Must be an odd number. Default is 3.
    method : str, optional
        The method used for local thresholding. Options include 'gaussian',
        'mean', 'median'. Default is 'gaussian'.

    Returns
    -------
    np.ndarray
        The thresholded image with pixels below their local threshold set to 0.

    Examples
    --------
    >>> img = np.random.rand(100, 100)
    >>> thresholded_img = th_local(img, block_size=5, method='mean')
    >>> print(thresholded_img.min(), thresholded_img.max())

    Notes
    -----
    Local thresholding is useful for images with varying lighting conditions,
    as it adjusts the threshold dynamically across the image.
    """
    th = skimage.filters.threshold_local(img, block_size=block_size, method=method)
    new_img = np.where(img > th, img, 0)
    return new_img

###########################################################################
##### Other Filters


def x_shaped_kernel(size):
    # kernel = [[0] * size for _ in range(size)]
    # kernel = np.zeros((size, size,1))
    kernel = np.zeros((size, size)) # withfor loop

    for i in range(size):
        kernel[i][i] = 1
        kernel[i][size - 1 - i] = 1
    return kernel

def plus_shaped_kernel(size):
    # kernel = [[False] * size for _ in range(size)]
    # kernel = np.zeros((size, size,1))
    kernel = np.zeros((size, size)) # withfor loop

    for i in range(size):
        kernel[size//2][i] = True
        kernel[i][size//2] = True
    return kernel

def center_pixel_kernel(size):
    # kernel = [[False] * size for _ in range(size)]
    # kernel = np.zeros((size, size,1))
    kernel = np.zeros((size, size)) # withfor loop

    kernel[size//2][size//2] = True
    return kernel

def hybrid_median_filter(img: np.array, window_size:int = 3, percentile:int=50, transf_bool = True )-> np.array:
    """
    Apply a hybrid median filter to the image.

    This function implements a hybrid median filter, which combines the results of percentile filtering
    using different shaped kernels (cross, plus, and center pixel) to reduce noise while preserving edges.

    Parameters
    ----------
    img : np.array
        The input image as a NumPy array.
    window_size : int, optional
        The size of the kernel window. Default is 3.
    percentile : int, optional
        The percentile value to be applied to the array inside each kernel.
    transf_bool : bool, optional
        If True, the function will transform the image to boolean values (0 or 1) before filtering.
        This helps identify noise and set those values to zero to prevent unnecessary blur.
        Default is True.

    Returns
    -------
    np.array
        The image filtered with the hybrid median filter.

    Notes
    -----
    - The hybrid median filter applies percentile filtering using three different shaped kernels:
      cross, plus, and center pixel.
    - Each kernel is applied to the input image separately, and the results are combined to form a stack of filtered images.
    - The final filtered image is obtained by computing the percentile of the stack along the channel axis.
    """

    from scipy.ndimage import percentile_filter

    if transf_bool:
        img_to_apply = np.where(img > 0, 1, 0)
    else:
        img_to_apply = copy.deepcopy(img)

    kernel_cross = x_shaped_kernel(window_size)
    kernel_plus = plus_shaped_kernel(window_size)
    kernel_center = center_pixel_kernel(window_size)

    median_stack = np.empty(img_to_apply.shape)
    for ch in range(img.shape[2]):
        # get median of kernel_cross and + shape
        img_med = img_to_apply[:,:,ch]
        median_cross = percentile_filter(img_med,percentile=percentile, footprint=kernel_cross)
        median_plus = percentile_filter(img_med,percentile=percentile, footprint=kernel_plus)
        median_pixel = percentile_filter(img_med,percentile=percentile, footprint=kernel_center)

        img_stack = np.dstack((median_cross, median_plus, median_pixel))

        hybrid_median = np.percentile(img_stack, q=percentile, axis=-1)
        median_stack[:,:,ch] = hybrid_median

    if transf_bool:
        bl = median_stack.astype(bool) # False is Zero
        median_stack = np.where(bl == False, 0, img)
    return median_stack

def filter_hot_pixelsBodenmiller(img: np.ndarray, thres: float) -> np.ndarray:
    '''
    function that implements hot pixel removal from Steinbock
    https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10

    :param img:
    :param thres:
    :return:

    '''
    from scipy.ndimage import maximum_filter
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    # array([[[ True,  True,  True],
    #         [ True, False,  True],
    #         [ True,  True,  True]]])
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)

def modified_hot_pixelsBodenmiller(img: np.ndarray, thres: float, window_size:int = 3) -> np.ndarray:
    """
    function that implements and slightly modified hot pixel removal from Steinbock
    https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10
    # changed for channels last and accept window
    # https://bodenmillergroup.github.io/steinbock/latest/cli/preprocessing/
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10


    :param img:
    :param thres:
    :param window_size:
    :return:
    """
    from scipy.ndimage import maximum_filter
    kernel = np.ones((window_size, window_size,1), dtype=bool)
    line = window_size//2
    kernel[line, line,0] = False  # cneter pixel
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img,0 ) #img



def gaussian_filter(img: np.array, sigma = 0.2)-> np.array:
    """
    Gaussian filter
    :param img:
    :param sigma:
    :return:
    """
    from scipy.ndimage import gaussian_filter
    # in skimage and scipy is defined by a sigmavalue . in open CVis a kernel window
    # open CV may be faster
    # gaussian_blur = cv.GaussianBlur(img,(5,5),0))  # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

    gaus = np.empty(img.shape)
    for ch in range(img.shape[2]):
        gaus_ch = img[:,:,ch]
        denoise_gaus = gaussian_filter(gaus_ch, sigma=sigma)
        gaus[:,:,ch] = denoise_gaus
    return gaus


def non_local_means_filter(img: np.ndarray, patch_size:int = 5, patch_distance: int = 11,
                           sigma:float = 0.2) -> np.ndarray:
    """
    Non local means filter
    :param img:
    :param patch_size:
    :param patch_distance:
    :param sigma:
    :return:
    """
    # they can accept by channel but maybe, it is more suited to do by channel?
    # https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html
    from skimage.restoration import denoise_nl_means, estimate_sigma
    # estimate the noise standard deviation from the noisy image
    # sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
    # patch_kw = dict(patch_size=5,      # 5x5 patches 7 by default
    #                 patch_distance=6,  # 13x13 search area 11 by distance
    #                 channel_axis=-1)
    # # If win_size is not specified, it is calculated as max(5, 2 * ceil(3 * sigma_spatial) + 1).
    #
    # denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
    #                        **patch_kw)
    nlm = np.empty(img.shape)
    patch_kw = dict(patch_size=patch_size,      # 5 - 5x5 patches    7 by default
                    patch_distance=patch_distance  #6 -  13x13 search area 11 by distance
                    )
    if sigma is None:
        sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
        h = 1.15 * sigma_est
    else:
        h = sigma
    for ch in range(img.shape[2]):
        nlm_ch = img[:,:,ch]
        denoise_nlm = denoise_nl_means(nlm_ch, h=h, fast_mode=True,
                           **patch_kw)
        nlm[:,:,ch] = denoise_nlm
    return nlm



def bilateral_filter(img: np.array, **params)-> np.array:
    """
    Bilateral filter
    :param img:
    :param sigma:
    :return:
    """
    # is slow
    from skimage.restoration import denoise_bilateral
    bil = np.empty(img.shape)
    for ch in range(img.shape[2]):
        bil_ch = img[:,:,ch]
        denoise_bil = denoise_bilateral(bil_ch, **params)
        bil[:,:,ch] = denoise_bil
    return bil


def total_variation_filter(img: np.array, weight:float = 0.3,**params)-> np.array:
    """
    Total variation filter
    :param img:
    :param weight:
    :param params:
    :return:
    """
    from skimage.restoration import denoise_tv_chambolle
    # check ifmultichannel needs to be like bilateral   Apply total-variation denoising separately for each channel.
    denoise_TV = denoise_tv_chambolle(img, weight=weight, multichannel=True, **params)
    return denoise_TV


def wavelet_filter(img: np.array)-> np.array:
    """
    Wavelet filtering
    :param img:
    :return:
    """
    from skimage.restoration import denoise_wavelet
    wavelet = np.empty(img.shape)
    for ch in range(img.shape[2]):
        img_ch = img[:,:,ch]
        wav = denoise_wavelet(img_ch, multichannel=False,method='BayesShrink',
                              mode='soft',rescale_sigma=True)
        wavelet[:,:,ch] = wav
    # multichannel should be ok
    # sigma if None is the standard deviation
    return wavelet








def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
    import scipy.ndimage.filters as flt
    import warnings
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    
    copied from https://www.kaggle.com/code/kmader/anisotropic-diffusion-example/notebook
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma);
            deltaEf=flt.gaussian_filter(deltaE,sigma);
        else:
            deltaSf=deltaS;
            deltaEf=deltaE;

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
            gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
        # sleep(0.01)

    return imgout





def anisotropic_filtering(img: np.array, niter:int = 1, kappa: int = 50,
                          gamma: float = 0.2, option:int = 1)-> np.array:
    """

    :param img:
    :param niter:
    :param kappa:
    :param gamma:
    :param option:
    :return:
    """
    # from medpy.filter.smoothing import anisotropic_diffusion
    # https://loli.github.io/medpy/generated/medpy.filter.smoothing.anisotropic_diffusion.html
    # niter= number of iterations
    #kappa = Conduction coefficient (20 to 100)
    #gamma = speed of diffusion (<=0.25)
    #Option: Perona Malik equation 1 or 2. A value of 3 is for Turkey's biweight function
    # Equation 1 favours high contrast edges over low contrast ones, while equation 2 favours wide regions over smaller ones. See [R9] for details. Equation 3 preserves sharper boundaries than previous formulations and improves the automatic stopping of the diffusion.
    af = np.empty(img.shape)
    for ch in range(img.shape[2]):
        af_ch = img[:,:,ch]

        # img_aniso_filtered = anisotropic_diffusion(af_ch, niter=niter, kappa=kappa, gamma=gamma, option=option)
        img_aniso_filtered = anisodiff(af_ch,niter=niter,kappa=kappa,gamma=gamma,step=(1.,1.),sigma=0, option=option,ploton=False)
        af[:,:,ch] = img_aniso_filtered
    return af







def bm3d_filter(img: np.array, sigma_psd:float = 0.2)-> np.array:
    """
    Bm3d filter
    :param img:
    :param sigma_psd:
    :return:
    """
    import bm3d
    if sigma_psd is None:
        from skimage.restoration import estimate_sigma
        sigma_psd = np.mean(estimate_sigma(img, multichannel=True))
        print(sigma_psd)

    # for each marker in image
    img_bm3d = np.empty(img.shape)
    for ch in range(img.shape[2]):
        bm3d_ch = img[:,:,ch]
        BM3D_denoised_image = bm3d.bm3d(bm3d_ch, sigma_psd= sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES) # more slow but more powerful
        #BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        img_bm3d[:,:,ch] = BM3D_denoised_image
    return img_bm3d









#
# def percentile_filter_changedpercentiles(img: np.array, window_size:int = 3, percentiles:list=[], transf_bool = True )-> np.array:
#     '''
#
#     :param img:
#     :param window_size:
#     :param percentiles:
#     :param transf_bool:
#     :return:
#     Percentile different from channel
#
#     '''
#     from scipy.ndimage import percentile_filter
#     kernel = np.ones((window_size, window_size))
#     # todo check if disk is better
#     # from skimage.morphology import disk
#     # kernel = disk(window_size) # disk(3) will create a window 5 *5 plus points on edges
#
#     # make sure that the percentiles input is a list of n channel elements
#     if len(percentiles)!= img.shape[-1]:
#         raise ValueError(f"Percentiles must have the same number of elements as "
#                          f"the number of channels in the image, "
#                          f"expected {img.shape[-1]} but got {len(percentiles)}")
#     if transf_bool:
#         img_to_apply = np.where(img > 0, 1, 0)
#     else:
#         img_to_apply = copy.deepcopy(img)
#
#     nzero = np.count_nonzero(img_to_apply)
#
#     percentile_blur = np.empty(img.shape)
#     for ch in range(img_to_apply.shape[2]):
#         img_ch = img_to_apply[:,:,ch]
#         med = percentile_filter(img_ch,
#                                         percentile=percentiles[ch],
#                                         footprint=kernel)
#         percentile_blur[:,:,ch] = med
#
#     nzero_filter = np.count_nonzero(percentile_blur)
#
#     if transf_bool:
#         bl = percentile_blur.astype(bool) # False is Zero
#         percentile_blur = np.where(bl == False, 0, img)
#         # or
#         # data[~bl] = 0
#     # print(np.unique(percentile_blur))
#     # # get the number of pixel changed. Can be that they are not zeros. or pixels changed that become non zero
#     # pixel_changed = nzero - nzero_filter
#     # total_pixel = img.shape[0] * img.shape[1] * img.shape[2]
#     # percentage_changed = np.round(pixel_changed/total_pixel*100,3)
#     # print('total number of pixel changed: {} of {} \n'
#     #       'percentage of pixels changed: {}'.format(pixel_changed, total_pixel,percentage_changed))
#
#     return percentile_blur

