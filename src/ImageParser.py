from apeer_ometiff_library import io
# import tensorflow as tf
import numpy as np
from src.file_specs import FileSpecifics
from tifffile import tifffile
# class ImageParser():

# def parse_image(img_path: str) -> dict:
def parse_image(img_path: str) -> np.array:
    """
    Load an image from the specified path, process it, and return it as a NumPy array.

    The function reads a TIFF image, squeezes the array to remove single-dimensional entries,
    and moves the axis to ensure the array has dimensions (X, Y, channels).

    Parameters
    ----------
    img_path : str
        Path to the image file (not the mask).

    Returns
    -------
    np.ndarray or None
        A NumPy array representing the image with shape (X, Y, channels).
        Returns None if there is an error reading the image.

    Notes
    -----
    This function is designed to handle OME-TIFF images specifically. If the image cannot be read,
    the function will print an error message and return None.

    Examples
    --------
    >>> img_array = parse_image('path/to/image.ome.tiff')
    >>> if img_array is not None:
    >>>     print("Image shape:", img_array.shape)

    """
    try:
        (img_apeer, omexml) = io.read_ometiff(img_path)
        img = np.squeeze(img_apeer)
        img = np.moveaxis(img, 0, -1)

            # return {'image': img, 'img_meta': omexml, 'filename': img_path}
        return img
    except Exception as e:
        print('did not read image', img_path)
        print(e)
        return None


def parse_image_pages(img_path: str) -> np.array:
    """
    Load a TIFF file with multiple pages and return it as a NumPy array.

    This function reads a multi-page TIFF file, extracts each page as an image,
    and combines them into a single NumPy array with dimensions (X, Y, channels).

    Parameters
    ----------
    img_path : str
        Path to the TIFF image file.

    Returns
    -------
    np.ndarray
        A NumPy array representing the image with shape (X, Y, channels).

    Examples
    --------
    >>> img_array = parse_image_pages('path/to/image.tiff')
    >>> print("Image shape:", img_array.shape)

    """
    im = []
    with tifffile.TiffFile(img_path) as tif:
        for page in tif.pages:
            image = page.asarray()
            im.append(image)
    im = np.array(im)
    img = np.moveaxis(im, 0, -1)
    return img

def parse_image_pages_namesCH(img_path):
    """
    Load a TIFF file with multiple pages and retrieve the names of the pages.

    This function reads a multi-page TIFF file, extracts each page as an image,
    and retrieves the names of the channels from the page tags. The images are
    combined into a single NumPy array with dimensions (X, Y, channels).

    Parameters
    ----------
    img_path : str
        Path to the TIFF image file.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A tuple containing:
        - A NumPy array representing the image with shape (X, Y, channels).
        - A list of strings representing the names of the channels.

    Notes
    -----
    The function looks for 'PageName' or 'ImageDescription' tags in each page
    to determine the channel names. If neither tag is found, the channel name
    is not added to the list.

    Examples
    --------
    >>> img_array, channel_names = parse_image_pages_namesCH('path/to/image.tiff')
    >>> if img_array is not None:
    >>>     print("Image shape:", img_array.shape)
    >>>     print("Channel names:", channel_names)

    """
    im = []
    ch_names = []
    with tifffile.TiffFile(img_path) as tif:
        for page in tif.pages:
            image = page.asarray()
            im.append(image)
            if 'PageName' in page.tags:
                ch_names.append(page.tags['PageName'].value)
            elif 'ImageDescription' in page.tags:
                ch_names.append(page.tags['ImageDescription'].value)
            else:
                pass

    im = np.array(im)
    img = np.moveaxis(im, 0, -1)
    return img, ch_names

def parse_image_with_meta(img_path: str) -> dict:
    """
    Load an image and its annotation and returns
    a dictionary with image and metadata

    Parameters
    ----------
    CHANNELS
    img_path : str
        Image location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    try:
        (img_apeer, omexml) = io.read_ometiff(img_path)
        img = np.squeeze(img_apeer)
        img = np.moveaxis(img, 0, -1)

            # return {'image': img, 'img_meta': omexml, 'filename': img_path}
        return img
    except Exception as e:
        print('did not read image', img_path)
        print(e)

    return {'img_meta': omexml, 'filename': img_path, 'shape_img': img.shape, 'img':img}

