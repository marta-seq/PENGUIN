"""
Title-->          HHM
Author-->         Ana Doblas, Raul Castaneda,
Date-->           03/08/2020
Last modified-->  24/01/2020
                  University of Memphis
                  Department of Electrical and Computer Engineering
                  Optical Imaging Research lab (OIRL)
                  Memphis, TN 38152, USA.
                  version 1.0 (2020)
Email-->          adoblas@memphis.
Abstract -->      Algorithm that allow to reduce the speckle noisy although the use of a hybrid median-mean method


Citations -->     If using this code for publishing your results, please kindly cite us:
                  R. Castaneda, J. Garcia-Sucerquia and A. Doblas, “Speckle Noise Reduction in Digital Holography via a
                  Hybrid Median-Mean Approach,” Appl. Opt. XX, XX, p.p xxxx–xxxx

Libraries-->      HMM use numpy, cv2, scipy and matplotlib as libraries.
Running -->       For running HMM.py the script must be inside the folder project.
                  The implementation you must call the method  HybridMedianMean as
                  HMM.HybridMedianMean(image, max_kernel_size, figures, plots, save_image).
                  Once the function is running the image to reduce the speckle appears, please select a region of
                  interest to measure the speckle contrast; after select the region click enter.

Inputs-->         image: Corresponds to the noise image.
                  max_kernel_size: The maximun dimension of the kernel, this number must be odd.
                  figures: Allow to show the original/noise image (named as image) and the denoising image after applied
                  the hybrid median-mean approach; Figures has two options: True for displaying both images or False for
                  not displaying.
                  plots: Allow to select a square region to measure/quantify the speckle contrast and plot the speckle
                  contrast vs number of iterations; Plots has two options True or False.
                  save_image: Allow to save the final denoising image after applying the hybrid median-mean method; Save
                  image has two options True or False.

Output-->         speckle denoising image.
                  If the parameter save_image is true, the image with the speckle reduced is saved as "denoising image"
                  in the same folder where the file HMM.py is located
"""

# libraries
import numpy as np
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt


def measure_speckle(sample):
    height, width = sample.shape  # get size of the image
    newimg = cv2.resize(sample, (int(height / 2), int(width / 2)))
    sample = newimg
    coordinates_ROI = cv2.selectROI(newimg, fromCenter=False)
    x1 = int(coordinates_ROI[1])
    y1 = int(coordinates_ROI[0])
    x2 = int(coordinates_ROI[1] + coordinates_ROI[3])
    y2 = int(coordinates_ROI[0] + coordinates_ROI[2])
    roi_mean = sample[x1:x2, y1:y2]
    std_mean = np.std(roi_mean)
    mean_mean = np.mean(roi_mean)
    max_speckle_contrast = std_mean / mean_mean
    return max_speckle_contrast, x1, y1, x2, y2, sample


def speckle_reduction(max_speckle_contrast, x1, y1, x2, y2, sample, max_kernel_size):
    mean_image = sample
    height, width = sample.shape  # get size of the image
    dim_vectors = round(int(max_kernel_size + 1) / 2)
    list_cont = list(range(1, dim_vectors + 1))
    array_cont = np.array(list_cont)
    list_kernel = list(range(1, max_kernel_size + 2, 2))
    array_kernel_size = np.array(list_kernel)
    cont = 1
    array_speckle_contrast = np.zeros(1)
    array_speckle_contrast[0] = max_speckle_contrast
    speckle_contrast = np.zeros([height, width, dim_vectors])
    for i in range(3, max_kernel_size + 2, 2):
        filter = ndimage.median_filter(sample, i, mode='constant', cval=0)
        mean_image = (mean_image + filter) / 2
        speckle_contrast[:, :, cont] = mean_image
        # measure of speckle contrast
        roi_mean = mean_image[x1:x2, y1:y2]
        std_mean = np.std(roi_mean)
        mean_mean = np.mean(roi_mean)
        constrast_speckle_mean = std_mean / mean_mean
        array_speckle_contrast = np.append(array_speckle_contrast, [constrast_speckle_mean])
        cont = cont + 1
    max_contrast = max(array_speckle_contrast)
    array_speckle_contrast = array_speckle_contrast / max_contrast
    return array_cont, array_speckle_contrast, speckle_contrast, dim_vectors, array_kernel_size


def images(sample, speckle_contrast, dim_z):
    plt.subplot(121).set_title("Original image")
    plt.imshow(sample, cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplot(122).set_title("Image denoising [" + str((2*dim_z-1)) + 'x' + str((2*dim_z-1)) + ']')
    plt.imshow(speckle_contrast[:, :, 2], cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
    plt.show()


def plots_true(array_cont, array_speckle_contrast, array_kernel_size):
    theory = 1 / np.sqrt(array_cont)
    max_contrast = max(array_speckle_contrast)
    vec_contras_speckle = array_speckle_contrast / max_contrast
    plt.plot(array_cont, vec_contras_speckle, label="Hybrid median-mean", marker='o', color='green', linestyle='solid')
    plt.plot(array_cont, theory, label="Theory", marker='s', color='red', linestyle='dashed')
    plt.axis([1, array_cont.size, 0, 1])
    plt.xlabel("Number of iterations")
    plt.ylabel("Speckle contrast [a.u.]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.title('Speckle contrast vs number of iterations')
    plt.grid()
    ax2 = plt.twiny()
    ax2.set_xlabel('k value', color='black')
    ax2.set_xticks(array_kernel_size)
    ax2.set_xlim((1, array_kernel_size[-1]))
    plt.show()


def save(sample, speckle_contrast):
    cv2.imwrite('original image.jpg', sample)
    cv2.imwrite('denoising image.jpg', speckle_contrast[:, :, -1])


def HybridMedianMean(sample, max_kernel_size, figures, plots, save_image):
    if max_kernel_size % 2 == 0:
        print('Kernel size must be a odd number')
        exit()
    max_speckle_contrast, x1, y1, x2, y2, sample = measure_speckle(sample)
    array_cont, array_speckle_contrast, speckle_contrast, dim_z, array_kernel_size = speckle_reduction\
        (max_speckle_contrast, x1, y1, x2, y2, sample, max_kernel_size)
    if figures == "True":
        images(sample, speckle_contrast, dim_z)
    elif figures == "False":
        pass
    else:
        print('figure only get two options: True or False')
        exit()
    if plots == "True":
        plots_true(array_cont, array_speckle_contrast, array_kernel_size)
    elif plots == "'False":
        pass
    else:
        print('plots only get two options: True or False')
        exit()
    if save_image == "True":
        save(sample, speckle_contrast)
    elif save_image == "False":
        pass
    else:
        print('save_image only get two options: True or False')
        exit()
