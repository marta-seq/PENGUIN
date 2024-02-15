import os
import re
import math
import tifffile
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import scoreatpercentile
from apeer_ometiff_library import io
from pathlib import Path
from IMC_Denoise_package.IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise_package.IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
from IMC_Denoise_package.IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator
sys.path.append("..")
import src.ImageParser as IP

channels = ['CD20', 'Vimentin', 'PD-L1', 'CD31', 'CD163', 'VISTA', 'Ki-67', 'DNA2', 'IDO',
            'FOXP3', 'CD68', 'CD57', 'CD14', 'D2-40', 'CD56', 'CD45RO', 'DNA1', 'CD11c', 'CD7',
            'HLA-DR', 'CD204', 'CD8a', 'P16Ink4a', 'CD3', 'Granzyme B', 'Bcatenin',
            'Caspase', 'CD4', 'CD103', 'TGFbeta', 'PD-1', 'CD45', 'LAG-3',
            'ICOS', 'CD11b', 'Keratin', 'TCRgd', 'CD15', 'TIM-3', 'CD38', 'Tbet', 'CD39']
channel_to_index = {name: index for index, name in enumerate(channels)}


def get_files_from_dir(fov):
    fov_folder = os.path.join(path_images_raw, fov)
    fov_files = os.listdir(fov_folder)
    fov_files = [filename for filename in fov_files if filename.lower().endswith((".tiff", ".tif"))]
    fov_files = [os.path.join(fov_folder, file) for file in fov_files]
    return fov_files


def get_stack(fov_files):
    # get a stack of images
    stack = []
    for channel in channels:
        file = [filename for filename in fov_files if channel in filename][0]
        img_apeer, _ = io.read_ometiff(file)
        img = img_apeer.squeeze()
        stack.append(np.array(img))
    stack = np.stack(stack, axis=-1)
    return stack


def plot_comparison(image1, image2, channel_to_compare):
    plt.figure(figsize=(10, 5), dpi=500)
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')  # ,vmin = 0, vmax = 1
    plt.title(channel_to_compare)

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')  # , vmin = 0, vmax = 1
    plt.title(channel_to_compare)

    plt.show()


def save_stack(img, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(img.shape[2]):
        slice = img[:, :, i]
        channel = channels[i]
        filename = os.path.join(output_dir, f"{channel}.tiff")  # .png

        # Save the image
        tifffile.imwrite(filename, np.float32(slice),
                         photometric="minisblack")


#         plt.imshow(np.float32(slice), cmap='gray')  # Assuming grayscale image, adjust cmap if needed
#         plt.axis('off')  # Turn off axes

#         # Save the image as a PNG file
#         plt.savefig(os.path.join(output_dir, f"{channel}.png"), bbox_inches='tight', pad_inches=0,dpi = 700)
#         plt.close()


# functions for neigbooring and counting pixels analysis
def get_counts_neigh(image):
    # Threshold the image to get binary values
    binary_image = (image > 0).astype(np.uint8)
    positive_pixel_count = np.count_nonzero(binary_image)
    #     print(positive_pixel_count)
    # Find the coordinates of positive (non-zero) pixels in the binary image
    positive_pixel_coords = np.argwhere(binary_image == 1)

    # Initialize lists to store results
    positive_counts = []
    medians = []
    percentile_25 = []
    percentile_75 = []
    # Iterate through positive pixels
    for coord in positive_pixel_coords:
        y, x = coord[0], coord[1]

        # Extract the 3x3 window around the current positive pixel
        # Define the coordinates for the 3x3 window
        y_start, y_end = max(y - 1, 0), min(y + 2, binary_image.shape[0])
        x_start, x_end = max(x - 1, 0), min(x + 2, binary_image.shape[1])

        # Extract the 3x3 window around the current positive pixel
        window = binary_image[y_start:y_end, x_start:x_end]

        # Count positive pixels in the 3x3 window
        positive_count = np.count_nonzero(window)
        positive_counts.append(positive_count)

        # Calculate the median and percentiles

        window = image[y_start:y_end, x_start:x_end]
        median = np.median(window)
        p_25 = scoreatpercentile(window, 25)
        p_75 = scoreatpercentile(window, 75)

        # Append the statistics to the respective lists
        medians.append(median)
        percentile_25.append(p_25)
        percentile_75.append(p_75)
    return positive_counts, medians, percentile_25, percentile_75


def plot_hist_positive_neig(positive_counts1, positive_counts2):
    # Create a figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first histogram on the left subplot
    axes[0].hist(positive_counts1, bins=range(max(positive_counts1) + 2), rwidth=0.8, align='left')
    axes[0].set_xlabel('Number of neighboring Positive Pixels of positive pixels')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram 1')

    # Plot the second histogram on the right subplot
    axes[1].hist(positive_counts2, bins=range(max(positive_counts2) + 2), rwidth=0.8, align='left')
    axes[1].set_xlabel('Number of neighboring Positive Pixels of positive pixels')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram 2')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_percentiles(list1, list2):
    # Create two separate figures for the first and second graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the distribution of medians and percentiles for the first graph
    ax1.hist(list1, bins=10, color=['blue', 'green', 'red'], alpha=0.7,
             label=['Median', 'Percentile 25', 'Percentile 75'])
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Medians and Percentiles ()')
    ax1.legend()
    # Plot the distribution of medians and percentiles for the first graph
    ax2.hist(list2, bins=10, color=['blue', 'green', 'red'], alpha=0.7,
             label=['Median', 'Percentile 25', 'Percentile 75'])
    ax2.set_xlabel('Values')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Medians and Percentiles ()')
    ax1.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_comparison_IMCDenoise(image1, image2, image3, channel_to_compare):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the first image in the first subplot
    im1 = axs[0].imshow(image1, vmin=0, vmax=0.5 * np.max(image1), cmap='jet')
    axs[0].set_title(channel_to_compare + ' RAW')

    im2 = axs[1].imshow(image2, vmin=0, vmax=0.5 * np.max(image1), cmap='jet')
    axs[1].set_title(channel_to_compare + ' DIMR')

    im3 = axs[2].imshow(image3, vmin=0, vmax=0.5 * np.max(image1), cmap='jet')
    axs[2].set_title(channel_to_compare + ' DIMR and DeepSNIF')

    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar3 = fig.colorbar(im3, ax=axs[2])

    plt.tight_layout()
    plt.show()


# the abve toook more than 17 hours. As the 200 epochs do not improve the loss comparing to 50. I wil run with 50
# # train DEEPSNIF
channel_name = "Bcatenin"
Raw_directory = "../data/IMC_ESD/raw"

n_neighbours = 4  # Larger n enables removing more consecutive hot pixels.
n_iter = 3  # Iteration number for DIMR
window_size = 3  # Slide window size. For IMC images, window_size = 3 is fine.

DataGenerator = DeepSNiF_DataGenerator(channel_name=channel_name, n_neighbours=n_neighbours, n_iter=n_iter,
                                       window_size=window_size)
generated_patches = DataGenerator.generate_patches_from_directory(load_directory=Raw_directory)
print('The shape of the generated training set is ' + str(generated_patches.shape) + '.')

train_epoches = 200  # training epoches, which should be about 200 for a good training result. The default is 200.
train_initial_lr = 1e-3  # inital learning rate. The default is 1e-3.
train_batch_size = 128  # training batch size. For a GPU with smaller memory, it can be tuned smaller. The default is 256.
pixel_mask_percent = 0.2  # percentage of the masked pixels in each patch. The default is 0.2.
val_set_percent = 0.15  # percentage of validation set. The default is 0.15.
loss_function = "I_divergence"  # loss function used. The default is "I_divergence".
weights_name = None  # trained network weights saved here. If None, the weights will not be saved.
loss_name = None  # training and validation losses saved here, either .mat or .npz format. If not defined, the losses will not be saved.
weights_save_directory = None  # location where 'weights_name' and 'loss_name' saved.
# If the value is None, the files will be saved in a sub-directory named "trained_weights" of  the current file folder.
is_load_weights = False  # Use the trained model directly. Will not read from saved one.
lambda_HF = 3e-6  # HF regularization parameter
deepsnif = DeepSNiF(train_epoches=train_epoches,
                    train_learning_rate=train_initial_lr,
                    train_batch_size=train_batch_size,
                    mask_perc_pix=pixel_mask_percent,
                    val_perc=val_set_percent,
                    loss_func=loss_function,
                    weights_name=weights_name,
                    loss_name=loss_name,
                    weights_dir=weights_save_directory,
                    is_load_weights=is_load_weights,
                    lambda_HF=lambda_HF)
train_loss, val_loss = deepsnif.train(generated_patches)

dimr_dir = 'img_results/IMC_specific/DIMR_bcat_200ep'
deepsnif_dir = 'img_results/IMC_specific/DeepSNIF_bcat_200ep'
if not os.path.exists(dimr_dir):
    os.makedirs(dimr_dir)
if not os.path.exists(deepsnif_dir):
    os.makedirs(deepsnif_dir)

bcat_files = [file for file in Path(Raw_directory).rglob(f'*{channel_name}*')]
bcat_raw = []
bcat_DIMR = []
bcat_DeepSNIF = []

for file in bcat_files:
    filename = Path(file).name
    filename_dimr = os.path.join(dimr_dir, filename)
    filename_deepsnif = os.path.join(deepsnif_dir, filename)

    # read_img
    img_apeer, _ = io.read_ometiff(file)
    img = img_apeer.squeeze()
    bcat_raw.append(img)

    # perform DIMR with the model trained
    dmr = DIMR(n_neighbours=4, n_iter=3, window_size=5)
    img_dimr = dmr.perform_DIMR(img)
    bcat_DIMR.append(img_dimr)

    tifffile.imwrite(filename_dimr, np.float32(img_dimr), photometric="minisblack")

    # DeepSNIF
    Img_DIMR_DeepSNiF = deepsnif.perform_IMC_Denoise(img, n_neighbours=n_neighbours, n_iter=n_iter,
                                                     window_size=window_size)
    bcat_DeepSNIF.append(Img_DIMR_DeepSNiF)

    tifffile.imwrite(filename_deepsnif, np.float32(Img_DIMR_DeepSNiF), photometric="minisblack")
