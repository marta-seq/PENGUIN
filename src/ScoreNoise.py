import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

from skimage.metrics import normalized_root_mse as rmse
#
# class NoiseMeasurement:
#     from sklearn.metrics import mean_absolute_error as mae
#     from skimage.metrics import mutual_information_score as mis
#
#     def __init__(self, noisy_image, denoised_image):
#         self.noisy_image = noisy_image
#         self.denoised_image = denoised_image
#
#     def mse(self):
#         # Mean Squared Error
#         return mse(self.noisy_image, self.denoised_image)
#
#     def psnr(self):
#         # Peak Signal-to-Noise Ratio
#         return psnr(self.noisy_image, self.denoised_image)
#
#     def ssim(self):
#         # Structural Similarity Index
#         return ssim(self.noisy_image, self.denoised_image, multichannel=True)
#
#     def mae(self):
#         # Mean Absolute Error
#         return mae(self.noisy_image, self.denoised_image)
#
#     def rmse(self):
#         # Root Mean Square Error
#         return rmse(self.noisy_image, self.denoised_image)
#
#     def mis(self):
#         # Mutual Information Score
#         return mis(self.noisy_image, self.denoised_image)
#
#     def all_metrics(self):
#         metrics = {}
#         metrics['MSE'] = self.mse()
#         metrics['PSNR'] = self.psnr()
#         metrics['SSIM'] = self.ssim()
#         metrics['MAE'] = self.mae()
#         metrics['RMSE'] = self.rmse()
#         metrics['MIS'] = self.mis()
#         return metrics
#
#     def calculate_psnr_snr(self, image1, image2):
#         # Assumes image1 and image2 are numpy arrays with the same shape and dtype
#         mse = np.mean((image1 - image2) ** 2, axis=(0, 1))
#         snr = np.mean(image1 ** 2, axis=(0, 1)) / mse
#         psnr = 10 * np.log10(np.amax(image1) ** 2 / mse)
#         return psnr, snr


# cehck this
# https://scikit-image.org/docs/stable/api/skimage.metrics.html?highlight=signal%20noise
# should i do this by channel?

def calculate_psnr_snr_save(image1_true, image2_test, save_file):
    # Assumes image1 and image2 are numpy arrays with the same shape and dtype
    # do this by channel
    mse = np.mean((image1_true - image2_test) ** 2, axis=(0, 1))
    snr = np.mean(image1_true ** 2, axis=(0, 1)) / mse
    psnr = 10 * np.log10(np.amax(image1_true) ** 2 / mse)

    with open(save_file, 'w') as fp:
        fp.write('\n'.join([str(psnr.round(4))]))
    return psnr
