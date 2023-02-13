
####### IMC denoise
from IMC_Denoise.IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
# https://github.com/PENGLU-WashU/IMC_Denoise/blob/main/Jupyter_Notebook_examples/IMC_Denoise_Train_and_Predict.ipynb
n_neighbours = 4 # Larger n enables removing more consecutive hot pixels.
n_iter = 3 # Iteration number for DIMR
window_size = 3 # Slide window size. For IMC images, window_size = 3 is fine.

# for each marker in image
img_dimr = np.empty(img_arr.shape)
for ch in range(img_arr.shape[2]):
    Img_raw = img_arr[:,:,ch]
    print(Img_raw)
    Img_DIMR = DIMR(n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size).perform_DIMR(Img_raw)
    img_dimr[:,:,ch] = Img_DIMR
    print(img_dimr)
    print(Img_DIMR)
img_dimr = np.float32(img_dimr)

denoise_img2_t = np.moveaxis(img_dimr, -1, 0)
tifffile.imwrite('img_dimr_nei4_it3.tiff', denoise_img2_t,
                 photometric="minisblack")
# para treinar preciso de mais patches e n sei se quero xb
# n ha trained para todos os markers

# from IMC_Denoise.IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
#
# train_epoches = 50 # training epoches, which should be about 200 for a good training result. The default is 200.
# train_initial_lr = 1e-3 # inital learning rate. The default is 1e-3.
# train_batch_size = 128 # training batch size. For a GPU with smaller memory, it can be tuned smaller. The default is 256.
# pixel_mask_percent = 0.2 # percentage of the masked pixels in each patch. The default is 0.2.
# val_set_percent = 0.15 # percentage of validation set. The default is 0.15.
# loss_function = "I_divergence" # loss function used. The default is "I_divergence".
# weights_name = None # trained network weights saved here. If None, the weights will not be saved.
# loss_name = None # training and validation losses saved here, either .mat or .npz format. If not defined, the losses will not be saved.
# weights_save_directory = None # location where 'weights_name' and 'loss_name' saved.
# # If the value is None, the files will be saved in a sub-directory named "trained_weights" of  the current file folder.
# is_load_weights = False # Use the trained model directly. Will not read from saved one.
# lambda_HF = 3e-6 # HF regularization parameter
# deepsnif = DeepSNiF(train_epoches = train_epoches,
#                     train_learning_rate = train_initial_lr,
#                     train_batch_size = train_batch_size,
#                     mask_perc_pix = pixel_mask_percent,
#                     val_perc = val_set_percent,
#                     loss_func = loss_function,
#                     weights_name = weights_name,
#                     loss_name = loss_name,
#                     weights_dir = weights_save_directory,
#                     is_load_weights = is_load_weights,
#                     lambda_HF = lambda_HF)
# train_loss, val_loss = deepsnif.train(generated_patches)
#
#
# # perform DIMR and DeepSNiF algorithms for low SNR raw images.
# Img_DIMR_DeepSNiF = deepsnif.perform_IMC_Denoise(mg_raw, n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size)
# plt.figure(figsize = (10,8))
# plt.imshow(Img_DIMR_DeepSNiF, vmin = 0, vmax = 0.5*np.max(Img_DIMR_DeepSNiF), cmap = 'jet')
# plt.colorbar()
# plt.show()
