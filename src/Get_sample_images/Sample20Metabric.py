import shutil
import os
import pandas as pd
import GetDataset

###############################################
## 1 Get data
metabric22 = GetDataset.imc_METABRIC2022()
metabric22 = metabric22.loc[:,~metabric22.columns.duplicated()].copy()
metabric22 = metabric22.loc[metabric22['HISTOLOGICAL_SUBTYPE']=='Ductal/NST']
met_long_surv = metabric22.loc[metabric22['OS_MONTHS']>60]
# count grade
# 3.0    189
# 2.0    112
# 1.0     26

met_long_surv_grade3 = met_long_surv.loc[met_long_surv['grade']==3.0].sample(4)
met_long_surv_grade2 = met_long_surv.loc[met_long_surv['grade']==2.0].sample(4)
met_long_surv_grade1 = met_long_surv.loc[met_long_surv['grade']==1.0].sample(4)

met_short_surv = metabric22.loc[metabric22['VITAL_STATUS']=='Died of Disease']
met_short_surv = met_short_surv.loc[met_short_surv['OS_MONTHS']<12]
# count grade
# 3.0    9
# 2.0    2
met_short_surv_grade3 = met_short_surv.loc[met_short_surv['grade']==3.0].sample(4)
met_short_surv_grade2 = met_short_surv.loc[met_short_surv['grade']==2.0]

try_cohort = pd.concat([met_long_surv_grade3, met_long_surv_grade2, met_long_surv_grade1,
                        met_short_surv_grade3, met_short_surv_grade2], ignore_index=True)
path_images = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/' \
              'breast_data/imc_metabric2022/MBTMEIMCPublic/Images/'

dir_to_wr='/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/PreprocessAnalysis/metabric22'
name_csv = os.path.join(dir_to_wr,'metabric22_sampled.csv')
try_cohort.to_csv(name_csv, index=False)

print(metabric22.columns)

image= try_cohort['FileName_FullStack']
images=[img[0] for img in image]
src_path = [os.path.join(path_images, image) for image in images]
dst_path = [os.path.join(dir_to_wr, image) for image in images]

for i in range(len(src_path)):
    shutil.copy(src_path[i], dst_path[i])


# for i in range(len(try_cohort)):
#     path = os.path.join(path_images, try_cohort['FileName_FullStack'][i][0])
#     large_image = IP.parse_image(path)  # read files
#     # print(large_image.shape)
#     imgNoOutlier = IP.remove_outliers(large_image,up_limit=99, down_limit=1)  # remove outliers
#     # plt.imshow(imgNoOutlier[:,:,[22,28,35]])
#     # plt.show()
#     # per image
#     imgNorm = IP.normalize_by_channel(imgNoOutlier)
#     # plt.imshow(imgNorm[:,:,[22,28,35]])
#     # plt.show()
#     thres = 0.1
#     imgDenoise = IP.filter_hot_pixelsBodenmiller(imgNorm, thres)
#     plt.imshow(imgDenoise[:,:,[22,28,35]]) # Cd45 Ki67 panCK
#     plt.show()
#     # print('clustering')
#
#     # x, y, z = imgDenoise.shape
#     # image_2d = imgDenoise.reshape(x*y, z)
#     # kmeans_cluster = cluster.KMeans(n_clusters=7)
#     # kmeans_cluster.fit(image_2d)
#     # cluster_centers = kmeans_cluster.cluster_centers_
#     # cluster_labels = kmeans_cluster.labels_
#     # cluster_img = cluster_centers[cluster_labels].reshape(x, y, z)
#     # plt.imshow(cluster_img[:,:,0])
#     # plt.show()
#
#     # try to get a segmentation based on edges xb
#     imgEdge = imgDenoise[:,:,22]
#     # Compute the Canny filter for two values of sigma
#     # edges = feature.canny(imgEdge)
#     edges = feature.canny(imgEdge, sigma=3)
#     plt.imshow(edges)
#     plt.show()
#
#
# # vectorized = imgDenoise.reshape((-1,3))
# #     vectorized = np.float32(vectorized)
# #     K = 30
# #     attempts=50
# #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# #     ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
# #     center = np.uint8(center)
# #     res = center[label.flatten()]
# #     result_image = res.reshape((imgDenoise.shape[0],imgDenoise.shape[1],K ))
# #     plt.imshow(result_image) # Cd45 Ki67 panCK
# #     plt.show()
# #
# #
#
#
#
