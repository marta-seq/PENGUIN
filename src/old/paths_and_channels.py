import os

######
# 1. DATA FILES
######
channelsBodenmiller2020 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 32, 33, 34, 37, 38, 40, 41,
                           42, 43, 44, 45, 46, 47, 48]
channels_would_be_good = [25, 29, 31, 35, 36, 39]

channelsBodenmiller2020_V2 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29,30,31, 32, 33, 34, 35, 36,37, 38, 39, 40, 41,
                              42, 43, 44, 45, 46, 47, 48] # join both lists

channels_METABRIC2020 = [1,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
    ,37,38,39,40,41,42,43]

channels_METABRIC2022 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]

data_dir = 'breast_cancer_imc/breast_data/'

data = {
    'Bodenmiller2020_Basel':
        {'path_meta':os.path.join(data_dir, 'bodenmiller2020/Basel_PatientMetadataALTERED.csv'),
         'path_images':os.path.join(data_dir, 'bodenmiller2020/OMEnMasks/ome/'),
         'channels': channelsBodenmiller2020_V2}, # use of all chennels even the ones with bad wuality

    'Bodenmiller2020_Zur':
        {'path_meta':os.path.join(data_dir, 'bodenmiller2020/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv'),
         'path_images':os.path.join(data_dir, 'bodenmiller2020/OMEnMasks/ome/'),
         'channels':channelsBodenmiller2020_V2},

    'Metabric2020':
        {'path_meta':os.path.join(data_dir, ''),
         'path_images':os.path.join(data_dir, 'imc_metabric/to_public_repository/full_stacks/'),
         'channels': channels_METABRIC2020},

    'Metabric2022':
        {'path_meta':os.path.join(data_dir, ''),
         'path_images':os.path.join(data_dir, 'imc_metabric2022/MBTMEIMCPublic/Images/'),
         'channels': channels_METABRIC2022},
}

