import shutil
import os
import pandas as pd
import GetDataset
from collections import Counter

###############################################
# ## 1 Boden 20 Basel
boden20 = GetDataset.imc_bodenmiller2020(just_tumor=True, just_invasive_ductal=True)
boden20_long_surv = boden20.loc[boden20['OSmonth']>60]
print(Counter(boden20_long_surv['grade']))
# Counter({2: 80, 3: 47, 1: 24})

boden20_long_surv_grade3 = boden20_long_surv.loc[boden20_long_surv['grade']==3.0].sample(4)
boden20_long_surv_grade2 = boden20_long_surv.loc[boden20_long_surv['grade']==2.0].sample(4)
boden20_long_surv_grade1 = boden20_long_surv.loc[boden20_long_surv['grade']==1.0].sample(4)


boden20_short_surv = boden20.loc[boden20['Patientstatus']=='death by primary disease']
boden20_short_surv = boden20_short_surv.loc[boden20_short_surv['OSmonth']<30]
print(Counter(boden20_short_surv['grade']))
# Counter({3: 12, 2: 1})

boden20_short_surv_grade3 = boden20_short_surv.loc[boden20_short_surv['grade']==3.0].sample(4)
boden20_short_surv_grade2 = boden20_short_surv.loc[boden20_short_surv['grade']==2.0]
# boden20_short_surv_grade1 = boden20_short_surv.loc[boden20_short_surv['grade']==1.0].sample(4)

try_cohort_boden20 = pd.concat([boden20_long_surv_grade3, boden20_long_surv_grade2, boden20_long_surv_grade1,
                        boden20_short_surv_grade3, boden20_short_surv_grade2], ignore_index=True)

path_images = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/' \
              'breast_data/bodenmiller2020/OMEnMasks/ome'

dir_to_wr='/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/PreprocessAnalysis/boden20'
name_csv = os.path.join(dir_to_wr,'boden20_sampled.csv')
try_cohort_boden20.to_csv(name_csv, index=False)

images= try_cohort_boden20['FileName_FullStack']
src_path = [os.path.join(path_images, image) for image in images]
dst_path = [os.path.join(dir_to_wr, image) for image in images]

for i in range(len(src_path)):
    shutil.copy(src_path[i], dst_path[i])



###############################################
## 1 Boden 20 Zur

boden20Z = GetDataset.imc_bodenmiller2020_Zur()
print(boden20Z.columns)
# boden20_long_surv = boden20Z.loc[boden20Z['OSmonth']>40]
print(Counter(boden20Z['grade']))
print(Counter(boden20Z['location']))
# Counter({'3': 130, '2': 122, '1': 95, 'METASTASIS': 12})
# Counter({'PERIPHERY': 140, 'CENTER': 136, '[]': 71, 'METASTASIS': 12})


boden20_long_surv_grade3 = boden20Z.loc[boden20Z['grade']=='3']
boden20_long_surv_grade2 = boden20Z.loc[boden20Z['grade']=='2']
boden20_long_surv_grade1 = boden20Z.loc[boden20Z['grade']=='1']

boden20_long_surv_met = boden20Z.loc[boden20Z['location']=='METASTASIS'].sample(4)

print(Counter(boden20_long_surv_grade3['location']))
# Counter({'CENTER': 52, 'PERIPHERY': 52, '[]': 26})
boden20_grade3_pheri = boden20_long_surv_grade3.loc[boden20_long_surv_grade3['location']=='PERIPHERY'].sample(4)
boden20_grade3_center = boden20_long_surv_grade3.loc[boden20_long_surv_grade3['location']=='CENTER'].sample(4)

print(Counter(boden20_long_surv_grade2['location']))
# Counter({'PERIPHERY': 51, 'CENTER': 46, '[]': 25})

boden20_grade2_pheri = boden20_long_surv_grade2.loc[boden20_long_surv_grade2['location']=='PERIPHERY'].sample(4)
boden20_grade2_center = boden20_long_surv_grade2.loc[boden20_long_surv_grade2['location']=='CENTER'].sample(4)

print(Counter(boden20_long_surv_grade1['location']))
# Counter({'CENTER': 38, 'PERIPHERY': 37, '[]': 20})
boden20_grade1_pheri = boden20_long_surv_grade1.loc[boden20_long_surv_grade1['location']=='PERIPHERY'].sample(4)
boden20_grade1_center = boden20_long_surv_grade1.loc[boden20_long_surv_grade1['location']=='CENTER'].sample(4)


try_cohort_boden20Z = pd.concat([boden20_grade1_pheri,boden20_grade1_center,
                                 boden20_grade2_pheri,boden20_grade2_center,
                                 boden20_grade3_pheri,boden20_grade3_center,boden20_long_surv_met], ignore_index=True)

path_images = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/' \
              'breast_data/bodenmiller2020/OMEnMasks/ome'

dir_to_wr='/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/PreprocessAnalysis/boden20'
name_csv = os.path.join(dir_to_wr,'boden20Zur_sampled.csv')
try_cohort_boden20Z.to_csv(name_csv, index=False)

images= try_cohort_boden20Z['FileName_FullStack']
src_path = [os.path.join(path_images, image) for image in images]
dst_path = [os.path.join(dir_to_wr, image) for image in images]

for i in range(len(src_path)):
    shutil.copy(src_path[i], dst_path[i])


#######################################################33
# MEtabric2020
metabric20 = GetDataset.imc_METABRIC2020()
metabric20 = metabric20.loc[:,~metabric20.columns.duplicated()].copy()
metabric20 = metabric20.loc[metabric20['HISTOLOGICAL_SUBTYPE']=='Ductal/NST']
met_long_surv = metabric20.loc[metabric20['OS_MONTHS']>60]
print(Counter(met_long_surv['grade']))
# Counter({3.0: 144, 2.0: 83, 1.0: 22, nan: 1, nan: 1})

met_long_surv_grade3 = met_long_surv.loc[met_long_surv['grade']==3.0].sample(4)
met_long_surv_grade2 = met_long_surv.loc[met_long_surv['grade']==2.0].sample(4)
met_long_surv_grade1 = met_long_surv.loc[met_long_surv['grade']==1.0].sample(4)

met_short_surv = metabric20.loc[metabric20['VITAL_STATUS']=='Died of Disease']
met_short_surv = met_short_surv.loc[met_short_surv['OS_MONTHS']<24]
print(Counter(met_short_surv['grade']))
# Counter({3.0:19, 2.0: 2, 1.0: 1, nan: 1, nan: 1})

met_short_surv_grade3 = met_short_surv.loc[met_short_surv['grade']==3.0].sample(4)
met_short_surv_grade2 = met_short_surv.loc[met_short_surv['grade']==2.0]

try_cohort = pd.concat([met_long_surv_grade3, met_long_surv_grade2, met_long_surv_grade1,
                        met_short_surv_grade3, met_short_surv_grade2], ignore_index=True)

path_images = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/' \
              'imc_metabric/to_public_repository/full_stacks'

dir_to_wr = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/PreprocessAnalysis/metabric20'
name_csv = os.path.join(dir_to_wr,'metabric20_sampled.csv')
try_cohort.to_csv(name_csv, index=False)


images= try_cohort['FileName_FullStack']
# images=[img[0] for img in image]
src_path = [os.path.join(path_images, image) for image in images]
dst_path = [os.path.join(dir_to_wr, image) for image in images]

for i in range(len(src_path)):
    shutil.copy(src_path[i], dst_path[i])

