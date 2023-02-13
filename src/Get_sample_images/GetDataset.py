import os
import pandas as pd
from sklearn.utils import shuffle

# functions fromGetDataset
def metabric_metadata():
    """
    :return:
    """
    data_dir = '/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/imc_metabric/'
    # data_dir='phd/breast_cancer_imc/breast_data/imc_metabric/'
    # data_dir='breast_data/imc_metabric/'

    path_data_meta_metabric = os.path.join(data_dir, 'Complete_METABRIC_Clinical_Features_Data.txt')
    # open file with metadat cohort basel
    metabric = pd.read_csv(path_data_meta_metabric) #[1981 rows x 25 columns] with 1981 unique IDs
    metabric = metabric.reset_index()
    metabric = metabric.rename(columns={'index':'Patient_ID'})
    metabric = metabric.set_index('Patient_ID')

    # second file
    path_data_meta_metabric2 = os.path.join(data_dir,'brca_metabric/data_clinical_patient.txt')
    metabric2 = pd.read_csv(path_data_meta_metabric2,sep='\t', comment='#') # [2509 rows x 24 columns] with 2509 unique
    # some of IDS are MTS-T2428   # 1985 IDS with MB
    metabric2 = metabric2.rename(columns={'PATIENT_ID':'Patient_ID'})
    metabric2 = metabric2.set_index('Patient_ID')

    path_data_meta_metabric3 = os.path.join(data_dir,'brca_metabric/data_clinical_sample.txt')
    metabric3 = pd.read_csv(path_data_meta_metabric3,sep='\t', comment='#') # [2509 rows x 13 columns] with 2509 unique PATIENT AND SAMPLE id
    # tem sample primary and recurrence ... and stuff
    metabric3 = metabric3.rename(columns={'PATIENT_ID':'Patient_ID'})
    metabric3 = metabric3.set_index('Patient_ID')

    path_metabric_survival = os.path.join(data_dir,'KM_Plot__Overall_Survival__(months).txt')
    metabric4 = pd.read_csv(path_metabric_survival,sep='\t', comment='#')
    metabric4 = metabric4.rename(columns={'Patient ID':'Patient_ID'}).reset_index()
    metabric4 = metabric4.set_index('Patient_ID') # [2509 rows x 4 columns]

    path_metabric_survival = os.path.join(data_dir,'KM_Plot__Relapse_Free__Survival_(months).txt')
    metabric5 = pd.read_csv(path_metabric_survival,sep='\t', comment='#')
    metabric5 = metabric5.rename(columns={'Patient ID':'Patient_ID'}).reset_index()
    metabric5 = metabric5.set_index('Patient_ID') # [2509 rows x 4 columns]

    path_metabric_rueda= os.path.join(data_dir,'rueda219metabric/')
    path_rueda1 = os.path.join(path_metabric_rueda,'41586_2019_1007_MOESM7_ESM_sup_table5_clinicalinfo.txt')
    path_rueda2 = os.path.join(path_metabric_rueda,'41586_2019_1007_MOESM8_ESM_sup_table6_clinicalinfo.txt')
    path_rueda3 = os.path.join(path_metabric_rueda,'41586_2019_1007_MOESM9_ESM_suptable7_clinicalinfo_recurrence.txt')

    rueda1 = pd.read_csv(path_rueda1,sep='\t')
    rueda1 = rueda1.rename(columns={'METABRIC.ID':'Patient_ID'}).reset_index()
    rueda1 = rueda1.set_index('Patient_ID')  # [3240 rows x 27 columns] 3240 IDs all unique

    rueda2 = pd.read_csv(path_rueda2,sep='\t')
    rueda2 = rueda2.rename(columns={'METABRIC.ID':'Patient_ID'}).reset_index()
    rueda2 = rueda2.set_index('Patient_ID') # [1980 rows x 35 columns] 1980 unique IDs

    # rueda3 # recurrence. has several lines with same IDS

    full_metabric = pd.concat([metabric,metabric2,metabric3,metabric4,metabric5 ], axis=1) # [2516 rows x 68 columns]
    full_metabric2 = pd.concat([full_metabric, rueda1, rueda2], axis=1) # [3774 rows x 130 columns]
    full_metabric2['ID'] = full_metabric2.index.str.replace(r"-", "", regex=True) # 3774 unique IDS
    return full_metabric2


def imc_METABRIC2022():
    """
    :return:
    """

    full_metabric = metabric_metadata()
    # select the ids that have stack images for IMC
    print('metabric full cohort shape', full_metabric.shape)
    # IMC 2022
    path_stacks = ('/home/martinha/PycharmProjects/phd/breast_cancer_imc/breast_data/imc_metabric2022/MBTMEIMCPublic/Images/')
    res = os.listdir(path_stacks)
    # select the fullstack
    res = [x for x in res if x.__contains__('FullStack')] # 794
    ids = [x[:6].upper() for x in res]  # 794 IDS  718 unique

    # indexes in metabric are with a MB-000 and in MB000
    new_metabric2 = full_metabric.loc[full_metabric['ID'].isin(ids)]  # 612 612 unique

    new_res = pd.DataFrame(res, columns=['FileName_FullStack'])
    new_res['ID'] = [x[:6].upper() for x in new_res['FileName_FullStack']]
    # todo get
    new_metabric = pd.merge(new_metabric2, new_res, on=['ID'])  # 683 todo something wring

    survival_df = shuffle(new_metabric)

    # Create Status column withh Boolean values. True if Died of disease. False if alive or dead by other cause
    l = []
    for x in survival_df['VITAL_STATUS']:
        if x == 'Died of Disease': l.append(True)
        else: l.append(False)
    survival_df['Status'] = l # flag where the event is complete or not

    # create second status with 1 if True (died of disease) and 0 if False
    survival_df['StatusBool'] = [1 if x else 0 for x in survival_df['Status']]

    # join all the filename Fullstack for the same patient
    survival_df2 = survival_df.groupby('ID')['FileName_FullStack'].apply(list)
    survival_df = survival_df.drop(['FileName_FullStack'], axis=1)
    survival_df = survival_df.drop_duplicates(subset=['ID'])
    survival_df = survival_df.merge(survival_df2, on = 'ID')

    survival_df = survival_df.dropna(subset=['FileName_FullStack'])

    return survival_df

def get_dataset_metabric():
    metabric22 = imc_METABRIC2022()

    survival_df = metabric22[['FileName_FullStack', 'ID', 'VITAL_STATUS', 'OS_MONTHS']] # 783
    survival_df = survival_df.dropna() # 681

    # Create Status column withh Boolean values. True if Died of disease. False if alive or dead by other cause
    l = []
    for x in survival_df['VITAL_STATUS']:
        if x == 'Died of Disease': l.append(True)
        else: l.append(False)
    survival_df['Status'] = l # flag where the event is complete or not

    # create second status with 1 if True (died of disease) and 0 if False
    survival_df['StatusBool'] = [1 if x else 0 for x in survival_df['Status']]

    # join all the filename Fullstack for the same patient
    survival_df2 = survival_df.groupby('ID')['FileName_FullStack'].apply(list)
    survival_df = survival_df.drop(['FileName_FullStack'], axis=1)
    survival_df = survival_df.drop_duplicates(subset=['ID'])
    survival_df = survival_df.merge(survival_df2, on = 'ID')

    survival_df = survival_df.dropna(subset=['FileName_FullStack'])

    return survival_df



def imc_bodenmiller2020(data_dir='breast_cancer_imc/breast_data/bodenmiller2020/',
                        just_tumor=True, just_invasive_ductal=False):
    """
    :param target Target to predict . e.g. grade Patientstatus, PRStatus, diseasestatus response treatment...
    :param data_dir: data dir folder for boh metadata and images.
    :param just_tumor: if it is to select only tumor images
    :param just_invasive_ductal: if it is only to take into account invasive ductal tumors or all the types
    of tumour (this can also be a target for models. however take iinto account that classes are highly imbalanced

    :return:
    """
    path_data_meta_basel = os.path.join(data_dir, 'Basel_PatientMetadataALTERED.csv')

    # open file with metadat cohort basel
    basel = pd.read_csv(path_data_meta_basel)
    print('basel shape', basel.shape)



    # select just tumor
    if just_tumor:
        basel = basel.loc[basel['diseasestatus'] == 'tumor']
        print('basel tumor samples: {}'.format(basel.shape))
    if just_invasive_ductal:
        basel = basel.loc[basel['histology'].str.startswith('invasive ductal breast')]
        print('basel invasive ductal samples: {}'.format(basel.shape))

    return basel


def imc_bodenmiller2020_Zur(data_dir='breast_cancer_imc/breast_data/bodenmiller2020/'):
    path_data_meta_zur = os.path.join(data_dir, 'Data_publication/ZurichTMA/Zuri_PatientMetadata.csv')

    # open file with metadat cohort basel
    zur = pd.read_csv(path_data_meta_zur)
    print('zur shape', zur.shape)

    return zur



def imc_METABRIC2020():
    """
    :return:
    """

    full_metabric = metabric_metadata()
    # select the ids that have stack images for IMC
    print('metabric full cohort shape', full_metabric.shape)

    # IMC 2020
    data_dir='breast_cancer_imc/breast_data/imc_metabric/'
    path_stacks = os.path.join(data_dir, 'to_public_repository/full_stacks/')
    res = os.listdir(path_stacks)
    ids = [x[:6].upper() for x in res]  # 549 IDS     484 unique  # 548 that start with MB (file with Channel)

    # indexes in metabric are with a MB-000 and in MB000
    new_metabric = full_metabric.loc[full_metabric['ID'].isin(ids)]  # 406 todo something wrong

    new_res = pd.DataFrame(res, columns=['FileName_FullStack'])
    new_res['ID'] = [x[:6].upper() for x in new_res['FileName_FullStack']]
    # todo get
    new_metabric = pd.merge(new_metabric, new_res, on=['ID'])  # 462 todo something wring
    # columns
    # ['age_at_diagnosis',
    # 'size', 'lymph_nodes_positive', 'grade',
    #  'histological_type',
    #  'ER_IHC_status', 'ER.Expr', 'PR.Expr', 'HER2_IHC_status', 'HER2_SNP6_state', 'Her2.Expr',
    #  'Treatment',
    #  'NOT_IN_OSLOVAL_menopausal_status_inferred', 'NOT_IN_OSLOVAL_group',
    #  'NOT_IN_OSLOVAL_stage', 'NOT_IN_OSLOVAL_lymph_nodes_removed',
    #  'NOT_IN_OSLOVAL_NPI', 'NOT_IN_OSLOVAL_cellularity',
    #  'NOT_IN_OSLOVAL_P53_mutation_status',
    #  'NOT_IN_OSLOVAL_P53_mutation_type',
    #  'NOT_IN_OSLOVAL_P53_mutation_details', 'NOT_IN_OSLOVAL_Pam50Subtype',
    #  'NOT_IN_OSLOVAL_IntClustMemb', 'NOT_IN_OSLOVAL_Site',
    #  'NOT_IN_OSLOVAL_Genefu'],


    return new_metabric

