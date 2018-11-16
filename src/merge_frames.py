import pandas as pd
import numpy as np

'''Start by editing description files
    1) Change all Benign_without_callback to Benign
    2) Find only unique instances of image path and remove duplicates
    3) Possibly split files into CC and MLO?'''

mass_train_desc = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/mammogram_descriptions/mass_case_description_train_set (11).csv')
mass_test_desc = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/mammogram_descriptions/mass_case_description_test_set.csv')

def benign_callback_to_benign(x):
    if x == 'BENIGN_WITHOUT_CALLBACK':
        return 'BENIGN'
    else:
        return x

def split_val_path_1(x):
    ls = x.split('/')
    return ls[0]

def split_val_path_2(x):
    ls = x.split('/')
    return ls[1]

mass_train_2_class = mass_train_desc.drop('pathology', axis=1)
mass_train_2_class['pathology'] = mass_train_desc['pathology'].apply(benign_callback_to_benign)
mass_train_2_class['study_instance_uid'] = mass_train_2_class['image file path'].apply(split_val_path_2)
mass_train_2_class.drop_duplicates(['study_instance_uid'], inplace=True)
mass_train_2_class.to_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/new_csv_descriptions/mass_train_with_class.csv')

mass_test_2_class = mass_test_desc.drop('pathology', axis=1)
mass_test_2_class['pathology'] = mass_test_desc['pathology'].apply(benign_callback_to_benign)
mass_test_2_class['study_instance_uid'] = mass_test_2_class['image file path'].apply(split_val_path_2)
mass_test_2_class.drop_duplicates(['study_instance_uid'], inplace=True)
mass_test_2_class.to_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/new_csv_descriptions/mass_test_with_class.csv')



mass_train = pd.read_csv('/Volumes/chris_external_drive/mass_train.csv')
mass_test = pd.read_csv('/Volumes/chris_external_drive/mass_test.csv')
mass_full = pd.concat([mass_train, mass_test])


#Separating image views in mass_full: MLO and CC
# mass_full_CC = mass_full[mass_full['orientation'] == 'CC']
# mass_full_MLO = mass_full[mass_full['orientation'] == 'MLO']

mass_train_new = mass_train_desc[['patient_id', 'image view', 'pathology']]
mass_train_new['target_dir'] = mass_train_desc['image file path'].apply(split_val_path_1)
mass_train_new['new_target'] = mass_train_desc['image file path'].apply(split_val_path_2)
target_df = mass_train_new.drop_duplicates(['new_target'])

calc_train = pd.read_csv('/Volumes/chris_external_drive/calc_train.csv')
calc_test = pd.read_csv('/Volumes/chris_external_drive/calc_test.csv')
calc_full = pd.concat([calc_train, calc_test])

#separating image views in calc_full: MLO and CC
# calc_full_CC = calc_full[calc_full['orientation'] == 'CC']
# calc_full_MLO = calc_full[calc_full['orientation'] == 'MLO']


calc_train_desc = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/mammogram_descriptions/calc_case_description_train_set.csv')
calc_test_desc = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/mammogram_descriptions/calc_case_description_test_set.csv')
calc_full_desc = pd.concat([calc_train_desc, calc_test_desc])

# meta = pd.read_csv('mass_case_description_train_set (11).csv')
# train = pd.read_csv('mass_train.csv')

# meta_info = meta[['patient_id', 'left or right breast', 'image view', 'pathology']]
# meta_info.columns = ['patient_id', 'laterality', 'orientation', 'pathology']
# meta_info.laterality = meta_info.laterality.map({'LEFT':'L', 'RIGHT':'R'})
# data_merge = meta_info.merge(train, how='inner', left_on=['patient_id','laterality', 'orientation'], right_on=['patient_id', 'laterality','orientation'])
# test_df = pd.DataFrame(0, index-np.arange(mass_full.shape[0], columns=['class'])
#
# for i in range(mass_full.shape[0]):
#     if mass_full_desc['patient_id'].iloc[i] == mass_full['patient_id'].iloc[]


#finding the cases with multiple diagnosis
# targets = mass_full_desc.groupby('patient_id')['pathology'].unique()[new[0]]
