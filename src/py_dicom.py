import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


test = pydicom.read_file('/Volumes/chris_external_drive/Mass-train-CBIS-DDSM/Mass-Training_P_00106_RIGHT_CC/07-20-2016-DDSM-41536/1-full mammogram images-9d2446/000000.dcm')
test_2 = pydicom.read_file('/Volumes/chris_external_drive/Mass-test-CBIS-DDSM/Mass-Test_P_00016_LEFT_CC/10-04-2016-DDSM-30104/1-full mammogram images-14172/000000.dcm')
test_3 = pydicom.read_file('/Volumes/chris_external_drive/Mass-test-CBIS-DDSM/Mass-Test_P_00032_RIGHT_CC/10-04-2016-DDSM-31069/1-full mammogram images-83371/000000.dcm')
test_4 = pydicom.read_file('/Volumes/chris_external_drive/Mass-test-CBIS-DDSM/Mass-Test_P_00114_LEFT_MLO/10-04-2016-DDSM-57563/1-full mammogram images-97726/000000.dcm')
# test_re = resize(test.pixel_array, (200,200,3), mode='constant')


# fig, axs = plt.subplots(2,2)
# axs[0, 0].imshow(ds_cc.pixel_array)
# axs[0, 0].set_title('Craniocaudal view')
# axs[0, 1].imshow(ds2_mlo.pixel_array)
# axs[0, 1].set_title('Mediolateral Oblique view')
# axs[1, 0].imshow(ds_roi_1.pixel_array)
# axs[1, 0].set_title('Region of interest marked')
# axs[1, 1].imshow(ds_roi_2.pixel_array)
# axs[1, 1].set_title('Region of interest cropped')
# plt.tight_layout()
# plt.savefig('/Users/christopherlawton/galvanize/module_2/capstone_2/proposal/full_cc_mlo_roi.png')


#mass dataframes
# mass_train = pd.read_csv('./capstone_2/proposal/mass_case_description_train_set (6).csv')
# mass_test = pd.read_csv('./capstone_2/proposal/mass_case_description_test_set (2).csv')
# mass_full = pd.concat([mass_train, mass_test])
# mass_cc = mass_full[mass_full['image view'] == 'CC']
#
# #calc dataframes
# calc_train = pd.read_csv('./capstone_2/proposal/calc_case_description_train_set (2).csv')
# calc_test = pd.read_csv('./capstone_2/proposal/calc_case_description_test_set.csv')
# calc_full = pd.concat([calc_train, calc_test])
# calc_cc = calc_full[calc_full['image view'] == 'CC']
