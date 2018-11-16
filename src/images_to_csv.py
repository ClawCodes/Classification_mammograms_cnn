import os,sys
import shutil
import pydicom
import pandas as pd
import numpy as np

def mammogram_train_set_to_csv(root_directory, target_directory):

        '''Specific for the CBIS-DDSM data structure for mammogram images.
        Put all images and their corresponding patient_id, laterality, orientation,
        and image data type into a pandas dataframe and save dataframe as csv

        *** THIS IS FOR THE TRAINING SETS ONLY DUE TO DIFFERENCE IN NAMING FROM TEST SETS ***'''

        count = 0
        full = []
        for dir,subdirs,files in sorted(os.walk(root_directory)):
                for file in files:
                    fullpath = os.path.realpath( os.path.join(dir,file) )
                    dirname = fullpath.split(os.path.sep)
                    ds = pydicom.read_file(fullpath)
                    ls = []
                    ls.append(dirname[4][14:21]) #naming specific to training sets
                    ls.append(dirname[4][22])
                    ls.append(ds.PatientOrientation)
                    ls.append(ds.pixel_array)
                    ls.append(ds.pixel_array.dtype)
                    full.append(ls)
                    count+=1
                    print(count) #just to confirm it's running

        df = pd.DataFrame(full, columns=['patient_id', 'laterality', 'orientation', 'image', 'dtype'])
        df.to_csv(target_directory)

def mammogram_test_set_to_csv(root_directory, target_directory):

        '''Specific for the CBIS-DDSM data structure for mammogram images.
        Put all images and their corresponding patient_id, laterality, orientation,
        and image data type into a pandas dataframe and save dataframe as csv.

        *** THIS IS FOR THE TEST SETS ONLY DUE TO DIFFERENCE IN NAMING FROM TRAINING SETS ***'''

        count = 0
        full = []
        for dir,subdirs,files in sorted(os.walk(root_directory)):
                for file in files:
                    fullpath = os.path.realpath( os.path.join(dir,file) )
                    dirname = fullpath.split(os.path.sep)
                    ds = pydicom.read_file(fullpath)
                    ls = []
                    ls.append(dirname[4][10:17]) #naming specific to test sets
                    ls.append(dirname[4][18])
                    ls.append(ds.PatientOrientation)
                    ls.append(ds.pixel_array)
                    ls.append(ds.pixel_array.dtype)
                    full.append(ls)
                    count+=1
                    print(count) #just to confirm it's running

        df = pd.DataFrame(full, columns=['patient_id', 'laterality', 'orientation', 'image', 'dtype'])
        df.to_csv(target_directory)


if __name__ == '__main__':

    #mass sets
    mammogram_train_set_to_csv('/Volumes/chris_external_drive/Mass-train-CBIS-DDSM', \
    '/Volumes/chris_external_drive/mass_train.csv')

    mammogram_test_set_to_csv('/Volumes/chris_external_drive/Mass-test-CBIS-DDSM', \
    '/Volumes/chris_external_drive/mass_test.csv')


    #calc sets
    mammogram_train_set_to_csv('/Volumes/chris_external_drive/Calc-train-CBIS-DDSM', \
    '/Volumes/chris_external_drive/calc_train.csv')

    mammogram_test_set_to_csv('/Volumes/chris_external_drive/Calc-test-CBIS-DDSM', \
    '/Volumes/chris_external_drive/calc_test.csv')
