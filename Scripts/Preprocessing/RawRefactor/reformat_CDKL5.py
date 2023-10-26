'''Script to reformat CDKL5 original brain states
'''

import os
import sys
import numpy as np 
import pandas as pd

from dat_to_numpy import parse_dat, convert_dat_to_npy

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from constants import CDKL5_ID_list, CDKL5_het, CDKL5_wt

path_to_folder = '/home/melissa/PREPROCESSING/CDKL5/CDKL5-2/'
downsampling = 1
montage_name = '/home/melissa/taini_main/scripts/standard_16grid_taini1.elc'
number_electrodes = 16
path_to_save_folder = '/home/melissa/PREPROCESSING/CDKL5/CDKL5_numpy/'

animal_ID_list = ['2102', '2104', '2105', '2107', '2027', '2028', '2307', '2308', '2306', '2309', '2322', '2319', '2318', '2320',
                '2323', '2324','2328', '2321', '2024', '2025', '2026', '2100','2101'] 

def reformat_br_file(file):
    column_names = file.columns.values.tolist()
    column_names = column_names[0]
    all_br_values = list(file[str(column_names)].values)
    start_epoch = np.arange(0, 86400, 5)
    end_epoch = np.arange(5, 86405, 5 )
    br_dict = {'brainstate': all_br_values, 'start_epoch': start_epoch, 'end_epoch': end_epoch}
    br_df = pd.DataFrame(data = br_dict)
    return br_df

for animal in animal_ID_list:
    dat_recording = [file for file  in os.listdir(path_to_folder) if file.endswith(animal + '.dat')]
    print(dat_recording[0])
    convert_dat_to_npy(filename = dat_recording[0], path_to_folder= path_to_folder, path_to_save_folder=path_to_save_folder, sample_rate=1000,
    number_electrodes=16, save_as_name = animal + '_CDKL5.npy')
    
unformatted_files = os.listdir('/home/melissa/PREPROCESSING/CDKL5/unformatted_br/')
save_path = '/home/melissa/PREPROCESSING/CDKL5/CDKL5_numpy/'
for file in unformatted_files:
    os.chdir('/home/melissa/PREPROCESSING/CDKL5/unformatted_br/')
    file_br = pd.read_csv(file)
    file_name = file[6:14]
    print(file_name)
    reform_file = reformat_br_file(file_br)
    reform_file.to_pickle(save_path + str(file_name) + '.pkl')