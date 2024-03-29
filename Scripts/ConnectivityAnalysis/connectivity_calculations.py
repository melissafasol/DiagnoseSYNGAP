import sys
import os
import pandas as pd
import numpy as np 

from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ID_ls, SYNGAP_2_ID_ls

from connectivity_class import process_data

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Rat/Connectivity/'
error_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/ConnectivityErrors/'

analysis_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091', 'S7101',
                'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']

connectivity_type = ['cross_corr',  'phase_lock_val'] 

for connectivity_cal in connectivity_type:
    for animal in analysis_ls:
        print(f'Processing {animal} for {connectivity_cal}')
        load_files = LoadFiles(directory_path, animal)
        if animal in SYNGAP_2_ID_ls:
            process_data(load_files, animal, connectivity_cal, results_path, is_single_file=False)
        elif animal in SYNGAP_1_ID_ls:
            process_data(load_files, animal, connectivity_cal, results_path, is_single_file=True)
