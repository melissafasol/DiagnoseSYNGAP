import sys
import os
import pandas as pd
import numpy as np 
from pathlib import Path
from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ID_ls, SYNGAP_2_ID_ls

from connectivity_class import process_data_mcc_plv, load_and_process_mne_connect, process_data_mne_connect

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Rat/Connectivity/'
results_path_mne = Path('/home/melissa/RESULTS/XGBoost/SYNGAP1/mne_connectivity/')
error_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/ConnectivityErrors/'

#frequency values for spectral connectivity 
frequency_bands = [[1, 5], [5, 11], [11, 16], [16, 30], [30, 48]]
frequency_names = ['delta', 'theta', 'sigma', 'beta', 'gamma']

analysis_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091', 'S7101',
                'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']


connectivity_type_mcc_plv = ['cross_corr',  'phase_lock_val'] 
connectivity_measure_mne = ['coh', 'plv', 'pli', 'wpli']

#connectivity loop for MCC and PLV
for connectivity_cal in connectivity_type_mcc_plv:
    for animal in analysis_ls:
        print(f'Processing {animal} for {connectivity_cal}')
        load_files = LoadFiles(directory_path, animal)
        if animal in SYNGAP_2_ID_ls:
            process_data_mcc_plv(load_files, animal, connectivity_cal, results_path, is_single_file=False)
        elif animal in SYNGAP_1_ID_ls:
            process_data_mcc_plv(load_files, animal, connectivity_cal, results_path, is_single_file=True)
            

for conn_mes in connectivity_measure_mne:
    print(f'Processing connectivity measure: {conn_mes}')
    for animal in analysis_ls:
        print(f'Processing animal: {animal}')
        try:
            load_files = LoadFiles(directory_path, animal)
            if animal in SYNGAP_2_ID_ls:
                data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
                print(f'{animal}: data loaded for two analysis files.')
                process_data_mne_connect(animal, conn_mes, data_1, brain_state_1, frequency_bands, frequency_names, results_path, sfreq=250.4, n_cycles=3)
                process_data_mne_connect(animal, conn_mes, data_2, brain_state_2, frequency_bands, frequency_names, results_path, sfreq=250.4, n_cycles=3)
            elif animal in SYNGAP_1_ID_ls:
                data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
                print(f'{animal}: data loaded for one analysis file.')
                process_data_mne_connect(animal, conn_mes, data_1, brain_state_1, frequency_bands, frequency_names, results_path, sfreq=250.4, n_cycles=3)
            else:
                raise ValueError(f'Animal ID {animal} not found.')
        except ValueError as e:
            print(e)