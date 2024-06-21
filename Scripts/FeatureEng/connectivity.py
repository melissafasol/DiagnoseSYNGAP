import os
import sys
import numpy as np 
import pandas as pd 
from mne_connectivity import spectral_connectivity_time
from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr
from connectivity_class import ConnectivityClass

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ls, SYNGAP_2_ls, analysis_ls

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
results_path = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Connectivity/PLV/'
channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
channel_labels = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
frequency_bands = [(1, 5), (5, 11), (11, 16), (16, 30), (30, 48)]
frequency_names = ['delta', 'theta', 'sigma', 'beta', 'gamma']
connectivity_cal = 'phase_lock'

analysis_ls = [ 'S7096', 'S7075', 'S7071']
for animal in analysis_ls:
    print(f'loading {animal}')
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    if animal in SYNGAP_2_ls:
        num_epochs = 34560
        data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data = np.concatenate([data_1, data_2], axis = 1)
        connectivity_calculations = ConnectivityClass(data = data, brain_state = brain_state_1,
                                                      channels = channel_labels, animal_id = animal)
        filtered_data = connectivity_calculations.prepare_data(num_epochs = num_epochs, connectivity = connectivity_cal,
                                                               low = None, high = None)
    if animal in SYNGAP_1_ls:
        num_epochs = 17280
        data, brain_state = load_files.load_one_analysis_file(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        connectivity_calculations = ConnectivityClass(data = data, brain_state = brain_state,
                                                      channels = channel_labels, animal_id = animal)
         
    freq_results = []
    for (low, high), label in zip(frequency_bands, frequency_names):
        print(f'Processing {label} band: {low}-{high} Hz for {animal}')
        if connectivity_cal == 'phase_lock':
            filtered_data = connectivity_calculations.prepare_data(num_epochs = num_epochs, connectivity = connectivity_cal)
            connect_array = connectivity_calculations.calculate_plv_mne(filtered_data = filtered_data, 
                                                                          freq_band = (low, high))
            plv_df = connectivity_calculations.analyse_plv(array = connect_array, freq_band = label)
            freq_results.append(plv_df)
        if connectivity_cal == 'cross_corr':
            filtered_data = connectivity_calculations.prepare_data(num_epochs = num_epochs, connectivity = connectivity_cal,
                                                                  low = low, high = high)
            df_result, error = connectivity_calculations.calculate_max_cross_corr(filtered_data = filtered_data,
                                                                                  num_epochs = num_epochs, freq_band = label)
            freq_results.append(df_result)
    freq_concat = pd.concat(freq_results)
    all_frequencies_concat = pd.concat(freq_concat, axis = 1)
    all_frequencies_concat.to_csv(os.path.join(results_path, f'{animal}_{connectivity_cal}.csv'))