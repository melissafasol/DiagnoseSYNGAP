import os
import sys
import numpy as np
import pandas as pd 
from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr
from mne_connectivity import spectral_connectivity_time

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ls, SYNGAP_2_ls, analysis_ls



class ConnectivityClass:
    def __init__(self, filtered_data, channels, animal_id):
        self.filtered_data = filtered_data
        self.channels = channels
        self.animal_id = animal_id
        
    def channel_pairs(self):
        pairs = []
        for i in range(len(self.channels)):
            for j in range(i + 1, len(self.channels)):
                pairs.append([self.channels[i], self.channels[j]])
        return pairs

    def calculate_max_cross_corr(self, num_epochs, freq_band):
        cross_corr_ls = []
        error_ls = []
        for i in range(num_epochs):
            try:
                one_epoch = compute_max_cross_corr(sfreq=250.4, data=self.filtered_data[:, i])
                df = pd.DataFrame([one_epoch], columns=[f'{ch[0]}_{ch[1]}_{freq_band}' for ch in self.channel_pairs()])
                df_other = pd.DataFrame(data = {'Epoch': [i], 'Animal_ID': [self.animal_id]})
                df_concat = pd.concat([df_other, df], axis = 1)
                cross_corr_ls.append(df_concat)
            except Exception as e:
                print(f'Error for index {i}: {e}')
                error_ls.append(i)

        cross_corr_concat = pd.concat(cross_corr_ls)
        error_array = np.array(error_ls)
        return cross_corr_concat, error_array
    
    def calculate_plv_mne(self, filtered_data):
        tr_filter = filtered_data.transpose(1, 0, 2)
        for freq_band, freq_name in zip(frequency_bands, frequency_names):
            connectivity_array = spectral_connectivity_time(tr_filter, freqs=freq_band, n_cycles= 3,
                                                        method= 'plv', average=False, sfreq=250.4,
                                                        faverage=True).get_data()
            for epoch in connectivity_array:
                df = pd.DataFrame([epoch], columns=[f'{ch[0]}_{ch[1]}_{freq_band}' for ch in self.channel_pairs()])
            print(connectivity_array)
            
        #save_path = results_path / f'{animal}_{conn_mes}_{freq_name}.npy'
        #np.save(save_path, connectivity_array)
    
    
def process_data_mne_connect(animal, conn_mes, data, brain_state, frequency_bands, frequency_names, results_path, sfreq=250.4, n_cycles=3):
    noise_filter = NoiseFilter(data, brain_state_file=brain_state, channelvariables=channel_variables, ch_type='eeg')
    bandpass_filtered_data = noise_filter.filter_data_type()
    filter_data = np.moveaxis(np.array(np.split(bandpass_filtered_data, 17280, axis=1)), 1, 0)
    tr_filter = filter_data.transpose(1, 0, 2)
    
    for freq_band, freq_name in zip(frequency_bands, frequency_names):
        connectivity_array = spectral_connectivity_time(tr_filter, freqs=freq_band, n_cycles=n_cycles,
                                                        method=conn_mes, average=False, sfreq=sfreq,
                                                        faverage=True).get_data()
        save_path = results_path / f'{animal}_{conn_mes}_{freq_name}.npy'
        np.save(save_path, connectivity_array)


directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
results_path = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Connectivity/'
channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
channel_labels = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]

for animal in analysis_ls:
    print(f'loading {animal}')
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    if animal in SYNGAP_2_ls:
        data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data_list = [(data_1, brain_state_1), (data_2, brain_state_2)]
        frequency_bands = [(1, 5), (5, 11), (11, 16), (16, 30), (30, 48)]
        frequency_names = ['delta', 'theta', 'sigma', 'beta', 'gamma']
        connectivity_cal = 'cross_corr'
        results = []
        for data, brain_state in data_list:
            freq_results = []
            for (low, high), label in zip(frequency_bands, frequency_names):
                print(f'Processing {label} band: {low}-{high} Hz')
                noise_filter = NoiseFilter(data, brain_state_file=brain_state, channelvariables=channel_variables, ch_type='eeg')
                bandpass_filtered_data = noise_filter.specify_filter(low=low, high=high)
                filtered_data = np.moveaxis(np.array(np.split(bandpass_filtered_data, 17280, axis=1)), 1, 0)
                complexity_calculations = ConnectivityClass(filtered_data, channels = channel_labels, animal_id = animal)
                if connectivity_cal == 'cross_corr':
                    df_result, error = complexity_calculations.calculate_max_cross_corr(num_epochs=17280, freq_band = label)
                    freq_results.append(df_result)

            freq_concat = pd.concat(freq_results)
            results.append(freq_concat) 
        all_frequencies_concat = pd.concat(results, axis = 1)
        all_frequencies_concat.to_csv(os.path.join(results_path, f'{animal}_{connectivity_cal}.csv'))
    else:
        pass