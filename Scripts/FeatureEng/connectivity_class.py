import os
import sys
import numpy as np
import pandas as pd 
from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr
from mne_connectivity import spectral_connectivity_time

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter 
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ls, SYNGAP_2_ls, analysis_ls

class ConnectivityClass:
    def __init__(self, data, brain_state, channels, animal_id):
        self.data = data
        self.brain_state = brain_state
        self.channels = channels
        self.animal_id = animal_id
        
    def prepare_data(self, num_epochs, connectivity, low = None, high = None):
        noise_filter = NoiseFilter(self.data, brain_state_file= self.brain_state, channelvariables= channel_variables, ch_type='eeg')
        if connectivity == 'cross_corr':
            if low is None or high is None:
                return KeyError("Both 'low' and 'high' frequency bands must be specified for cross-correlation.")
            else:
                bandpass_filtered_data = noise_filter.specify_filter(low=low, high=high) 
                filtered_data = np.moveaxis(np.array(np.split(bandpass_filtered_data, num_epochs, axis=1)), 1, 0)
        elif connectivity == 'phase_lock':
            bandpass_filtered_data = noise_filter.filter_data_type()
            filtered_data = np.moveaxis(np.array(np.split(bandpass_filtered_data, num_epochs, axis=1)), 1, 0)
        else:
            raise KeyError(f"Invalid connectivity type: {connectivity}. Supported types are 'cross_corr' or 'phase_lock'.")
        return filtered_data
        
    def channel_pairs(self):
        pairs = []
        for i in range(len(self.channels)):
            for j in range(i + 1, len(self.channels)):
                pairs.append([self.channels[i], self.channels[j]])
        return pairs

    
    def calculate_max_cross_corr(self, filtered_data, num_epochs):
        cross_corr_ls = []
        error_ls = []
        for i in range(num_epochs):
            try:
                one_epoch = compute_max_cross_corr(sfreq=250.4, data=filtered_data[:, i])
                # Add metadata (Epoch, Animal_ID) as a separate array, then concatenate it with one_epoch data
                metadata = np.array([i, self.animal_id])  # Assuming self.animal_id is numeric or can be converted to numeric
                combined_data = np.concatenate([metadata, one_epoch])  # Combine metadata and data
                cross_corr_ls.append(combined_data)
            except Exception as e:
                print(f'Error for index {i}: {e}')
                error_ls.append(i)

        # Stack all epoch arrays into one 2D array (epochs x features) and save
        cross_corr_array = np.stack(cross_corr_ls)
        #np.save(os.path.join(save_path, f'{self.animal_id}_{freq_band}_cross_corr.npy'), cross_corr_array)

        # Convert error list to an array for consistency in return format
        error_array = np.array(error_ls)
    
        return cross_corr_array, error_array
    
    def calculate_plv_mne(self, filtered_data, freq_band):
        tr_filter = filtered_data.transpose(1, 0, 2)
        connectivity_array = spectral_connectivity_time(tr_filter, freqs=freq_band, n_cycles= 3,
                                                        method= 'plv', average=False, sfreq=250.4, fmin=freq_band[0],
                                                        fmax=freq_band[1], faverage=True).get_data()
        return connectivity_array
    
    def analyse_plv(self, array, freq_band, num_channels = 14, 
                    channel_labels = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]):

        all_epochs = []
        for epoch in array: 
            data_dict = {}
    
            # ignore self-pairs in array
            for idx_i, i in enumerate(channel_labels):
                for idx_j, j in enumerate(channel_labels):
                    if i != j:
                        index = idx_i * num_channels + idx_j
                        pair_label = f"{i}_{j}_{freq_band}_plv"
                        data_dict[pair_label] = epoch[index]

            # convert the dictionary to df
            df = pd.DataFrame(data_dict)
            all_epochs.append(df)
        df_concat = pd.concat(all_epochs)
        return df_concat