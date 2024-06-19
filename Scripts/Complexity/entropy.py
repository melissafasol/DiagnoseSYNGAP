import os 
import sys
import numpy as np 
import pandas as pd
import EntropyHub as EH

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, analysis_ls, SYNGAP_1_ls, SYNGAP_2_ls

class DispersionEntropy:
    
    def __init__(self, dat_array):
        self.dat_array = dat_array
        
        
    def disp_en(self):
        disp_en_ls =  []

        for epoch in self.dat_array:
            Dispx_1, Ppi_1 = EH.DispEn(epoch, m = 3, tau = 2, c = 4, Typex = 'ncdf')
            disp_en_ls.append(Dispx_1)
        
        disp_en_array = np.array(disp_en_ls)
        
        return disp_en_array

channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
channel_labels = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
results_path = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Complexity/'

for animal in analysis_ls:
    print(f'loading {animal}')
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    
    if animal in SYNGAP_2_ls:
        data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data = np.concatenate([data_1, data_2], axis=1)
        noise_filter = NoiseFilter(data, brain_state_file=brain_state_1, channelvariables=channel_variables, ch_type='eeg')    
        bandpass_filtered_data = noise_filter.filter_data_type()
        
        channel_ls = []
        for i, (chan_idx, chan_label) in enumerate(zip(channel_indices, channel_labels)):
            print(chan_label)
            filter_channel = np.split(bandpass_filtered_data[chan_idx], 34560, axis=0)
            disp_en_chan = DispersionEntropy(filter_channel)
            dispen_array = disp_en_chan.disp_en()
            
            if i == 0:
                # Include 'Epoch' and 'Animal_ID' in the first channel DataFrame
                ent_df_chan = pd.DataFrame(data={'Epoch': np.arange(0, 34560, 1), 'Animal_ID': [animal]*len(filter_channel), f'DispEn_{chan_label}': dispen_array})
            else:
                # Only include the specific channel data in subsequent DataFrames
                ent_df_chan = pd.DataFrame(data={f'DispEn_{chan_label}': dispen_array})
            
            channel_ls.append(ent_df_chan)
        
        # Concatenate on axis=1 to merge DataFrames
        channel_concat = pd.concat(channel_ls, axis=1)
        channel_concat.to_csv(f'{results_path}{animal}_disp_entropy.csv', index=False)
    
    elif animal in SYNGAP_1_ls:
        data, brain_state = load_files.load_one_analysis_file(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data_list = [(data, brain_state)]    
        for data, brain_state in data_list:
            noise_filter = NoiseFilter(data, brain_state_file = brain_state, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data = noise_filter.filter_data_type()
            channel_ls = []
            for i, (chan_idx, chan_label) in enumerate(zip(channel_indices, channel_labels)):
                print(chan_label)
                filter_channel = np.split(bandpass_filtered_data[chan_idx], 17280, axis=0)
                disp_en_chan = DispersionEntropy(filter_channel)
                dispen_array = disp_en_chan.disp_en()
            
                if i == 0:
                    ent_df_chan = pd.DataFrame(data={'Epoch': np.arange(0, 17280, 1), 'Animal_ID': [animal]*len(filter_channel), f'DispEn_{chan_label}': dispen_array})
                else:
                    ent_df_chan = pd.DataFrame(data={f'DispEn_{chan_label}': dispen_array})
            
                channel_ls.append(ent_df_chan)
            channel_concat = pd.concat(channel_ls, axis=1)
            channel_concat.to_csv(f'{results_path}{animal}_disp_entropy.csv', index=False)