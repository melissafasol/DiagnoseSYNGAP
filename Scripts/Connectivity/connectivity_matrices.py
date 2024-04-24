import os 
import pandas as pd
import numpy as np 
import mne_connectivity
from mne_connectivity import spectral_connectivity_time
import mne
import sys

from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

import matplotlib.pyplot as plt

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, channel_names, SYNGAP_2_ID_ls, SYNGAP_1_ID_ls


analysis_ls = ['S7068', 'S7074', 'S7075', 'S7071', 'S7076', 'S7083', 'S7088', 'S7092', 'S7086', 'S7063', 'S7064', 
               'S7069', 'S7070', 'S7072', 'S7086', 'S7091', 'S7101', 'S7094', 'S7096', 'S7098']


directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'


info = mne.create_info(ch_names= channel_names, sfreq=250.4, ch_types='eeg')

frequency_range =  [10, 16] # [1,5] 
frequency_band =  'sigma' #'delta'

data_results_path = '/home/melissa/RESULTS/SYNGAP1/Thesis/plv/' + str(frequency_band) + '/'

for animal in analysis_ls:
        print(animal)
        print(frequency_band)
        load_files = LoadFiles(directory_path, animal)
        if animal in analysis_ls:
            data_1,  brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
            print('data loaded')
            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
            filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
            tr_filter_1  = filter_1.transpose(1, 0, 2)
            
            br_direct = '/home/melissa/PREPROCESSING/SYNGAP1/cleaned_br_files/'
            br_file = pd.read_pickle(animal + '_BL1.pkl')
            all_indices = br_file.loc[br_file['brainstate'].isin([0, 1, 2])].index.to_list()
            wake_indices = br_file.loc[br_file['brainstate'].isin([0])].index.to_list()
            nrem_indices = br_file.loc[br_file['brainstate'].isin([1])].index.to_list()
            rem_indices = br_file.loc[br_file['brainstate'].isin([2])].index.to_list()
            
        
           
            if len(br_file.loc[br_file['brainstate'].isin([4])]) > 0:
                seizure_indices = br_file.loc[br_file['brainstate'].isin([4])].index.to_list()
                seizure_epochs = tr_filter_1[seizure_indices]
                seiz_con = mne_connectivity.spectral_connectivity_time(seizure_epochs,method='plv', sfreq=250.4,
                                                              freqs = frequency_range, n_cycles = 3, 
                                                              sm_kernel = 'hanning',average=True,
                                                              faverage = True, n_jobs=1)
                
                seiz_matrix = np.squeeze(seiz_con.get_data(output='dense'))
                
                np.save(data_results_path + animal + '_seizure.npy', seiz_matrix)
            
            else:
                pass
        
            all_epochs = tr_filter_1[all_indices]
            all_epochs_con = mne_connectivity.spectral_connectivity_time(all_epochs,method='plv', sfreq=250.4,
                                                              freqs = frequency_range, n_cycles = 3, 
                                                              sm_kernel = 'hanning',average=True,
                                                              faverage = True, n_jobs=1)
            
            
                
            all_epochs_matrix = np.squeeze(all_epochs_con.get_data(output='dense'))
            np.save(data_results_path + animal + '_' + frequency_band + '_all_sleepstages.npy', all_epochs_matrix)
            
            
            wake_epochs = tr_filter_1[wake_indices]
            wake_epochs_con = mne_connectivity.spectral_connectivity_time(wake_epochs,method='plv', sfreq=250.4,
                                                              freqs = frequency_range, n_cycles = 3, 
                                                              sm_kernel = 'hanning',average=True,
                                                              faverage = True, n_jobs=1)
            
            
                
            wake_epochs_matrix = np.squeeze(wake_epochs_con.get_data(output='dense'))
            np.save(data_results_path + animal + '_' + frequency_band + '_wake.npy', wake_epochs_matrix)
            
            
            
            nrem_epochs = tr_filter_1[nrem_indices]
            
            nrem_epochs_con = mne_connectivity.spectral_connectivity_time(nrem_epochs,method='plv', sfreq=250.4,
                                                              freqs = frequency_range, n_cycles = 3, 
                                                              sm_kernel = 'hanning',average=True,
                                                              faverage = True, n_jobs=1)
            
            
                
            nrem_epochs_matrix = np.squeeze(nrem_epochs_con.get_data(output='dense'))
            np.save(data_results_path + animal + '_' + frequency_band + '_nrem.npy', nrem_epochs_matrix)
            
            
            rem_epochs = tr_filter_1[rem_indices]
            
            rem_epochs_con = mne_connectivity.spectral_connectivity_time(rem_epochs,method='plv', sfreq=250.4,
                                                              freqs = frequency_range, n_cycles = 3, 
                                                              sm_kernel = 'hanning',average=True,
                                                              faverage = True, n_jobs=1)
            
            
                
            rem_epochs_matrix = np.squeeze(wake_epochs_con.get_data(output='dense'))
            np.save(data_results_path + animal + '_' + frequency_band + '_rem.npy', rem_epochs_matrix)
        
            
        