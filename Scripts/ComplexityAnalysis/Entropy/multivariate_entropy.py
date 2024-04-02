## have to run package in python 3.6
import os 
import sys
import numpy as np 
import pandas as pd 
import scipy
from scipy import signal 

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs

directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'

def load_analysis_files_te(directory_path, animal_id, start_times_dict, end_times_dict):
    animal_recording = [filename for filename in os.listdir(directory_path) if filename.startswith(animal_id) and filename.endswith('.npy')]
    os.chdir(directory_path)
    recording = np.load(animal_recording[0]) 
    
    start_dict_1 = animal_id + '_1'
    start_dict_2 = animal_id + '_2'
    end_dict_1 = animal_id + '_1A'
    end_dict_2 = animal_id + '_2A'
    
    start_time_1 = start_times_dict[start_dict_1]
    start_time_2 = start_times_dict[start_dict_2]
    end_time_1 = end_times_dict[end_dict_1]
    end_time_2 = end_times_dict[end_dict_2]
    
    recording_1 = recording[:, start_time_1: end_time_1 + 1]
    recording_2 = recording[:, start_time_2: end_time_2 + 1]
        
    return recording_1, recording_2

def filter_data(unfiltered_data):
    
    def butter_bandpass(data):
        order = 3
        sampling_rate = 250.4 
        nyquist = sampling_rate/2
        low = 0.2/nyquist
        high = 100/nyquist
        butter_b, butter_a = signal.butter(order, [low, high], btype = 'band', analog = False)
        filtered_data = signal.filtfilt(butter_b, butter_a, data)
        return filtered_data
    
    indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]

    #Select all, emg, or eeg channel indices to apply bandpass filter                                    
    selected_channels = unfiltered_data[indices, :]     
    bandpass_filtered_data=butter_bandpass(data=selected_channels) 
                    
    return bandpass_filtered_data

for animal in GRIN2B_ID_list:
    print('loading ' + str(animal))
    animal = str(animal)
    data_1, data_2 = load_analysis_files_te(directory_path, animal, start_time_GRIN2B_baseline, end_time_GRIN2B_baseline)
    print('data loaded')
    #only select eeg channels and filter with bandpass butterworth filter before selecting indices  
    bandpass_filtered_data_1 = filter_data(data_1)
    bandpass_filtered_data_2 = filter_data(data_2)
    print('data filtered')
    filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
    filter_2 = np.moveaxis(np.array(np.split(bandpass_filtered_data_2, 17280, axis = 1)), 1, 0)
    for i in np.arange(0, 17280, 1):
        eeg_data = filter_1[:, i]
        data_container = Data(eeg_data, dim_order= 'ps')
        
        #initialise the analysis object and define settings 
        network_analysis = MultivariateTE()

        settings = {'cmi_estimator': 'JidtGaussianCMI',
                    'max_lag_sources': 5,
                    'min_lag_sources': 1}

        #run analysis 
        results = network_analysis.analyse_network(settings = settings, data= data_container)
        results.get_single_target(1, fdr = False) #get the te variables after running analysis 
        print(results)
        break

#plot inferred network to console and via matplotlib
#   results.print_edge_list(weights = 'max_te_lag', fdr = False)
#   plt.show()
