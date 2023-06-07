import os 
import sys
import pandas as pd 
import numpy as np
import scipy


import fooof
from fooof import FOOOF
from fooof.core.io import save_fm, load_json
from fooof.core.reports import save_report_fm

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
#from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables
from spec_slope_functions import SpectralSlope

def theta(dat_array):

    theta_power_ls = []

    for epoch in dat_array:
            freq, power = scipy.signal.welch(epoch, window='hann', fs=250.4, nperseg=1252)
            #freq_interest (30 - 48Hz, 30/0.2 = 150: 48/0.2 = 240)
            freq_theta = np.mean(power[25:51])
            theta_power_ls.append(freq_theta)
    
    theta_array = np.array(theta_power_ls)
    return theta_array
    

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Theta_Power/'

SYNGAP_2_ID_ls =  ['S7096', 'S7070','S7072','S7083', 'S7063','S7064','S7069', 'S7086','S7091'] #'S7101', 
SYNGAP_1_ID_ls = ['S7101','S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']
 
motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]
somatosensory = [0, 6, 9, 13]

for animal in SYNGAP_2_ID_ls:
    print('motor starting')
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    #data_1,brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    print('data loaded')
    #only select eeg channels and filter with bandpass butterworth filter before selecting indices
    noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
    noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
    bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
    bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
    print('data filtered')
    filter_1_1 = np.split(bandpass_filtered_data_1[1], 17280, axis = 0)
    filter_1_2 = np.split(bandpass_filtered_data_1[2], 17280, axis = 0)
    filter_1_3 = np.split(bandpass_filtered_data_1[3], 17280, axis = 0)
    filter_1_10 = np.split(bandpass_filtered_data_1[10], 17280, axis = 0)
    filter_1_11 = np.split(bandpass_filtered_data_1[11], 17280, axis = 0)
    filter_1_12 = np.split(bandpass_filtered_data_1[12], 17280, axis = 0)
    
    filter_2_1 = np.split(bandpass_filtered_data_2[1], 17280, axis = 0)
    filter_2_2 = np.split(bandpass_filtered_data_2[2], 17280, axis = 0)
    filter_2_3 = np.split(bandpass_filtered_data_2[3], 17280, axis = 0)
    filter_2_10 = np.split(bandpass_filtered_data_2[10], 17280, axis = 0)
    filter_2_11 = np.split(bandpass_filtered_data_2[11], 17280, axis = 0)
    filter_2_12 = np.split(bandpass_filtered_data_2[12], 17280, axis = 0)

    print('all channels filtered')
    
    power_1_1 = theta(filter_1_1)
    power_1_2 = theta(filter_1_2)
    power_1_3 = theta(filter_1_3)
    power_1_10 = theta(filter_1_10)
    power_1_11 = theta(filter_1_11)
    power_1_12 = theta(filter_1_12)
    
    power_2_1 = theta(filter_2_1)
    power_2_2 = theta(filter_2_2)
    power_2_3 = theta(filter_2_3)
    power_2_10 = theta(filter_2_10)
    power_2_11 = theta(filter_2_11)
    power_2_12 = theta(filter_2_12)
   
    print('all channels calculated')
    
    mean_array_1 = np.mean(np.array([power_1_1, power_1_2, power_1_3, power_1_10, power_1_11, power_1_12]), axis = 0)
    mean_array_2 = np.mean(np.array([power_2_1, power_2_2, power_2_3, power_2_10, power_2_11, power_2_12]), axis = 0)
    power_array = np.concatenate((mean_array_1, mean_array_2), axis = 0)
  
    os.chdir(results_path)
    np.save(str(animal) + '_power.npy', power_array)