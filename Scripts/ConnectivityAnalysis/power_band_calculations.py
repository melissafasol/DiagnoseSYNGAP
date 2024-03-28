import os 
import sys
import pandas as pd 
import numpy as np

from power_band_class import PowerBands

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Rat/Power/'
error_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/ConnectivityErrors/'


SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101']
syngap_2_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091']

analysis_ls = ['S7074'] #['S7101', 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075',
              # 'S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091', 'S7101']

frequency_names = ['delta', 'theta', 'sigma', 'beta', 'gamma']
frequency_bands = [[1, 5], [5, 11], [11, 16], [16, 30], [30, 48]]
motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]
somatosensory = [0, 6, 9, 13]

for animal in analysis_ls:
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    if animal in syngap_2_ls:
        data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
        print('data loaded')
        #only select eeg channels and filter with bandpass butterworth filter before selecting indices
        noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
        noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
        bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
        freq_ls = []
        for freq_name, freq_band in zip(frequency_names, frequency_bands):
            print(freq_band[0])
            print(freq_band[1])
            power_calc = PowerBands(freq_low = freq_band[0], freq_high = freq_band[1],fs = 250.4, nperseg = 1252)
            motor_mean_1, visual_mean_1, soma_mean_1 = power_calc.functional_region_power_recording(bandpass_filtered_data_1, motor, visual, somatosensory)
            motor_mean_2, visual_mean_2, soma_mean_2 = power_calc.functional_region_power_recording(bandpass_filtered_data_2, motor, visual, somatosensory)
            motor_mean = np.concatenate((motor_mean_1, motor_mean_2), axis = 0)
            visual_mean = np.concatenate((visual_mean_1, visual_mean_2), axis = 0)
            soma_mean = np.concatenate((soma_mean_1, soma_mean_2), axis = 0)
    
            power_dict = {str('Motor') + '_' + str(freq_name): motor_mean.tolist(), 
            str('Visual') + '_' + str(freq_name): visual_mean.tolist(),
            str('Soma') + '_' + str(freq_name): soma_mean.tolist()}
            freq_ls.append(pd.DataFrame(data = power_dict))
        all_freqs = pd.concat(freq_ls, axis = 1)
        id_df = pd.DataFrame(data = {'Animal_ID': [animal]*17280})
        #df_concat = pd.concat([all_freqs, id_df], axis = 1)
        #df_concat.to_csv(results_path + str(animal) + '_power_all_frequency_bands.csv')

    if animal in SYNGAP_1_ID_ls :
        print('one recording')
        data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
        noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
        freq_ls = []
        for freq_name, freq_band in zip(frequency_names, frequency_bands):
            print(freq_name)
            power_calc = PowerBands(freq_low = freq_band[0], freq_high = freq_band[1],fs = 250.4, nperseg = 1252)
            motor_mean, visual_mean, soma_mean = power_calc.functional_region_power_recording(bandpass_filtered_data_1, motor, visual, somatosensory)
            os.chdir('/home/melissa/RESULTS/XGBoost/SYNGAP1/')
            np.save('S7074_motor' + freq_name + '.npy', motor_mean)
            np.save('S7074_visual' + freq_name + '.npy', visual_mean)
            np.save('S7074_soma' + freq_name + '.npy', soma_mean)
            print('saved delta')
            power_dict = {
            str('Motor') + '_' + str(freq_name): motor_mean.tolist(), 
            str('Visual') + '_' + str(freq_name): visual_mean.tolist(),
            str('Soma') + '_' + str(freq_name): soma_mean.tolist()}
            freq_ls.append(pd.DataFrame(data = power_dict))
        #all_freqs = pd.concat(freq_ls, axis = 1)
        #id_df = pd.DataFrame(data = {'Animal_ID': [animal]*17280})
        #df_concat = pd.concat([all_freqs, id_df], axis = 1)
        #df_concat.to_csv(results_path + str(animal) + '_power_all_frequency_bands.csv')
        
        