import os 
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import scipy
from sklearn import mixture
import scipy.stats as stats
import openpyxl
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables



def extract_visual_scores_packet_loss(path_to_data, file_name, column_name, target_value):
    '''path_to_data = '/home/melissa/PREPROCESSING
    file_name = 70_noise_validation.xlsx
    column_to_extract = 'B'
    target_value = 1 (if you want to extract the seizures)
    '''
    
    visual_workbook = openpyxl.load_workbook(str(path_to_data) + str(file_name))
    visual_sheet = visual_workbook.active
    column_data = [cell.value for cell in visual_sheet[column_name]]
    
    vis_score_noise = np.array(column_data[1:17281])
    
    for idx, value in enumerate(vis_score_noise):
        if value == None:
            print(idx)
            
    noise_unique, noise_counts = np.unique(vis_score_noise, return_counts=True)
    print('unique values')
    print(noise_unique)
    print('unique value counts')
    print(noise_counts)
    
    def find_indices(arr, target):
        indices = []
        for i, value in enumerate(arr):
            if value == target:
                indices.append(i)
        return indices


    target_value = 1
    noise_indices = find_indices(vis_score_noise, target_value)

    return vis_score_noise, noise_indices


def remove_typical_packet_loss(packet_loss_dir, file_name):
    '''remove indices with mV values > 3000 mV'''
    file = pd.read_pickle(str(packet_loss_dir) + str(file_name))
    packet_loss = file.loc[file['brainstate'] == 6]
    packet_loss_indices = np.array(packet_loss.index)

    return packet_loss_indices


def extract_sleep_stages(file_path, file_name, column_name):
    '''function to extract sleep stages from visually scored data'''
    workbook = openpyxl.load_workbook(str(file_path) + str(file_name))
    sheet = workbook.active
    column_data = [cell.value for cell in sheet[column_name]]
    test_vis_score = column_data[8:17288]
    vis_score = np.array(test_vis_score)
    
    return vis_score


#def generate_3d_data(emg_array, eeg_array, noise_indices):
#    
#    '''calculate features to feed into GMM'''
#    
#    indices = np.arange(0, 17280, 1)
#    
#    array_3D_ls = []
#    
#    for idx, eeg_epoch, emg_epoch in zip(indices, eeg_array, emg_array):
#        if idx in noise_indices:
#            pass
#        else:
#            try:
#                epoch_data_ls = []
#                freq, power_emg = scipy.signal.welch(emg_epoch, window='hann', fs=250.4, nperseg=1252)
#                freq, power_eeg = scipy.signal.welch(eeg_epoch, window='hann', fs=250.4, nperseg=1252)
#                slope, intercept = np.polyfit(freq, power_eeg, 1)
#                smoothed_theta_power = np.max(np.log(np.convolve(freq_theta, np.ones(num_coefficients)/num_coefficients, mode='same')))
#
#                #gamma_slope, gamma_intercept = np.polyfit(freq[150:241], power_eeg[150:241], 1)
#
#                #epoch_data_ls.append(gamma_slope)
#                epoch_data_ls.append(slope)
#                array_3D_ls.append(epoch_data_ls)
#            except:
#                print(idx)
#    
#    all_epochs = np.array(array_3D_ls)
#    
#    
#    return all_epochs #all_epochs #standardized_epochs

def generate_3d_data(emg_array, eeg_array, theta_band=(4, 8), num_coefficients=10):
    """
    Calculate features to feed into GMM.
    """
    indices = np.arange(len(eeg_array))  # Assuming eeg_array's length matches the required range
    
    array_3D_ls = []
    
    for idx, (eeg_epoch, emg_epoch) in enumerate(zip(eeg_array, emg_array)):
        #if idx in noise_indices:
        #    continue  # Skip noisy indices

        try:
            # Calculate power spectrum for EMG
            freq_emg, power_emg = scipy.signal.welch(emg_epoch, fs=250.4, window='hann', nperseg=1252)
            # EMG feature: peak power frequency
            power_emg_avg = power_emg[300:450]

            # Calculate power spectrum for EEG
            freq_eeg, power_eeg = scipy.signal.welch(eeg_epoch, fs=250.4, window='hann', nperseg=1252)
            power_eeg_avg = power_eeg[5:100]
            # EEG feature: slope of the power spectrum

            # Theta power calculation
            theta_mask = (freq_eeg >= theta_band[0]) & (freq_eeg <= theta_band[1])
            smoothed_theta_power = np.mean(np.log(power_eeg[theta_mask]))  # Mean log power in theta band

            # Create a 3D data point
            epoch_data_ls = [power_eeg_avg, power_emg_avg, smoothed_theta_power]
            array_3D_ls.append(epoch_data_ls)

        except Exception as e:
            print(f"Error at index {idx}: {e}")

    all_epochs = np.array(array_3D_ls)
    return all_epochs


def calculate_spectral_features(animal_id, sleep_score_values, emg_array, eeg_array):
    
    '''calculate features to feed into GMM'''
    
    indices = np.arange(0, 17281, 1)
    
    array_3D_ls = []
    
    for idx, eeg_epoch, emg_epoch, sleep_score in zip(indices, eeg_array, emg_array, sleep_score_values):
        freq, power_eeg = scipy.signal.welch(eeg_epoch, window='hann', fs=250.4, nperseg=1252)
        freq, power_emg = scipy.signal.welch(emg_epoch, window='hann', fs=250.4, nperseg=1252)
        slope, intercept = np.polyfit(freq[5:100], power_eeg[5:100], 1)
        if slope < -8:
            pass
        else:
            
            theta_band = power_eeg[25:50] #5 - 10
            freq_theta = np.max(np.log(theta_band + 1e-8))

            full_band_eeg = power_eeg[5:100] #1 - 20Hz
            freq_eeg = np.mean(np.log(full_band_eeg + 1e-8))

            full_band_emg = power_emg[300:450] #60 - 90
            freq_emg = np.mean(np.log(full_band_emg + 1e-8))

            data_dict = {'SleepScore': [sleep_score], 'Animal': [animal_id], 'Theta': [freq_theta],
                        'overall_eeg': [freq_eeg], 'overall_emg': [freq_emg],
                        'Slope': [slope]}
            data_df = pd.DataFrame(data = data_dict)
            
            
            array_3D_ls.append(data_df)
    
    all_epochs = pd.concat(array_3D_ls)
    
    return all_epochs