import os 
import sys
import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 
from mne_connectivity import spectral_connectivity_epochs

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  = ['P27 N1'] 


for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    stacked_array = np.stack(epochs, axis=0)
    test_pli = spectral_connectivity_epochs(stacked_array, method='pli', sfreq=256,fmin=9, fmax=11, faverage=True).get_data()[:, 0]
    print(test_pli)
    print(len(test_pli))
    break