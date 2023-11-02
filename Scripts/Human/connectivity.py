import os
import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_time

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/GRAPH/HUMAN/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

frequency_bands = [[1,4], [4, 8], [8, 12], [13, 30]]
frequency_names = ['delta', 'theta', 'sigma', 'beta']
connectivity_measure = ['coh', 'plv', 'pli', 'wpli']

patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                  'P28 N2', 'P29 N2', 'P30 N1']  

for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    stacked_array = np.stack(epochs, axis=0)
    for connectiv in connectivity_measure:
        for freq_band, freq_name in zip(frequency_bands, frequency_names):
            connectivity_array = spectral_connectivity_time(stacked_array, freqs = freq_band, method= connectiv,
                                          average = False, sfreq=256, faverage=True).get_data()
            np.save(results_path + connectiv + '/' + str(patient) + '_' + str(freq_name) + '.npy', connectivity_array)
    