import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 
import EntropyHub as EH

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path =  '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Human/DispEn/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                   'P29 N2', 'P30 N1']  


def calculate_dispen(data_input, patient_id, channel_list, clean_indices, results_path):
    # Initialize a list to collect all epochs' data
    all_epochs_data = []

    # Iterate over all clean epochs indices
    for idx in clean_indices:
        epoch_data = {}

        # Calculate DispEn for each channel in the current epoch
        for chan_idx, channel in enumerate(channel_list):
            # Assume EH.DispEn returns a dispersion entropy value
            Dispx, RDE = EH.DispEn(Sig = np.array(data_input[idx][chan_idx, :]), m=3, tau=2, c=4, Typex='ncdf')
            epoch_data[str(channel) + '_dispen'] = Dispx

        # Append the dictionary to the list after processing all channels for this epoch
        all_epochs_data.append(epoch_data)

    # Convert the list of dictionaries to a DataFrame
    dispen_df = pd.DataFrame(all_epochs_data)

    # Save the DataFrame to a CSV file
    dispen_df.to_csv(results_path + str(patient_id) + '_dispen.csv', index=False)

for patient in patient_list:
    print(f'Processing {patient}')
    file_name = f'{patient}_(1).edf'
    filtered_data = load_filtered_data(human_data_folder, file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, 256, 5)
    clean_indices = select_clean_indices(noise_directory, patient, number_epochs)
    
    # Calculate DispEn for each channel
    channels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']
    calculate_dispen(data_input=epochs, patient_id=patient,
                     channel_list=channels, clean_indices=clean_indices,
                     results_path=results_path)
    
    print('All channels calculated')
