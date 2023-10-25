import os 
import pandas as pd
import numpy as np 
import mne
import itertools

metrics = ['coh', 'pli', 'plv', 'wpli']

results_path = '/home/melissa/RESULTS/GRAPH/HUMAN/'
noise_path = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'
frequency_names = ['delta', 'theta', 'sigma', 'beta']
connectivity_measure = ['coh', 'plv', 'pli', 'wpli']

patient_list  =  [  'P10 N1', 'P11 N1', 'P15 N1', 'P16 N1', 'P17 N1', 'P18 N1',
                'P20 N1', 'P21 N1', 'P21 N2', 'P22 N1', 'P23 N1', 'P24 N1','P27 N1',
                 'P28 N1', 'P28 N2', 'P29 N2', 'P30 N1', 'P23 N2', 'P23 N3', 'P21 N3', 'P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 
                'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1','P6 N1', 'P6 N2', 'P7 N1', 'P7 N2',
                'P8 N1',]

channel_labels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']

patient_ls_df = []
for patient in patient_list:
    print(patient)
    noise_file = np.load(noise_path + patient + '_noise.npy')
    metric_ls = []
    for metric in metrics:
        print(metric + ' starting')
        for frequency in frequency_names:
            print(frequency + ' starting')
            folder_path = results_path + str(metric) + '/'
            patient_file = np.load(folder_path  + patient + '_' + frequency + '.npy')
            channel_combinations = list(itertools.product(channel_labels, repeat=2))
            frequency_ls_conn = []
            # Iterate through the connectivity matrix and print channel combinations
            for idx, value in enumerate(patient_file):
                if idx in noise_file:
                    pass
                else:
                    for i, conn_value in enumerate(value):
                        if conn_value != 0:
                            connect_dict = {'Metric': [metric], 'Patient_ID': [patient],
                                'Frequency': [frequency],
                            'Channel': [channel_combinations[i]], 
                            str(metric): conn_value, 'Idx': [idx]}
                            connect_df = pd.DataFrame(data = connect_dict)
                            frequency_ls_conn.append(connect_df)
            frequency_df = pd.concat(frequency_ls_conn)
            metric_ls.append(frequency_df)
    print('metric complete')
    patient_df = pd.concat(metric_ls)
    patient_ls_df.append(patient_df)
    print('patient complete')
    
all_patients = pd.concat(patient_ls_df)
all_patients.to_csv(results_path + 'all_patients_all_conn_measures.csv')
print('dataframe saved')