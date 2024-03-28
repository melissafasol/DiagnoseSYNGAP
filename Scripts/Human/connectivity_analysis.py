import os 
import pandas as pd
import numpy as np 
import mne
import itertools

folder_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Connectivity_MNE/raw_data/'
results_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Connectivity_MNE/'
noise_path = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'
frequency_names = ['delta', 'alpha', 'beta', 'gamma']
connectivity_measure = ['coh', 'plv', 'pli', 'wpli']

patient_list  =  [ 'P15 N1','P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P22 N1',
                  'P23 N1', 'P24 N1','P27 N1', 'P28 N1', 'P28 N2', 'P29 N2', 'P30 N1',
                  'P23 N2', 'P23 N3', 'P21 N3', 'P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 
                  'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1', 'P6 N2', 'P7 N1', 'P7 N2', 'P8 N1',
                  'P10 N1', 'P11 N1'] # , #'P6 N1'] 



genotype_human = {'P1': 'WT', 'P2': 'GAP', 'P3': 'GAP', 'P4': 'WT', 
                  'P5': 'GAP', 'P6': 'GAP', 'P7': 'GAP', 'P8' : 'WT',
                  'P9': 'GAP', 'P10': 'GAP', 'P11': 'WT', 'P12': 'WT',
                  'P13': 'GAP', 'P14': 'WT', 'P15': 'GAP', 'P16': 'GAP',
                  'P17': 'WT', 'P18': 'WT', 'P19': 'WT', 'P20': 'GAP',
                  'P21': 'WT', 'P22': 'GAP', 'P23': 'GAP', 'P24': 'WT',
                  'P25': 'WT', 'P26': 'GAP', 'P27': 'WT', 'P28': 'WT',
                  'P29': 'WT', 'P30': 'GAP'} 

channel_labels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']

patient_ls_df = []
for patient in patient_list:
    print(patient)
    patient_id = patient.split()[0]
    print(patient_id)
    for key, value in genotype_human.items():
        if key == patient_id:
            genotype = value
        else:
            pass
    noise_file = np.load(noise_path + patient + '_noise.npy')
    metric_ls = []
    for metric in connectivity_measure:
        print(metric + ' starting')
        frequency_ls = []
        for frequency in frequency_names:
            print(frequency + ' starting')
            folder_path = folder_dir + str(metric) + '/'
            patient_file = np.load(folder_path  + patient + '_' + frequency + '.npy')
            channel_combinations = list(itertools.product(channel_labels, repeat=2))
            pat_freq_ls_conn = []
            # Iterate through the connectivity matrix and print channel combinations
            for idx, value in enumerate(patient_file):
                if idx in noise_file:
                    pass
                else:
                    idx_ls = []
                    for i, conn_value in enumerate(value):
                        if conn_value != 0:
                            chan_1 = channel_combinations[i][0]
                            chan_2 = channel_combinations[i][1]
                            connect_dict = {'Idx': [idx], 'Genotype': [genotype], 'Patient_ID': [patient],
                                            'Frequency': [str(frequency)], 'Channel': [chan_1 + '_' + chan_2],
                                            str(metric): conn_value}
                            connect_df = pd.DataFrame(data = connect_dict)
                            
                            idx_ls.append(connect_df)
                            #print(connect_df)
                            #if connect_df.isna().any().any():
                            #    pass
                            #else:
                    #if len(idx_ls) > 0:
                    idx_concat = pd.concat(idx_ls, axis = 0)
                    #idx_dup = idx_concat.loc[:, ~idx_concat.columns.duplicated()]
                    pat_freq_ls_conn.append(idx_concat)
                    #else:
                    #    pass
            #if len(pat_freq_ls_conn) > 0:
            pat_freq_df = pd.concat(pat_freq_ls_conn, axis = 0)#.reset_index(drop=True)
            frequency_ls.append(pat_freq_df)
            #else:
            #    pass
        #if len(frequency_ls) > 0:
        frequency_df = pd.concat(frequency_ls, axis = 0)
        print(frequency_df)
        freq_dup = frequency_df.loc[:, ~frequency_df.columns.duplicated()]
        freq_dup_na = freq_dup.dropna()
        freq_dup_na.to_csv(results_dir + (str(patient) + '_' + str(metric) + '.csv'))
        #metric_ls.append(freq_dup_na)
        #else:
            #pass
    #metric_concat = pd.concat(metric_ls, axis = 1)
    #print('metric complete')
    #metric_dup = metric_concat.loc[:, ~metric_concat.columns.duplicated()].dropna(axis = 0)
    #print(metric_dup)
    #metric_dup.to_csv(results_dir + (str(patient) + '_all_conn_measures.csv'))
    #patient_ls_df.append(metric_dup)
    #print('patient complete')
    
#all_patients = pd.concat(patient_ls_df)
#all_patients.to_csv(results_dir + 'all_patients_all_conn_measures.csv')
#print('dataframe saved')