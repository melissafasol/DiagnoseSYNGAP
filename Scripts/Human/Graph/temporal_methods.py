import os
import mne 
import numpy as np
import pandas as pd
import eegraph


clean_directory = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP_Human_Data_Clean/'  
save_directory = '/home/melissa/RESULTS/GRAPH/'

patient_list = ['P1 N1', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1','P6 N1', 'P6 N2', 'P7 N1', 'P7 N2', 'P8 N1',
                 'P10 N1', 'P11 N1', 'P15 N1', 'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                 'P22 N1','P23 N1', 'P23 N3', 'P24 N1', 'P27 N1', 'P28 N1', 'P28 N2', 'P29 N2', 'P30 N1']

methods = ['shannon_entropy'] #'pearson_correlation #'cross_correlation' #'corr_cross_correlation', 


for freq_method in methods:
    os.chdir(clean_directory)
    for file in os.listdir(clean_directory):
        os.chdir(clean_directory)
        if file == 'ipynb_checkpoints':
            pass
        else:
            print(file[0:6] + ' starting graph calculations')
            G = eegraph.Graph()
            window = 5
            connectivity_measure = freq_method
            G.load_data(path = file)
            graphs, connectivity_matrix = G.modelate(window_size = window, connectivity = connectivity_measure)
            os.chdir(save_directory + freq_method)
            np.save(str(file[0:6]) + '_' + str(freq_method) + '.npy', connectivity_matrix)
            print(file[0:6] + ' saved')
        