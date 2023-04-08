''''Script for all functions related to seizures'''

import os 
import numpy as np 
import pandas as pd 
import sys 


seizure_br_path = '/home/melissa/PREPROCESSING/GRIN2B/seizures/'

class Seizures:
    
    sampling_rate = 250.4
    
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_seizure_br(self, animal_id):
        br_1_file = 'GRIN2B_' + str(animal_id) + '_BL1_Seizures.csv'
        br_2_file = 'GRIN2B_' + str(animal_id) + '_BL2_Seizures.csv'
        
        if os.path.isfile(self.file_path + br_1_file) == True and os.path.isfile(self.file_path + br_1_file) == True:
            os.chdir(self.file_path)
            br_1 = pd.read_csv(br_1_file)
            br_2 = pd.read_csv(br_2_file)
            return br_1, br_2
        else:
            pass
        
    def extract_long_seizure_epochs(self, data_file, br_file):
        '''function to find indices greater than 5 seconds for entropy analysis'''
        
        long_indices = [idx for idx, value in enumerate(br_file['dur'].astype(int).to_list()) if value > 5]
        seiz_start = [int(br_file['sec_start'][value]*self.sampling_rate) for value in long_indices]
        seiz_end = [int(br_file['sec_end'][value]*self.sampling_rate) for value in long_indices]    
        epoch_ls = [data_file[:, start:end] for start, end in zip(seiz_start, seiz_end)]
        return seiz_start, seiz_end, epoch_ls
    
    
    
    
    
    