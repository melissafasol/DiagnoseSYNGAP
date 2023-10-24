import os 
import numpy as np 
import pandas as pd


class LoadFiles():
    
    def __init__(self, directory_path, animal_id):
        self.directory_path = directory_path
        self.animal_id = animal_id
        self.brain_1 = animal_id + '_BL1.pkl'
        self.brain_2 = animal_id + '_BL2.pkl'
        self.start_dict_1 = animal_id + '_1'
        self.start_dict_2 = animal_id + '_2'
        self.end_dict_1 = animal_id + '_1A'
        self.end_dict_2 = animal_id + '_2A'
        
        
    def load_two_analysis_files(self, start_times_dict, end_times_dict):
        animal_recording = [filename for filename in os.listdir(self.directory_path) if filename.startswith(self.animal_id) and filename.endswith('.npy')]
        os.chdir(self.directory_path)
        recording = np.load(animal_recording[0]) 
        brain_file_1 = [filename for filename in os.listdir(self.directory_path) if filename == self.brain_1]
        brain_state_1 = pd.read_pickle(brain_file_1[0])
        brain_file_2 = [filename for filename in os.listdir(self.directory_path) if filename == self.brain_2]
        brain_state_2 = pd.read_pickle(brain_file_2[0])
        
        start_time_1 = start_times_dict[self.start_dict_1]
        start_time_2 = start_times_dict[self.start_dict_2]
        
        end_time_1 = end_times_dict[self.end_dict_1]
        end_time_2 = end_times_dict[self.end_dict_2]

        recording_1 = recording[:, start_time_1: end_time_1 + 1]
        recording_2 = recording[:, start_time_2: end_time_2 + 1]
        
        return recording_1, recording_2, brain_state_1, brain_state_2
    
    def load_one_analysis_file(self, start_times_dict, end_times_dict):
        animal_recording = [filename for filename in os.listdir(self.directory_path) if filename.startswith(self.animal_id) and filename.endswith('.npy')]
        os.chdir(self.directory_path)
        recording = np.load(animal_recording[0]) 
        brain_file_1 = [filename for filename in os.listdir(self.directory_path) if filename == self.brain_1]
        brain_state_1 = pd.read_pickle(brain_file_1[0])
        
        start_time_1 = start_times_dict[self.start_dict_1]
        end_time_1 = end_times_dict[self.end_dict_1]
        
        recording_1 = recording[:, start_time_1: end_time_1 + 1]

        return recording_1, brain_state_1 
    
    def extract_br_state(self, recording, br_state_file, br_number):
        split_epochs = np.split(recording, len(br_state_file), axis = 1)
        br_indices = br_state_file.loc[br_state_file['brainstate'] == br_number].index.to_list()
        br_epochs = np.array(split_epochs)[br_indices]
        return br_epochs
        