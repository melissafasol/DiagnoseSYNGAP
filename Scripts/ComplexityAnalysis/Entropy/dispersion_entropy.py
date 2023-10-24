import os 
import sys
import numpy as np 
import pandas as pd
import EntropyHub as EH


sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
#from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

class DispersionEntropy:
    
    def __init__(self, dat_array):
        self.dat_array = dat_array
        
        
    def disp_en(self):
        disp_en_ls =  []

        for epoch in self.dat_array:
            Dispx_1, Ppi_1 = EH.DispEn(epoch, m = 3, tau = 2, c = 4, Typex = 'ncdf')
            disp_en_ls.append(Dispx_1)
        
        disp_en_array = np.array(disp_en_ls)
        
        return disp_en_array


directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/DispEn'
motor = [1,2,3,10,11,12]
somatosensory = [0, 6, 9, 13]
visual = [4, 5, 7, 8]

syngap_2_ls =  ['S7096'] #['S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091']
SYNGAP_1_ID_ls = ['S7101'] #[ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']

for animal in syngap_2_ls:
    print('somato starting 1')
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    #data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    print('data loaded')
    #only select eeg channels and filter with bandpass butterworth filter before selecting indices
    noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
    noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
    bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
    bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
    print('data filtered')
    
    filter_0_1 = np.split(bandpass_filtered_data_1[0], 17280, axis = 0)
    filter_6_1 = np.split(bandpass_filtered_data_1[6], 17280, axis = 0)
    filter_9_1 = np.split(bandpass_filtered_data_1[9], 17280, axis = 0)
    filter_13_1 = np.split(bandpass_filtered_data_1[13], 17280, axis = 0)
    #filter_11_1 = np.split(bandpass_filtered_data_1[11], 17280, axis = 0)
    #filter_12_1 = np.split(bandpass_filtered_data_1[12], 17280, axis = 0)
    
    filter_0_2 = np.split(bandpass_filtered_data_2[0], 17280, axis = 0)
    filter_6_2 = np.split(bandpass_filtered_data_2[6], 17280, axis = 0)
    filter_9_2 = np.split(bandpass_filtered_data_2[9], 17280, axis = 0)
    filter_13_2 = np.split(bandpass_filtered_data_2[13], 17280, axis = 0)
    #filter_11_2 = np.split(bandpass_filtered_data_2[11], 17280, axis = 0)
    #filter_12_2 = np.split(bandpass_filtered_data_2[12], 17280, axis = 0)
   
    disp_en_0_1 = DispersionEntropy(filter_0_1)
    disp_en_6_1 = DispersionEntropy(filter_6_1)
    disp_en_9_1 = DispersionEntropy(filter_9_1)
    disp_en_13_1 = DispersionEntropy(filter_13_1)
    #isp_en_11_1 = DispersionEntropy(filter_11_1)
    #disp_en_12_1 = DispersionEntropy(filter_12_1)
    
    disp_en_0_2 = DispersionEntropy(filter_0_2)
    disp_en_6_2 = DispersionEntropy(filter_6_2)
    disp_en_9_2 = DispersionEntropy(filter_9_2)
    disp_en_13_2 = DispersionEntropy(filter_13_2)
    #disp_en_11_2 = DispersionEntropy(filter_11_2)
    #disp_en_12_2 = DispersionEntropy(filter_12_2)
    
    dispen_array_0_1 = disp_en_0_1.disp_en() 
    dispen_array_6_1 = disp_en_6_1.disp_en() 
    dispen_array_9_1 = disp_en_9_1.disp_en() 
    dispen_array_13_1 = disp_en_13_1.disp_en() 
    #ispen_array_11_1 = disp_en_11_1.disp_en() 
    #dispen_array_12_1 = disp_en_12_1.disp_en() 
    
    print(str(animal) + 'dispersion entropy calculations br 1 complete')
    
    dispen_array_0_2 = disp_en_0_2.disp_en() 
    dispen_array_6_2 = disp_en_6_2.disp_en() 
    dispen_array_9_2 = disp_en_9_2.disp_en() 
    dispen_array_13_2 = disp_en_13_2.disp_en() 
    #dispen_array_11_2 = disp_en_11_2.disp_en() 
    #dispen_array_12_2 = disp_en_12_2.disp_en() 
    #print(str(animal) + 'dispersion entropy calculations br 2 complete')
    
    dispen_array_1 = np.mean(np.array([dispen_array_0_1, dispen_array_6_1, dispen_array_9_1,  dispen_array_13_1]), axis = 0)
    dispen_array_2 = np.mean(np.array([dispen_array_0_2, dispen_array_6_2, dispen_array_9_2,  dispen_array_13_2]), axis = 0)
    dispen_array = np.concatenate((dispen_array_1, dispen_array_2), axis = 0)
    
    os.chdir(results_path)
    np.save(str(animal) + '_dispen.npy', dispen_array)
    
    
    #filter_1_1 = np.split(bandpass_filtered_data_1[1], 17280, axis = 0)
    #filter_2_1 = np.split(bandpass_filtered_data_1[2], 17280, axis = 0)
    #filter_3_1 = np.split(bandpass_filtered_data_1[3], 17280, axis = 0)
    #filter_10_1 = np.split(bandpass_filtered_data_1[10], 17280, axis = 0)
    #filter_11_1 = np.split(bandpass_filtered_data_1[11], 17280, axis = 0)
    #filter_12_1 = np.split(bandpass_filtered_data_1[12], 17280, axis = 0)
    #
    #filter_1_2 = np.split(bandpass_filtered_data_2[1], 17280, axis = 0)
    #filter_2_2 = np.split(bandpass_filtered_data_2[2], 17280, axis = 0)
    #filter_3_2 = np.split(bandpass_filtered_data_2[3], 17280, axis = 0)
    #filter_10_2 = np.split(bandpass_filtered_data_2[10], 17280, axis = 0)
    #filter_11_2 = np.split(bandpass_filtered_data_2[11], 17280, axis = 0)
    #filter_12_2 = np.split(bandpass_filtered_data_2[12], 17280, axis = 0)
    #
    #disp_en_1_1 = DispersionEntropy(filter_1_1)
    #disp_en_2_1 = DispersionEntropy(filter_2_1)
    #disp_en_3_1 = DispersionEntropy(filter_2_1)
    #disp_en_10_1 = DispersionEntropy(filter_10_1)
    #disp_en_11_1 = DispersionEntropy(filter_11_1)
    #disp_en_12_1 = DispersionEntropy(filter_12_1)
    #
    #disp_en_1_2 = DispersionEntropy(filter_1_2)
    #disp_en_2_2 = DispersionEntropy(filter_2_2)
    #disp_en_3_2 = DispersionEntropy(filter_2_2)
    #disp_en_10_2 = DispersionEntropy(filter_10_2)
    #disp_en_11_2 = DispersionEntropy(filter_11_2)
    #disp_en_12_2 = DispersionEntropy(filter_12_2)
    #
    #dispen_array_1_1 = disp_en_1_1.disp_en() 
    #dispen_array_2_1 = disp_en_2_1.disp_en() 
    #dispen_array_3_1 = disp_en_3_1.disp_en() 
    #dispen_array_10_1 = disp_en_10_1.disp_en() 
    #dispen_array_11_1 = disp_en_11_1.disp_en() 
    #dispen_array_12_1 = disp_en_12_1.disp_en() 
    #
    #print(str(animal) + 'dispersion entropy calculations br 1 complete')
    #
    #dispen_array_1_2 = disp_en_1_2.disp_en() 
    #dispen_array_2_2 = disp_en_2_2.disp_en() 
    #dispen_array_3_2 = disp_en_3_2.disp_en() 
    #dispen_array_10_2 = disp_en_10_2.disp_en() 
    #dispen_array_11_2 = disp_en_11_2.disp_en() 
    #dispen_array_12_2 = disp_en_12_2.disp_en() 