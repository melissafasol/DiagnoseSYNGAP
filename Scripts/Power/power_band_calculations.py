import os 
import sys
import pandas as pd 
import numpy as np

from power_band_class import PowerBands

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ID_ls, SYNGAP_2_ID_ls


directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Rat/Power/'
error_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/ConnectivityErrors/'


analysis_ls = ['S7074'] #['S7101', 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075',
              # 'S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091', 'S7101']

frequency_names = ['delta', 'theta', 'sigma', 'beta', 'gamma']
frequency_bands = [[1, 5], [5, 11], [11, 16], [16, 30], [30, 48]]
motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]
somatosensory = [0, 6, 9, 13]

for animal in analysis_ls:
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    data_loaded = []
    
    # Load data based on the group the animal belongs to
    if animal in SYNGAP_2_ID_ls:
        data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data_loaded.extend([(data_1, brain_state_1), (data_2, brain_state_2)])
    elif animal in SYNGAP_1_ID_ls:
        data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data_loaded.append((data_1, brain_state_1))
    
    print('data loaded')
        
    freq_ls = []
    for freq_name, freq_band in zip(frequency_names, frequency_bands):
        region_means = {region: [] for region in ['Motor', 'Visual', 'Soma']}
        
        for data, brain_state in data_loaded:
            noise_filter = NoiseFilter(data, brain_state_file=brain_state, channelvariables=channel_variables, ch_type='eeg')
            bandpass_filtered_data = noise_filter.filter_data_type()
            
            power_calc = PowerBands(freq_low=freq_band[0], freq_high=freq_band[1], fs=250.4, nperseg=1252)
            motor_mean, visual_mean, soma_mean = power_calc.functional_region_power_recording(bandpass_filtered_data, motor, visual, somatosensory)
            
            region_means['Motor'].append(motor_mean)
            region_means['Visual'].append(visual_mean)
            region_means['Soma'].append(soma_mean)
        
        # Combine data from multiple recordings if necessary
        for region in region_means:
            combined_mean = np.concatenate(region_means[region], axis=0) if len(region_means[region]) > 1 else region_means[region][0]
            region_means[region] = combined_mean.tolist()
        
        power_dict = {f'{region}_{freq_name}': region_means[region] for region in region_means}
        freq_ls.append(pd.DataFrame(data=power_dict))
    
    all_freqs = pd.concat(freq_ls, axis=1)
    id_df = pd.DataFrame(data={'Animal_ID': [animal]*len(all_freqs)})
    df_concat = pd.concat([all_freqs, id_df], axis=1)
    print(df_concat)
    #results_file_path = os.path.join(results_path, f'{animal}_power_all_frequency_bands.csv')
    #df_concat.to_csv(results_file_path)

    #print(f'Processed and saved data for animal {animal}')
        
    