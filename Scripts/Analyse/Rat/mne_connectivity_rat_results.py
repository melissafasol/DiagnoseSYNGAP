import os 
import pandas as pd
import numpy as np
import itertools

folder_dir = '/home/melissa/RESULTS/FINAL_MODEL/Rat/MNEConnectivity/'
#coh_dir = '/home/melissa/RESULTS/FINAL_MODEL/Rat/MNEConnectivity/coh/'
#pli_dir = '/home/melissa/RESULTS/FINAL_MODEL/Rat/MNEConnectivity/pli/'
#plv_dir = '/home/melissa/RESULTS/FINAL_MODEL/Rat/MNEConnectivity/plv/'
#wpli_dir = '/home/melissa/RESULTS/FINAL_MODEL/Rat/MNEConnectivity/wpli/'
noise_dir = '/home/melissa/PREPROCESSING/SYNGAP1/cleaned_br_files/'

SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101']
syngap_2_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086'] 
analysis_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101',
               'S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086']

frequencies = ['delta', 'theta', 'sigma', 'beta', 'gamma']

channel_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
channel_names = ['S1Tr_RIGHT', 'M2_FrA_RIGHT','M2_ant_RIGHT','M1_ant_RIGHT', 'V2ML_RIGHT',
                 'V1M_RIGHT', 'S1HL_S1FL_RIGHT', 'V1M_LEFT','V2ML_LEFT', 'S1HL_S1FL_LEFT', 'M1_ant_LEFT',
                 'M2_ant_LEFT','M2_FrA_LEFT','S1Tr_LEFT']

def channel_pairs(channels):
    pairs = []
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            pairs.append([channels[i], channels[j]])
    
    return pairs

def chan_region_combinations(motor_channels, visual_channels, somatosensory_channels):
    
    # Define channel indices for each region
    '''
    motor_channels = [1, 2, 3, 10, 11, 12]
    visual_channels = [4, 5, 7, 8]
    somatosensory_channels = [0, 6, 9, 13]
    '''

    # Initialize sets to store combinations and their positions
    motor_combinations = set()
    visual_combinations = set()
    somatosensory_combinations = set()

    # Generate combinations within the motor region
    for combo in itertools.combinations(motor_channels, 2):
        motor_combinations.add(combo)

    # Generate combinations within the visual region
    for combo in itertools.combinations(visual_channels, 2):
        visual_combinations.add(combo)

    # Generate combinations within the other region
    for combo in itertools.combinations(somatosensory_channels, 2):
        somatosensory_combinations.add(combo)
        
    return motor_combinations, visual_combinations, somatosensory_combinations

def calculate_region_avgs(conn_array, conn_idx, 
                          motor_combinations, 
                          visual_combinations, 
                          somatosensory_combinations, conn_measure, freq_band, animal_id):
    motor_only_values = []
    visual_only_values = []
    somatosensory_only_values = []
    somatosensory_motor = []
    visual_somatosensory = []
    visual_motor = []
    
    motor_channels = [1, 2, 3, 10, 11, 12]
    visual_channels = [4, 5, 7, 8]
    somatosensory_channels = [0, 6, 9, 13]
    
    
    
    for idx, value in enumerate(channel_comb_tup):
        if value in motor_combinations:
            motor_conn_value = conn_array[idx]
            motor_only_values.append(motor_conn_value)
        elif value in visual_combinations:
            visual_conn_value = conn_array[idx]
            visual_only_values.append(visual_conn_value)
        elif value in somatosensory_combinations:
            soma_conn_value = conn_array[idx]
            somatosensory_only_values.append(soma_conn_value)
        elif value[0] in motor_channels and value[1] in somatosensory_channels:
            motor_som_conn_value = conn_array[idx]
            somatosensory_motor.append(motor_som_conn_value)
        elif value[0] in somatosensory_channels and value[1] in motor_channels:
            som_motor_conn_value = conn_array[idx]
            somatosensory_motor.append(som_motor_conn_value)
        elif value[0] in visual_channels and value[1] in somatosensory_channels:
            vis_som_conn_value = conn_array[idx]
            visual_somatosensory.append(vis_som_conn_value)
        elif value[0] in somatosensory_channels and value[1] in visual_channels:
            som_visual_conn_value = conn_array[idx]
            visual_somatosensory.append(som_visual_conn_value)
        elif value[0] in visual_channels and value[1] in motor_channels:
            vis_mot_conn_value = conn_array[idx]
            visual_motor.append(vis_mot_conn_value)
        elif value[0] in motor_channels and value[1] in visual_channels:
            mot_visual_conn_value = conn_array[idx]
            visual_motor.append(mot_visual_conn_value)
            
    motor_mean = np.mean(motor_only_values)
    visual_mean = np.mean(visual_only_values)
    soma_mean = np.mean(somatosensory_only_values)
    soma_motor_mean = np.mean(somatosensory_motor)
    vis_soma_mean = np.mean(visual_somatosensory)
    vis_mot_mean = np.mean(visual_motor)
        
    conn_dict = {'Animal_ID': [animal_id], 'Idx': [conn_idx],
                 'Motor' + '_' + conn_measure + '_' + freq_band: motor_mean, 
                 'Visual' '_' + conn_measure + '_' + freq_band: visual_mean, 
                 'Somatosensory' '_' + conn_measure + '_' + freq_band: soma_mean, 
                 'Soma_Motor' + '_' + conn_measure + '_' + freq_band: soma_motor_mean, 
                    'Vis_Soma' + '_' + conn_measure + '_' + freq_band: vis_soma_mean,
                    'Vis_Mot' + '_' + conn_measure + '_' + freq_band: vis_mot_mean}
    conn_df = pd.DataFrame(data = conn_dict)
    
    return conn_df

###beginning analysis

channel_comb = channel_pairs(channel_values)
channel_comb_tup = tuple(tuple(sub) for sub in channel_comb)

motor_combinations, visual_combinations, somatosensory_combinations = chan_region_combinations(motor_channels = [1, 2, 3, 10, 11, 12],
                                                                     visual_channels = [4, 5, 7, 8],
                                                                     somatosensory_channels = [0, 6, 9, 13])

analysis_methods = ['coh', 'plv', 'wpli'] #'pli',

for analysis in analysis_methods:
    analysis_dir = folder_dir + analysis + '/'
    frequency_ls = []
    for freq_band in frequencies:
        print(freq_band)
        animal_ls = []
        for animal in analysis_ls:
            print(animal)
            if animal in syngap_2_ls:
                analysis_file = np.load(analysis_dir + animal + '_' + analysis + '_' + freq_band + '.npy')
                all_indices = np.arange(0, 34560, 1)
                br_1 = pd.read_pickle(noise_dir + animal + '_BL1.pkl')
                br_2 = pd.read_pickle(noise_dir + animal + '_BL2.pkl')
                clean_idx_1 = br_1[br_1['brainstate'].isin([0, 1, 2])].index.to_list()
                idx_2 = br_2[br_2['brainstate'].isin([0, 1, 2])].index.to_list()
                clean_idx_2 = [(idx + 17280) for idx in idx_2]
                clean_indices = clean_idx_1 + clean_idx_2
                clean_values_ls = []
                for idx in all_indices:
                    if idx in clean_indices:
                        conn_df = calculate_region_avgs(conn_array = analysis_file[idx], conn_idx = idx,
                                                        motor_combinations = motor_combinations,
                                                        visual_combinations = visual_combinations,
                                                        somatosensory_combinations = somatosensory_combinations,
                                                        conn_measure = str(analysis),
                                                        freq_band = freq_band, 
                                                        animal_id = animal)
                        clean_values_ls.append(conn_df)
                    else:
                        pass
                clean_array_2 = pd.concat(clean_values_ls)
                animal_ls.append(clean_array_2)
            if animal in SYNGAP_1_ID_ls:
                analysis_file = np.load(analysis_dir + animal + '_' + analysis + '_' + freq_band + '.npy')
                all_indices = np.arange(0, 17280, 1)
                br_1 = pd.read_pickle(noise_dir + animal + '_BL1.pkl')
                clean_idx_1 = br_1[br_1['brainstate'].isin([0, 1, 2])].index.to_list()
                clean_values_ls = []
                for idx in all_indices:
                    if idx in clean_idx_1:
                        conn_df = calculate_region_avgs(conn_array = analysis_file[idx], conn_idx = idx,
                                                        motor_combinations = motor_combinations,
                                                        visual_combinations = visual_combinations,
                                                        somatosensory_combinations = somatosensory_combinations,
                                                        conn_measure = str(analysis),
                                                        freq_band = freq_band,
                                                        animal_id = animal)
                        clean_values_ls.append(conn_df)
                    else:
                        pass  
                clean_array_1 = pd.concat(clean_values_ls)
                animal_ls.append(clean_array_1)
        animal_concat = pd.concat(animal_ls)
        frequency_ls.append(animal_concat)
        
    freq_test_concat = pd.concat(frequency_ls, axis = 1)
    df_dup = freq_test_concat.loc[:, ~freq_test_concat.columns.duplicated()]
    results_path = '/home/melissa/RESULTS/FINAL_MODEL/Rat/MNEConnectivity/'
    df_dup.to_csv(results_path + str(analysis) + '_connectivity_dataframe.csv')
    print(analysis + ' saved')