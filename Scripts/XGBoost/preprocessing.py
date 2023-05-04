import os 
import pandas as pd 
import numpy as np 
import math 

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from constatns import GRIN_het_IDs, GRIN_wt_IDs

#directories 
dispen_direc = '/home/melissa/RESULTS/XGBoost/DispEn/'
f_direc = '/home/melissa/RESULTS/XGBoost/FOOOF/'
gamma_direc = '/home/melissa/RESULTS/XGBoost/Gamma_Power/'
br_directory = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy/'
test_list = [ '140', '402', '228', '138', '367', '363', '238', '378']

def remove_nan_indices(exponent_array, offset_array):
    exponent_nan_index = [i for i, x in enumerate(exponent_array) if math.isnan(x)]
    offset_nan_index = [i for i, x in enumerate(offset_array) if math.isnan(x)]
    nan_indices =  list(set(offset_nan_index + exponent_nan_index))
    return nan_indices

feature_df = []

for animal in test_list:
    print(animal)
    #load br file 
    br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
    br_2 = pd.read_pickle(br_directory + str(animal) + '_BL2.pkl')
    br_state_1 = br_1['brainstate'].to_numpy()
    br_state_2 = br_2['brainstate'].to_numpy()
    #load other analysis files 
    fooof_offset_1 = np.load(f_direc + str(animal) + 'offset_BR1.npy')
    fooof_offset_2 = np.load(f_direc + str(animal) + 'offset_BR2.npy')
    fooof_exp_1 = np.load(f_direc + str(animal) + 'exponent_BR1.npy')
    fooof_exp_2 = np.load(f_direc + str(animal) + 'exponent_BR1.npy')
    dispen_br1 = np.load(dispen_direc + animal + '_BR1.npy')
    dispen_br2 = np.load(dispen_direc + animal + '_BR2.npy')
    gamma_br1 = np.load(gamma_direc + animal + '_BR1.npy')
    gamma_br2 = np.load(gamma_direc + animal + '_BR2.npy')
    avg_gamma_br1 = [np.mean(epoch) for epoch in gamma_br1]
    avg_gamma_br2 = [np.mean(epoch) for epoch in gamma_br2]
    dispen = np.concatenate([dispen_br1, dispen_br2])
    gamma = np.concatenate([avg_gamma_br1, avg_gamma_br2])
    br_state = np.concatenate([br_state_1, br_state_2])
    
    fooof_offset_nan = np.concatenate([fooof_offset_1, fooof_offset_2])
    fooof_exponent_nan = np.concatenate([fooof_exp_1, fooof_exp_2]) 
    nan_indices = remove_nan_indices(fooof_offset_nan, fooof_exponent_nan)
    
    #clean arrays
    clean_offset = np.delete(fooof_offset_nan, nan_indices)
    clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
    clean_dispen = np.delete(dispen, nan_indices)
    clean_gamma = np.delete(gamma, nan_indices)
    clean_br_state = np.delete(br_state, nan_indices)
    
    
    if animal in GRIN_het_IDs:
        genotype = 1
    elif animal in GRIN_wt_IDs:
        genotype = 0
    
    
    test_dict = {'Genotype': [genotype]*len(clean_dispen), 'DispEn': clean_dispen,
                'Gamma': clean_gamma, 'Offset': clean_offset, 'Exponent': clean_exponent, 
                 'SleepStage': clean_br_state}
    test_df = pd.DataFrame(data = test_dict)
    clean_df = test_df[test_df["SleepStage"].isin([0,1,2])]
    print(clean_df)
    feature_df.append(clean_df)

feature_concat = pd.concat(feature_df)