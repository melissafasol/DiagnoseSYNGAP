'''Script to calculate graph theory metrics from MNE connectivity arrays from taini recordings'''

import os 
import numpy as np 
import pandas as pd 
import networkx as nx
from scipy.sparse import lil_matrix


from graph_preprocessing import standardise_matrix, weight_threshold_matrix 

SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101']
syngap_2_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086'] 
analysis_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101',
               'S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086']
frequencies = ['delta', 'theta', 'sigma', 'beta', 'gamma']

channel_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
channel_names = ['S1Tr_RIGHT', 'M2_FrA_RIGHT','M2_ant_RIGHT','M1_ant_RIGHT', 'V2ML_RIGHT',
                 'V1M_RIGHT', 'S1HL_S1FL_RIGHT', 'V1M_LEFT','V2ML_LEFT', 'S1HL_S1FL_LEFT', 'M1_ant_LEFT',
                 'M2_ant_LEFT','M2_FrA_LEFT','S1Tr_LEFT']


def wf_closeness_centrality(data_dict):
    '''
    Function to compute Wasserman & Faus Closeness Centrality per functional
    region, given a dictionary where each key is a channel.
    Returns mean per functional region.
    '''
    # Define the regional indices
    somatosensory_indices = [0, 6, 9, 13]
    motor_indices = [1, 2, 3, 10, 11, 12]
    visual_indices = [4, 5, 7, 8]

    # Create empty DataFrames for each region
    df_somatosensory = pd.DataFrame(columns=["Somatosensory"])
    df_motor = pd.DataFrame(columns=["Motor"])
    df_visual = pd.DataFrame(columns=["Visual"])

    # Iterate through the dictionary and populate the DataFrames
    for key, value in data_dict.items():
        if key in somatosensory_indices:
            df_somatosensory.loc[key] = value
        elif key in motor_indices:
            df_motor.loc[key] = value
        elif key in visual_indices:
            df_visual.loc[key] = value


    mean_soma = df_somatosensory.mean().values
    mean_motor = df_motor.mean().values
    mean_visual = df_visual.mean().values
    
    return mean_soma, mean_motor, mean_visual


folder_dir = '/home/melissa/RESULTS/FINAL_MODEL/Rat/MNEConnectivity/'
noise_dir = '/home/melissa/PREPROCESSING/SYNGAP1/cleaned_br_files/'
results_dir = '/home/melissa/RESULTS/FINAL_MODEL/Rat/Graph_Theory/'

measures = ['coh', 'plv', 'pli', 'wpli']

for conn_measure in measures:
    for freq_band in frequencies:
        print(freq_band)
        animal_ls = []
        for animal in analysis_ls:
            print(animal)
            if animal in syngap_2_ls:
                conn_file = np.load(folder_dir + conn_measure + '/' + animal + '_' + conn_measure + '_' + freq_band + '.npy')
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
                        reshaped_matrix = conn_file[idx].flatten().reshape(14, 14)
                        normalised_matrix = standardise_matrix(reshaped_matrix)
                        threshold_matrix = weight_threshold_matrix(normalised_matrix)
                        G = nx.Graph(threshold_matrix)
                        transitivity = nx.transitivity(G)
                        glob_eff = nx.global_efficiency(G)
                        avg_clust_coeff = nx.average_clustering(G)
                        modularity = nx.community.modularity(G, [{0, 6, 9, 13}, {1, 2, 3, 10, 11, 12}, {4, 5, 7, 8}])
                        label_propagation_modularity = nx.community.modularity(G, nx.community.label_propagation_communities(G))
                        wf_closeness = nx.closeness_centrality(G, wf_improved = True)
                        mean_soma, mean_motor, mean_visual = wf_closeness_centrality(wf_closeness)
                        graph_dict = {'Idx': [idx], 
                                      'Animal_ID': [animal],
                                      'Transitivity_' + freq_band + '_' + conn_measure: [transitivity],
                                      'glob_eff_' + freq_band + '_' + conn_measure: [glob_eff],
                                      'avg_clust_coeff_' + freq_band + '_' +  conn_measure: [avg_clust_coeff],
                                      'modularity_' + freq_band + '_' + conn_measure: [modularity],
                                      'soma_wf_close_' + freq_band + '_' + conn_measure: mean_soma,
                                      'motor_wf_close_' + freq_band + '_' + conn_measure: mean_motor,
                                      'visual_wf_close_' + freq_band + '_' + conn_measure: mean_visual}
                        graph_df = pd.DataFrame(data = graph_dict)
                        clean_values_ls.append(graph_df)
                indices_concat = pd.concat(clean_values_ls)
                animal_ls.append(indices_concat)
            
            elif animal in SYNGAP_1_ID_ls:
                conn_file = np.load(folder_dir + conn_measure + '/' + animal + '_' + conn_measure + '_' + freq_band + '.npy')
                all_indices = np.arange(0, 17280, 1)
                br_1 = pd.read_pickle(noise_dir + animal + '_BL1.pkl')
                clean_idx_1 = br_1[br_1['brainstate'].isin([0, 1, 2])].index.to_list()
                clean_values_ls = []
                for idx in all_indices:
                    if idx in clean_idx_1:
                        reshaped_matrix = conn_file[idx].flatten().reshape(14, 14)
                        normalised_matrix = standardise_matrix(reshaped_matrix)
                        threshold_matrix = weight_threshold_matrix(normalised_matrix)
                        G = nx.Graph(threshold_matrix)
                        transitivity = nx.transitivity(G)
                        glob_eff = nx.global_efficiency(G)
                        avg_clust_coeff = nx.average_clustering(G)
                        modularity = nx.community.modularity(G, [{0, 6, 9, 13}, {1, 2, 3, 10, 11, 12}, {4, 5, 7, 8}])
                        label_propagation_modularity = nx.community.modularity(G, nx.community.label_propagation_communities(G))
                        wf_closeness = nx.closeness_centrality(G, wf_improved = True)
                        mean_soma, mean_motor, mean_visual = wf_closeness_centrality(wf_closeness)
                        graph_dict = {'Idx': [idx], 
                                      'Animal_ID': [animal],
                                      'Transitivity_' + freq_band + '_' + conn_measure: [transitivity],
                                      'glob_eff_' + freq_band + '_' + conn_measure: [glob_eff],
                                      'avg_clust_coeff_' + freq_band + '_' +  conn_measure: [avg_clust_coeff],
                                      'modularity_' + freq_band + '_' + conn_measure: [modularity],
                                      'soma_wf_close_' + freq_band + '_' + conn_measure: mean_soma,
                                      'motor_wf_close_' + freq_band + '_' + conn_measure: mean_motor,
                                      'visual_wf_close_' + freq_band + '_' + conn_measure: mean_visual}
                        graph_df = pd.DataFrame(data = graph_dict)
                        clean_values_ls.append(graph_df)
                indices_concat = pd.concat(clean_values_ls)
                animal_ls.append(indices_concat)
        all_freq_concat = pd.concat(animal_ls)
        all_freq_concat.to_csv(results_dir + conn_measure + '/' + conn_measure + '_' + freq_band + '.csv')
        print('all animals saved for ' + str(freq_band))