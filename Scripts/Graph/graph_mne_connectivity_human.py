'''Script to calculate graph theory metrics from MNE connectivity arrays from patient recordings'''

import os 
import numpy as np 
import pandas as pd 
import networkx as nx
from scipy.sparse import lil_matrix

from graph_preprocessing import standardise_matrix, weight_threshold_matrix 


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


def wf_closeness_centrality_human(data_dict):
    '''
    Function to compute Wasserman & Faus Closeness Centrality per functional
    region, given a dictionary where each key is a channel.
    Returns mean per functional region.
    '''
    # Define the regional indices
    ocular_indices = [0, 1]
    central_indices = [2, 3]
    occipital_indices = [4]

    # Create empty DataFrames for each region
    df_ocular = pd.DataFrame(columns=["Ocular"])
    df_central = pd.DataFrame(columns=["Central"])
    df_occipital = pd.DataFrame(columns=["Occipital"])

    # Iterate through the dictionary and populate the DataFrames
    for key, value in data_dict.items():
        if key in ocular_indices:
            df_ocular.loc[key] = value
        elif key in central_indices:
            df_central.loc[key] = value
        elif key in occipital_indices:
            df_occipital.loc[key] = value


    mean_ocular = df_ocular.mean().values
    mean_central = df_central.mean().values
    mean_occipital = df_occipital.mean().values
    
    return mean_ocular[0], mean_central[0], mean_occipital[0]


folder_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Connectivity_MNE/'
results_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/'
noise_path = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'
frequency_names = ['delta', 'theta', 'sigma', 'beta']
connectivity_measure = ['coh', 'plv', 'pli', 'wpli']


metric_ls = []
for metric in connectivity_measure:
    print(f'{metric} starting')
    frequency_ls = []
    for frequency in frequency_names:
        print(f'{frequency} starting')
        patient_ls = []
        for patient in patient_list:
            print(patient)
            patient_id = patient.split()[0]
            print(patient_id)
            genotype = genotype_human.get(patient_id, None)
            if genotype is None:
                continue

            noise_file = np.load(os.path.join(noise_path, f'{patient}_noise.npy'))
            folder_path = os.path.join(folder_dir, metric)
            patient_file = np.load(os.path.join(folder_path, f'{patient}_{frequency}.npy'))
            idx_ls = []
            for idx, value in enumerate(patient_file):
                if idx in noise_file:
                    continue
                reshaped_matrix = value.flatten().reshape(6, 6)
                normalised_matrix = standardise_matrix(reshaped_matrix)
                threshold_matrix = weight_threshold_matrix(normalised_matrix)
                G = nx.Graph(threshold_matrix)
                transitivity = nx.transitivity(G)
                glob_eff = nx.global_efficiency(G)
                avg_clust_coeff = nx.average_clustering(G)
                modularity = nx.community.modularity(G, [{0, 1}, {2, 3}, {4}, {5}])
                label_propagation_modularity = nx.community.modularity(G, nx.community.label_propagation_communities(G))
                wf_closeness = nx.closeness_centrality(G, wf_improved = True)
                mean_ocular, mean_central, mean_occipital = wf_closeness_centrality_human(wf_closeness)
                graph_dict = {'Idx': [idx], 
                                      'Patient_ID': [patient],
                                      'Genotype': [genotype],
                                      'Frequency': [frequency],
                                      'Transitivity_' + frequency + '_' + metric: [transitivity],
                                      'glob_eff_' + frequency + '_' + metric: [glob_eff],
                                      'avg_clust_coeff_' + frequency + '_' +  metric: [avg_clust_coeff],
                                      'modularity_' + frequency + '_' + metric: [modularity],
                                      'ocular_wf_close_' + frequency + '_' + metric: mean_ocular,
                                      'central_wf_close_' + frequency + '_' + metric: mean_central,
                                      'occip_wf_close_' + frequency + '_' + metric: mean_occipital}
                graph_df = pd.DataFrame(data = graph_dict)
                idx_ls.append(graph_df)
            all_indices = pd.concat(idx_ls)
            patient_ls.append(all_indices)
            
        patient_concat = pd.concat(patient_ls)
        print(patient_concat)
        patient_concat.to_csv(results_dir + metric + '/' + str(frequency) + '_graph_theory.csv')
        print(str(patient) + ' saved')