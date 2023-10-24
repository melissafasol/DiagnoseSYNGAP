'Script to analyse results of connectivity matrices and apply graph theory calculations'

import os 
import numpy as np 
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix


from graph_preprocessing import standardise_matrix, weight_threshold_matrix

feature_measures = ['wpli', 'corr_cross_correlation', 'cross_correlation', 'imag_coherence', 'pearson_correlation',
                    'shannon_entropy', 'squared_coherence']
results_path = 

features_df_ls = []
for feature in feature_measures:
    path = '/home/melissa/RESULTS/GRAPH/' + str(feature) + '/'
    patient_df_ls = []
    for file in os.listdir(path):
        patient = file[0:6]
        os.chdir(path)
        graph_ls = np.load(file)
        feature_ls = []
        for idx, graph_matrix in enumerate(graph_ls):
            if (graph_matrix == 0.0).all():
                pass
            elif np.isnan(graph_matrix).any():
                pass
            else:
        
                normalized_matrix = standardise_matrix(graph_matrix=graph_matrix)
        
                thresholded_matrix = weight_threshold_matrix(normalized_matrix = normalized_matrix)
        
                G = nx.Graph(thresholded_matrix)
        
                # Calculate additional graph measures
      
                transitivity = nx.transitivity(G)
                global_efficiency = nx.global_efficiency(G)
                average_clustering_coefficient = nx.average_clustering(G)
            
                feature_dict = {'Patient': str(patient), 'Index': [idx], 'transitivity': [transitivity],
                       'global_efficiency': [global_efficiency], 
                       'average_clust_coeff': [average_clustering_coefficient],
                        'Feature': [feature]}
            
                feature_df = pd.DataFrame(data = feature_dict)
                feature_ls.append(feature_df)
            
        feature_concat = pd.concat(feature_ls)
        print(feature_concat)
        patient_df_ls.append(feature_concat)
    
    features_df_ls.append(pd.concat(patient_df_ls))