import os
import numpy as np 
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix

def standardise_matrix(graph_matrix):
    
    mean_value = np.mean(graph_matrix)
    std_value = np.std(graph_matrix)
    
    normalised_matrix = (graph_matrix - mean_value)/std_value
    
    return normalised_matrix

def weight_threshold_matrix(normalized_matrix):
    all_normalized_edge_weights = normalized_matrix.flatten()

    # Sort the list of normalized edge weights in descending order
    sorted_normalized_edge_weights = np.sort(all_normalized_edge_weights)[::-1]

    # Determine the threshold value (edge weight at the 70th percentile)
    threshold_index = int(0.3 * len(sorted_normalized_edge_weights))
    threshold_value = sorted_normalized_edge_weights[threshold_index]

    #Create a binary mask for edges to keep (1 for above threshold, 0 for below threshold)
    threshold_mask = (normalized_matrix >= threshold_value).astype(float)

    # Apply the binary mask to the original matrix to create the thresholded matrix
    thresholded_matrix = normalized_matrix * threshold_mask