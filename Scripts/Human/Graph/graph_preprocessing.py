import os
import numpy as np 
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix

def preprocess_graph(graph_matrix_array):
    
    for graph_matrix in graph_matrix_array:
        if (graph_matrix == 0.0).all():
            pass
        elif np.isnan(graph_matrix).any():
            pass
        else:

            # Perform Min-Max scaling to normalize edge weights between 0 and 1
            min_value = np.min(graph_matrix)
            max_value = np.max(graph_matrix)
            normalized_matrix = (graph_matrix - min_value) / (max_value - min_value)
            
             # Flatten the normalized matrix to create a list of all normalized edge weights
            all_normalized_edge_weights = normalized_matrix.flatten()
    
            # Sort the list of normalized edge weights in descending order
            sorted_normalized_edge_weights = np.sort(all_normalized_edge_weights)[::-1]
    
            # Determine the threshold value (edge weight at the 90th percentile)
            threshold_index = int(0.3 * len(sorted_normalized_edge_weights))
            threshold_value = sorted_normalized_edge_weights[threshold_index]
    
            # Create a binary mask for edges to keep (1 for above threshold, 0 for below threshold)
            threshold_mask = (normalized_matrix >= threshold_value).astype(int)
    
            # Apply the binary mask to the original matrix to create the thresholded matrix
            thresholded_matrix = graph_matrix * threshold_mask
            
            # Create a NetworkX graph from the normalized matrix
            G = nx.Graph(thresholded_matrix)
    
            # Calculate the Minimum Spanning Tree using Prim's algorithm
            mst = nx.minimum_spanning_tree(G)
    
            # Create a sparse matrix to represent the inverted MST
            inverted_mst_matrix = lil_matrix(normalized_matrix.shape, dtype=np.float64)
    
            # Iterate through the edges of the MST
            for u, v, w in mst.edges(data=True):
                inverted_mst_matrix[u, v] = 1.0
                inverted_mst_matrix[v, u] = 1.0
    
            # Subtract the MST from the normalized matrix to get the inverted MST
            inverted_mst_matrix = normalized_matrix - inverted_mst_matrix.toarray()
    
            print(inverted_mst_matrix)
    